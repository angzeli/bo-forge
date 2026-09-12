"""Caller-owned fit evidence, isolation, and session-only retention."""

import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from threading import Barrier, Event, RLock, local
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

import bo_forge.models as models_module
import bo_forge.suggestions as suggestions_module
from bo_forge import CampaignSession
from bo_forge._campaign.provenance import load_manifest, manifest_path_for_log
from bo_forge._fit_metadata import FitMetadata
from bo_forge.config import ModelConfig
from bo_forge.errors import SuggestionError
from bo_forge.models import fit_gp_model, fit_multi_fidelity_gp_model, model_summary
from tests._session_support import write_config, write_log
from tests.test_models import (
    model_profile_config,
    model_profile_two_row_log,
    multi_fidelity_config,
    multi_fidelity_log,
)


@pytest.fixture
def fake_gp(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    created = []

    class FakeModel:
        likelihood = object()

        def __init__(self, train_x, train_y, **kwargs):
            self.train_x = train_x
            self.train_y = train_y
            created.append(self)

        def posterior(self, x):
            mean = torch.zeros((len(x), self.train_y.shape[-1]), dtype=torch.double)
            return SimpleNamespace(mean=mean, variance=torch.full_like(mean, 0.04))

    monkeypatch.setattr(models_module, "SingleTaskGP", FakeModel)
    monkeypatch.setattr(models_module, "SingleTaskMultiFidelityGP", FakeModel)
    monkeypatch.setattr(
        models_module, "ExactMarginalLogLikelihood", lambda _likelihood, model: model
    )
    monkeypatch.setattr(models_module, "fit_gpytorch_mll", lambda _mll: None)
    return created


def _values(summary: pd.DataFrame) -> dict[str, object]:
    return dict(zip(summary["field"], summary["value"], strict=True))


@pytest.mark.parametrize("multi_fidelity", [False, True])
def test_public_fit_returns_original_model_with_immutable_explicit_metadata(
    fake_gp: list[object], multi_fidelity: bool
) -> None:
    cfg = multi_fidelity_config() if multi_fidelity else model_profile_config("default")
    df = multi_fidelity_log(cfg) if multi_fidelity else model_profile_two_row_log(cfg)
    fit = fit_multi_fidelity_gp_model if multi_fidelity else fit_gp_model

    model = fit(cfg, df)

    assert model is fake_gp[0]
    metadata = model._bo_forge_fit_metadata
    assert isinstance(metadata, FitMetadata)
    recorded = _values(model_summary(cfg, df, metadata=metadata))
    assert recorded["last_fit_status"] == "ok"
    assert recorded["model_class"] == (
        "SingleTaskMultiFidelityGP" if multi_fidelity else "SingleTaskGP"
    )
    with pytest.raises(FrozenInstanceError):
        metadata.fields = ()
    exported = metadata.as_dict()
    assert all(value is None or isinstance(value, (str, int, bool)) for value in exported.values())
    exported["fit_status"] = "failed"
    assert metadata.as_dict()["fit_status"] == "ok"

    standalone = _values(model_summary(cfg, df))
    assert standalone["last_fit_status"] == "not_recorded"
    assert standalone["fallback_status"] == "not_recorded"
    assert standalone["last_fit_warning_count"] == 0


def test_interleaved_fits_keep_each_models_explicit_evidence(fake_gp: list[object]) -> None:
    cfg = model_profile_config("default")
    first_df = model_profile_two_row_log(cfg)
    second_df = first_df.copy(deep=True)
    second_df.loc[0, "activity"] = 3.0
    first = fit_gp_model(cfg, first_df)
    metadata = first._bo_forge_fit_metadata
    before = model_summary(cfg, first_df, metadata=metadata)

    second = fit_gp_model(cfg, second_df)

    assert first is not second
    assert first._bo_forge_fit_metadata is metadata
    assert second._bo_forge_fit_metadata != metadata
    pd.testing.assert_frame_equal(before, model_summary(cfg, first_df, metadata=metadata))
    for df, own, other in (
        (first_df, metadata, second._bo_forge_fit_metadata),
        (second_df, second._bo_forge_fit_metadata, metadata),
    ):
        assert _values(model_summary(cfg, df, metadata=own))["last_fit_status"] == "ok"
        assert _values(model_summary(cfg, df, metadata=other))["last_fit_status"] == "not_recorded"
        assert _values(model_summary(cfg, df))["last_fit_status"] == "not_recorded"


@pytest.mark.parametrize("profile", ["default", "robust"])
def test_concurrent_fits_keep_each_callers_explicit_evidence(
    fake_gp: list[object], monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    fitting = Barrier(2)
    fitted = Barrier(2)
    cfg = model_profile_config(profile)

    def fake_fit(model):
        if profile == "robust":
            warnings.warn(f"fit {float(model.train_y[0, 0])}", RuntimeWarning, stacklevel=2)
            time.sleep(0.02)

    monkeypatch.setattr(models_module, "fit_gpytorch_mll", fake_fit)

    def run_fit(value):
        df = model_profile_two_row_log(cfg)
        df.loc[0, "activity"] = value
        fitting.wait(timeout=10)
        model = fit_gp_model(cfg, df)
        # Both fits must finish before either caller asks for its evidence.
        fitted.wait(timeout=10)
        metadata = model._bo_forge_fit_metadata
        return df, metadata, _values(model_summary(cfg, df, metadata=metadata))

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = list(executor.map(run_fit, [2.0, 3.0]))

    assert first[1] != second[1]
    for own, other in ((first, second), (second, first)):
        df, metadata, summary = own
        expected_status = "ok_with_warnings" if profile == "robust" else "ok"
        assert summary["last_fit_status"] == expected_status
        assert (
            _values(model_summary(cfg, df, metadata=metadata))["last_fit_status"] == expected_status
        )
        assert summary["last_fit_warning_count"] == (1 if profile == "robust" else 0)
        if profile == "robust":
            assert summary["last_fit_warnings"] == f"fit {float(df.loc[0, 'activity'])}"
        assert (
            _values(model_summary(cfg, df, metadata=other[1]))["last_fit_status"] == "not_recorded"
        )
        assert _values(model_summary(cfg, df))["last_fit_status"] == "not_recorded"


@pytest.mark.parametrize("change", ["profile", "seed", "bounds"])
def test_explicit_metadata_is_invalidated_by_same_shape_config_changes(
    fake_gp: list[object], change: str
) -> None:
    cfg = model_profile_config("default")
    df = model_profile_two_row_log(cfg)
    metadata = fit_gp_model(cfg, df)._bo_forge_fit_metadata
    assert _values(model_summary(cfg, df, metadata=metadata))["last_fit_status"] == "ok"
    if change == "profile":
        changed = replace(cfg, model=ModelConfig(profile="smooth"))
    elif change == "seed":
        changed = replace(cfg, bo=replace(cfg.bo, random_seed=cfg.bo.random_seed + 1))
    else:
        changed = replace(cfg, variables=(replace(cfg.variables[0], upper=2.0),))

    summary = _values(model_summary(changed, df, metadata=metadata))

    assert summary["encoded_dimension"] == 1
    assert summary["observed_rows_used_for_fitting"] == 2
    assert summary["last_fit_status"] == "not_recorded"
    assert summary["fallback_status"] == "not_recorded"
    assert _values(model_summary(cfg, df, metadata=metadata))["last_fit_status"] == "ok"


@pytest.fixture
def fitted_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gp: list[object]):
    config_path = write_config(tmp_path / "campaign.yaml")
    session = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    initial = session.suggest_next(batch_size=2)
    session.append_suggestions(initial)
    for index, row_id in enumerate(initial["row_id"]):
        session.mark_observed(row_id, objective_value=float(index + 1))

    def fake_optimize(**kwargs):
        assert kwargs["model"] is fake_gp[-1]
        return torch.tensor([[0.5]], dtype=torch.double), torch.tensor([0.1]), "log_ei"

    monkeypatch.setattr(suggestions_module, "optimize_log_ei", fake_optimize)
    return session


def test_session_retains_generated_metadata_without_persisting_fit_storage(fitted_session) -> None:
    session = fitted_session
    manifest_path = manifest_path_for_log(session.log_path)
    before_manifest = manifest_path.read_bytes()
    before_log = session.log_path.read_bytes()
    before_df = session.df.copy(deep=True)
    assert _values(session.model_summary())["last_fit_status"] == "not_recorded"

    suggestions = session.suggest_next()
    metadata = session._fit_metadata

    assert isinstance(metadata, FitMetadata)
    assert "_bo_forge_fit_metadata" not in suggestions.attrs
    assert "_bo_forge_fit_metadata" not in session.df.attrs
    assert _values(session.model_summary())["last_fit_status"] == "ok"
    assert _values(model_summary(session.config, session.df))["last_fit_status"] == "not_recorded"
    pd.testing.assert_frame_equal(session.df, before_df)

    other = CampaignSession.from_files(session.config_path, session.log_path)
    assert _values(other.model_summary())["last_fit_status"] == "not_recorded"
    other.suggest_next()
    session.model_profile_comparison(profiles=["smooth"])
    assert session._fit_metadata is metadata
    assert _values(session.model_summary())["last_fit_status"] == "ok"
    assert manifest_path.read_bytes() == before_manifest
    assert session.log_path.read_bytes() == before_log

    session.append_suggestions(suggestions)
    assert _values(session.model_summary())["last_fit_status"] == "ok"
    manifest = load_manifest(session.log_path)
    assert manifest["events"][-1]["operation"] == "append_suggestions"
    assert manifest["events"][-1]["metadata"] == {"appended_row_count": 1}
    for path in (manifest_path, session.log_path):
        persisted = path.read_text(encoding="utf-8")
        for key in ("_bo_forge_fit_metadata", "training_fingerprint", "fit_status", "fit_warnings"):
            assert key not in persisted
    restarted = CampaignSession.from_files(session.config_path, session.log_path)
    assert restarted._fit_metadata is None
    assert _values(restarted.model_summary())["last_fit_status"] == "not_recorded"

    session.mark_observed(suggestions.iloc[0]["row_id"], objective_value=4.0)
    assert _values(session.model_summary())["last_fit_status"] == "not_recorded"
    assert session._fit_metadata is None


def test_session_discards_metadata_after_config_change(fitted_session, tmp_path: Path) -> None:
    log_path = write_log(tmp_path / "legacy.csv", fitted_session.config, fitted_session.df)
    session = CampaignSession.from_files(fitted_session.config_path, log_path)
    session.suggest_next()
    metadata = session._fit_metadata
    assert _values(session.model_summary())["last_fit_status"] == "ok"
    original_config = session.config
    session.config = replace(original_config, model=ModelConfig(profile="rough"))

    assert _values(session.model_summary())["last_fit_status"] == "not_recorded"
    assert session._fit_metadata is None
    session.config = original_config
    assert _values(session.model_summary())["last_fit_status"] == "not_recorded"
    assert (
        _values(model_summary(session.config, session.df, metadata=metadata))["last_fit_status"]
        == "ok"
    )


@pytest.mark.parametrize("profile", ["default", "smooth", "rough", "robust"])
def test_fit_failure_preserves_original_exception_and_warning_evidence(
    fake_gp: list[object], monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    cfg = model_profile_config(profile)
    df = model_profile_two_row_log(cfg)
    failure = RuntimeError("synthetic fit failure")

    def fail_fit(_mll):
        warnings.warn("synthetic fit warning", RuntimeWarning, stacklevel=2)
        raise failure

    monkeypatch.setattr(models_module, "fit_gpytorch_mll", fail_fit)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="synthetic fit failure") as caught:
            fit_gp_model(cfg, df)

    assert caught.value is failure
    metadata = failure._bo_forge_fit_metadata
    assert isinstance(metadata, FitMetadata)
    summary = _values(model_summary(cfg, df, metadata=metadata))
    assert summary["last_fit_status"] == "failed"
    assert summary["last_fit_warning_count"] == 1
    assert summary["last_fit_warnings"] == "synthetic fit warning"
    assert summary["fallback_status"] == "raised"
    assert _values(model_summary(cfg, df))["last_fit_status"] == "not_recorded"


def test_session_retains_failed_fit_evidence_through_translated_exception(
    fitted_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = fitted_session
    session.suggest_next()
    previous = session._fit_metadata
    failure = RuntimeError("synthetic fit failure")

    def fail_fit(_mll):
        raise failure

    def translated_suggest(config, df, **kwargs):
        try:
            fit_gp_model(config, df)
        except RuntimeError as error:
            raise SuggestionError("translated fit failure") from error

    monkeypatch.setattr(models_module, "fit_gpytorch_mll", fail_fit)
    monkeypatch.setattr(suggestions_module, "suggest_next", translated_suggest)
    with pytest.raises(SuggestionError, match="translated fit failure") as caught:
        session.suggest_next()

    assert caught.value.__cause__ is failure
    assert session._fit_metadata is failure._bo_forge_fit_metadata
    assert session._fit_metadata is not previous
    summary = _values(session.model_summary())
    assert summary["last_fit_status"] == "failed"
    assert summary["fallback_status"] == "raised"


def test_session_clears_previous_evidence_when_suggestion_fails_without_fit(fitted_session) -> None:
    session = fitted_session
    session.suggest_next()
    assert _values(session.model_summary())["last_fit_status"] == "ok"

    with pytest.raises(SuggestionError, match="batch_size must be >= 1"):
        session.suggest_next(batch_size=0)

    assert session._fit_metadata is None
    assert _values(session.model_summary())["last_fit_status"] == "not_recorded"


@pytest.mark.parametrize("profile", ["default", "robust"])
def test_concurrent_real_construction_warnings_belong_to_their_own_fit(monkeypatch, profile):
    cfg = model_profile_config(profile)
    first = model_profile_two_row_log(cfg)
    second = first.copy(deep=True)
    second["activity"] = 99.0
    fitting, second_lock_attempt = Event(), Event()
    caller = local()

    class ObservedLock:
        def __init__(self):
            self.lock = RLock()

        def __enter__(self):
            if caller.name == "second":
                second_lock_attempt.set()
            self.lock.acquire()

        def __exit__(self, *_args):
            self.lock.release()

    def fake_fit(_mll):
        if caller.name == "first":
            fitting.set()
            assert second_lock_attempt.wait(5)

    def run_fit(name, df):
        caller.name = name
        if name == "second":
            assert fitting.wait(5)
        return fit_gp_model(cfg, df)._bo_forge_fit_metadata.as_dict()

    monkeypatch.setattr(models_module, "_FIT_WARNING_LOCK", ObservedLock())
    monkeypatch.setattr(models_module, "fit_gpytorch_mll", fake_fit)
    # Construction is real: the constant-outcome model emits an InputDataWarning.
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with ThreadPoolExecutor(max_workers=2) as executor:
            a = executor.submit(run_fit, "first", first)
            b = executor.submit(run_fit, "second", second)
            first_evidence, second_evidence = a.result(), b.result()
    assert first_evidence["fit_warning_count"] == 0
    assert first_evidence["fit_status"] == "ok"
    assert second_evidence["fit_warning_count"] == 1
    assert "outcome observations" in second_evidence["fit_warnings"]
    assert second_evidence["fit_status"] == "ok_with_warnings"


@pytest.mark.parametrize("multi_fidelity", [False, True])
def test_constructor_failure_keeps_own_warning_and_releases_fit_lock(monkeypatch, multi_fidelity):
    cfg = multi_fidelity_config() if multi_fidelity else model_profile_config("default")
    df = multi_fidelity_log(cfg) if multi_fidelity else model_profile_two_row_log(cfg)
    fitter = fit_multi_fidelity_gp_model if multi_fidelity else fit_gp_model
    name = "SingleTaskMultiFidelityGP" if multi_fidelity else "SingleTaskGP"
    constructor = getattr(models_module, name)
    failure = RuntimeError("constructor failed")

    def fail_constructor(*_args, **_kwargs):
        warnings.warn("constructor warning", RuntimeWarning, stacklevel=2)
        raise failure

    monkeypatch.setattr(models_module, name, fail_constructor)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="constructor failed") as caught:
            fitter(cfg, df)
    assert caught.value is failure
    evidence = failure._bo_forge_fit_metadata.as_dict()
    assert evidence["fit_status"] == "failed"
    assert evidence["fit_warnings"] == "constructor warning"
    assert evidence["fitting_rng_fingerprint"] is None
    monkeypatch.setattr(models_module, name, constructor)
    monkeypatch.setattr(models_module, "fit_gpytorch_mll", lambda _mll: None)
    with ThreadPoolExecutor(max_workers=1) as executor:
        model = executor.submit(fitter, cfg, df).result(timeout=5)
    assert model._bo_forge_fit_metadata.as_dict()["fit_status"] == "ok"


def test_comparison_posterior_failure_keeps_warnings_without_leaking_to_next_profile(monkeypatch):
    cfg = model_profile_config("robust")
    df = model_profile_two_row_log(cfg)
    metadata = FitMetadata((("fit_warning_count", 2), ("fit_warnings", "own fitting warnings")))

    def fail_posterior(_x):
        raise RuntimeError("posterior failure")

    def fit(config, _df):
        if config.model.profile == "default":
            raise RuntimeError("later fit failed before recording evidence")
        return SimpleNamespace(_bo_forge_fit_metadata=metadata, posterior=fail_posterior)

    monkeypatch.setattr(models_module, "fit_gp_model", fit)
    result = models_module.model_profile_comparison(cfg, df, ["robust", "default"])
    assert result.model_profile.tolist() == ["robust", "default"]
    assert result.fit_status.tolist() == ["failed", "failed"]
    assert result.fit_warning_count.tolist() == [2, 0]
    assert result.fit_message.tolist() == [
        "posterior failure", "later fit failed before recording evidence",
    ]
