import pandas as pd
import pytest
import torch

import bo_forge.models as models_module
from bo_forge.config import (
    BOConfig,
    CampaignConfig,
    FidelityConfig,
    ModelConfig,
    ObjectiveConfig,
    ReplicateConfig,
    VariableConfig,
)
from bo_forge.errors import ConfigError
from bo_forge.models import (
    dataframe_to_training_tensors,
    fit_gp_model,
    fit_multi_fidelity_gp_model,
    model_profile_comparison,
    model_summary,
)
from bo_forge.validation import canonical_columns


def replicate_config() -> CampaignConfig:
    return CampaignConfig(
        campaign_name="replicate_model",
        objective=ObjectiveConfig("activity", "maximize"),
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=BOConfig(batch_size=1, initial_design_size=1),
        replicates=ReplicateConfig(enabled=True),
    )


def replicate_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "rep_0a",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "replicate_group": "group_0",
                "replicate_index": 0,
                "x": 0.2,
                "activity": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "rep_0b",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "replicate_group": "group_0",
                "replicate_index": 1,
                "x": 0.2,
                "activity": 1.4,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def multi_replicate_config() -> CampaignConfig:
    return CampaignConfig(
        campaign_name="multi_replicate_model",
        objective=ObjectiveConfig("yield_score", "maximize", 0.0),
        objectives=(
            ObjectiveConfig("yield_score", "maximize", 0.0),
            ObjectiveConfig("waste_score", "minimize", 1.0),
        ),
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=BOConfig(batch_size=1, initial_design_size=1, acquisition="qlog_ehvi"),
        replicates=ReplicateConfig(enabled=True),
    )


def multi_replicate_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "rep_0a",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "replicate_group": "group_0",
                "replicate_index": 0,
                "x": 0.2,
                "yield_score": 0.6,
                "waste_score": 0.4,
                "predicted_mean_yield_score": "",
                "predicted_std_yield_score": "",
                "predicted_mean_waste_score": "",
                "predicted_std_waste_score": "",
                "acquisition": "",
            },
            {
                "row_id": "rep_0b",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "replicate_group": "group_0",
                "replicate_index": 1,
                "x": 0.2,
                "yield_score": 0.8,
                "waste_score": 0.2,
                "predicted_mean_yield_score": "",
                "predicted_std_yield_score": "",
                "predicted_mean_waste_score": "",
                "predicted_std_waste_score": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def multi_fidelity_config() -> CampaignConfig:
    return CampaignConfig(
        campaign_name="multi_fidelity_model",
        objective=ObjectiveConfig("activity", "maximize"),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("fidelity", "continuous", 0.2, 1.0),
        ),
        bo=BOConfig(batch_size=1, initial_design_size=1, acquisition="qmf_kg"),
        fidelity=FidelityConfig(variable="fidelity", target=1.0),
    )


def multi_fidelity_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "mf_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "x": 0.2,
                "fidelity": 0.4,
                "activity": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "mf_1",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "x": 0.8,
                "fidelity": 1.0,
                "activity": 1.6,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def model_profile_config(profile: str) -> CampaignConfig:
    return CampaignConfig(
        campaign_name=f"{profile}_profile_model",
        objective=ObjectiveConfig("activity", "maximize"),
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=BOConfig(batch_size=1, initial_design_size=1),
        model=ModelConfig(profile=profile),
    )


def model_profile_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "model_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "x": 0.2,
                "activity": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def model_profile_two_row_log(cfg: CampaignConfig) -> pd.DataFrame:
    df = model_profile_log(cfg)
    extra = pd.DataFrame(
        [
            {
                "row_id": "model_1",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "x": 0.8,
                "activity": 1.8,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )
    return pd.concat([df, extra], ignore_index=True)


def patch_fake_gp(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePosterior:
        def __init__(self, mean: torch.Tensor) -> None:
            self.mean = mean
            self.variance = torch.full_like(mean, 0.04)

    class FakeModel:
        def __init__(self, train_y: torch.Tensor) -> None:
            self.likelihood = object()
            self._train_y = train_y

        def posterior(self, _train_x: torch.Tensor) -> FakePosterior:
            return FakePosterior(self._train_y)

    def fake_single_task_gp(
        _train_x: torch.Tensor,
        train_y: torch.Tensor,
        *_args: object,
        **_kwargs: object,
    ) -> FakeModel:
        return FakeModel(train_y)

    monkeypatch.setattr(models_module, "SingleTaskGP", fake_single_task_gp)
    monkeypatch.setattr(models_module, "ExactMarginalLogLikelihood", lambda *_args: object())
    monkeypatch.setattr(models_module, "fit_gpytorch_mll", lambda *_args: None)


def test_dataframe_to_training_tensors_includes_replicate_yvar() -> None:
    cfg = replicate_config()

    tensors = dataframe_to_training_tensors(cfg, replicate_log(cfg))

    assert tensors.train_x.shape == torch.Size([1, 1])
    assert tensors.train_y.squeeze(-1).tolist() == pytest.approx([1.2])
    assert tensors.train_yvar is not None
    assert tensors.train_yvar.squeeze(-1).tolist() == pytest.approx([0.04])


def test_dataframe_to_training_tensors_supports_multi_objective_yvar() -> None:
    cfg = multi_replicate_config()

    tensors = dataframe_to_training_tensors(cfg, multi_replicate_log(cfg))

    assert tensors.train_y.shape == torch.Size([1, 2])
    assert tensors.train_y.squeeze(0).tolist() == pytest.approx([0.7, -0.3])
    assert tensors.train_yvar is not None
    assert tensors.train_yvar.squeeze(0).tolist() == pytest.approx([0.01, 0.01])


def test_dataframe_to_training_tensors_keeps_learned_noise_without_repeats() -> None:
    cfg = replicate_config()
    df = replicate_log(cfg).iloc[[0]].copy()

    tensors = dataframe_to_training_tensors(cfg, df)

    assert tensors.train_yvar is None


def test_fit_gp_model_passes_train_yvar_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = replicate_config()
    captured: dict[str, object] = {}

    class FakeModel:
        likelihood = object()

    def fake_single_task_gp(*args: object, **kwargs: object) -> FakeModel:
        captured["kwargs"] = kwargs
        return FakeModel()

    monkeypatch.setattr(models_module, "SingleTaskGP", fake_single_task_gp)
    monkeypatch.setattr(models_module, "ExactMarginalLogLikelihood", lambda *_args: object())
    monkeypatch.setattr(models_module, "fit_gpytorch_mll", lambda *_args: None)

    fit_gp_model(cfg, replicate_log(cfg))

    assert "train_Yvar" in captured["kwargs"]
    assert captured["kwargs"]["train_Yvar"].squeeze(-1).tolist() == pytest.approx([0.04])


def test_fit_gp_model_omits_train_yvar_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = replicate_config()
    captured: dict[str, object] = {}

    class FakeModel:
        likelihood = object()

    def fake_single_task_gp(*args: object, **kwargs: object) -> FakeModel:
        captured["kwargs"] = kwargs
        return FakeModel()

    monkeypatch.setattr(models_module, "SingleTaskGP", fake_single_task_gp)
    monkeypatch.setattr(models_module, "ExactMarginalLogLikelihood", lambda *_args: object())
    monkeypatch.setattr(models_module, "fit_gpytorch_mll", lambda *_args: None)

    fit_gp_model(cfg, replicate_log(cfg).iloc[[0]].copy())

    assert "train_Yvar" not in captured["kwargs"]


@pytest.mark.parametrize(
    ("profile", "kernel_name"),
    [("smooth", "RBFKernel"), ("rough", "MaternKernel")],
)
def test_fit_gp_model_passes_profile_covariance_module(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    kernel_name: str,
) -> None:
    cfg = model_profile_config(profile)
    captured: dict[str, object] = {}

    class FakeModel:
        likelihood = object()

    def fake_single_task_gp(*args: object, **kwargs: object) -> FakeModel:
        captured["kwargs"] = kwargs
        return FakeModel()

    monkeypatch.setattr(models_module, "SingleTaskGP", fake_single_task_gp)
    monkeypatch.setattr(models_module, "ExactMarginalLogLikelihood", lambda *_args: object())
    monkeypatch.setattr(models_module, "fit_gpytorch_mll", lambda *_args: None)

    fit_gp_model(cfg, model_profile_log(cfg))

    covar_module = captured["kwargs"]["covar_module"]
    assert covar_module.__class__.__name__ == "ScaleKernel"
    assert covar_module.base_kernel.__class__.__name__ == kernel_name


def test_model_summary_reports_profile_and_fit_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = model_profile_config("robust")

    class FakeModel:
        likelihood = object()

    monkeypatch.setattr(models_module, "SingleTaskGP", lambda *_args, **_kwargs: FakeModel())
    monkeypatch.setattr(models_module, "ExactMarginalLogLikelihood", lambda *_args: object())
    monkeypatch.setattr(models_module, "fit_gpytorch_mll", lambda *_args: None)

    df = model_profile_log(cfg)
    fit_gp_model(cfg, df)

    summary = model_summary(cfg, df)
    values = dict(zip(summary["field"], summary["value"], strict=True))

    assert values["model_profile"] == "robust"
    assert values["model_class"] == "SingleTaskGP"
    assert values["covariance_profile"] == "default/robust"
    assert values["observed_rows_used_for_fitting"] == 1
    assert values["last_fit_status"] == "ok"


def test_model_summary_ignores_stale_fit_metadata_when_log_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = model_profile_config("smooth")

    class FakeModel:
        likelihood = object()

    monkeypatch.setattr(models_module, "SingleTaskGP", lambda *_args, **_kwargs: FakeModel())
    monkeypatch.setattr(models_module, "ExactMarginalLogLikelihood", lambda *_args: object())
    monkeypatch.setattr(models_module, "fit_gpytorch_mll", lambda *_args: None)

    df = model_profile_log(cfg)
    fit_gp_model(cfg, df)
    changed_df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "row_id": "model_1",
                        "iteration": 1,
                        "status": "observed",
                        "source": "manual",
                        "x": 0.8,
                        "activity": 1.4,
                        "predicted_mean": "",
                        "predicted_std": "",
                        "acquisition": "",
                    }
                ],
                columns=canonical_columns(cfg),
            ),
        ],
        ignore_index=True,
    )

    summary = model_summary(cfg, changed_df)
    values = dict(zip(summary["field"], summary["value"], strict=True))

    assert values["observed_rows_used_for_fitting"] == 2
    assert values["last_fit_status"] == "not_recorded"
    assert values["fallback_status"] == "not_recorded"


@pytest.mark.parametrize(
    ("column", "value"),
    [("x", 0.8), ("activity", 1.8)],
)
def test_model_summary_ignores_stale_fit_metadata_when_same_shape_values_change(
    monkeypatch: pytest.MonkeyPatch,
    column: str,
    value: float,
) -> None:
    cfg = model_profile_config("smooth")

    class FakeModel:
        likelihood = object()

    monkeypatch.setattr(models_module, "SingleTaskGP", lambda *_args, **_kwargs: FakeModel())
    monkeypatch.setattr(models_module, "ExactMarginalLogLikelihood", lambda *_args: object())
    monkeypatch.setattr(models_module, "fit_gpytorch_mll", lambda *_args: None)

    df = model_profile_log(cfg)
    fit_gp_model(cfg, df)
    changed_df = df.copy()
    changed_df.loc[0, column] = value

    summary = model_summary(cfg, changed_df)
    values = dict(zip(summary["field"], summary["value"], strict=True))

    assert values["observed_rows_used_for_fitting"] == 1
    assert values["last_fit_status"] == "not_recorded"


def test_model_summary_ignores_stale_fit_metadata_when_same_shape_yvar_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replicate_config()

    class FakeModel:
        likelihood = object()

    monkeypatch.setattr(models_module, "SingleTaskGP", lambda *_args, **_kwargs: FakeModel())
    monkeypatch.setattr(models_module, "ExactMarginalLogLikelihood", lambda *_args: object())
    monkeypatch.setattr(models_module, "fit_gpytorch_mll", lambda *_args: None)

    df = replicate_log(cfg)
    fit_gp_model(cfg, df)
    changed_df = df.copy()
    changed_df.loc[0, "activity"] = 0.9
    changed_df.loc[1, "activity"] = 1.5

    summary = model_summary(cfg, changed_df)
    values = dict(zip(summary["field"], summary["value"], strict=True))

    assert values["observed_rows_used_for_fitting"] == 1
    assert values["train_yvar_used"] is True
    assert values["last_fit_status"] == "not_recorded"


def test_model_summary_ignores_stale_fit_metadata_when_config_shape_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = model_profile_config("rough")
    wider_cfg = CampaignConfig(
        campaign_name=cfg.campaign_name,
        objective=cfg.objective,
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("z", "continuous", 0.0, 1.0),
        ),
        bo=cfg.bo,
        model=cfg.model,
    )

    class FakeModel:
        likelihood = object()

    monkeypatch.setattr(models_module, "SingleTaskGP", lambda *_args, **_kwargs: FakeModel())
    monkeypatch.setattr(models_module, "ExactMarginalLogLikelihood", lambda *_args: object())
    monkeypatch.setattr(models_module, "fit_gpytorch_mll", lambda *_args: None)

    fit_gp_model(cfg, model_profile_log(cfg))
    wider_df = pd.DataFrame(
        [
            {
                "row_id": "model_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "x": 0.2,
                "z": 0.6,
                "activity": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(wider_cfg),
    )

    summary = model_summary(wider_cfg, wider_df)
    values = dict(zip(summary["field"], summary["value"], strict=True))

    assert values["encoded_dimension"] == 2
    assert values["last_fit_status"] == "not_recorded"


def test_model_profile_comparison_returns_default_profiles_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = model_profile_config("smooth")
    df = model_profile_two_row_log(cfg)
    original = df.copy(deep=True)
    patch_fake_gp(monkeypatch)

    comparison = model_profile_comparison(cfg, df)

    assert comparison["model_profile"].tolist() == ["default", "smooth", "rough", "robust"]
    assert comparison["fit_message"].tolist() == ["", "", "", ""]
    assert set(comparison["fit_status"]) == {"ok"}
    assert comparison["observed_rows_used_for_fitting"].tolist() == [2, 2, 2, 2]
    assert comparison["encoded_dimension"].tolist() == [1, 1, 1, 1]
    assert comparison["rmse_model_space"].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert comparison["mae_model_space"].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert comparison["mean_predicted_std"].tolist() == pytest.approx([0.2, 0.2, 0.2, 0.2])
    pd.testing.assert_frame_equal(df, original)


def test_model_profile_comparison_preserves_custom_profile_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = model_profile_config("default")
    patch_fake_gp(monkeypatch)

    comparison = model_profile_comparison(
        cfg,
        model_profile_two_row_log(cfg),
        profiles=["smooth", "default", "rough"],
    )

    assert comparison["model_profile"].tolist() == ["smooth", "default", "rough"]


def test_model_profile_comparison_rejects_duplicate_profile_names() -> None:
    cfg = model_profile_config("default")

    with pytest.raises(ConfigError, match="Duplicate model profile requested: smooth"):
        model_profile_comparison(
            cfg,
            model_profile_two_row_log(cfg),
            profiles=["smooth", "rough", "smooth"],
        )


def test_model_profile_comparison_rejects_unknown_profile_names() -> None:
    cfg = model_profile_config("default")

    with pytest.raises(ConfigError, match="Unknown model profile 'experimental'"):
        model_profile_comparison(
            cfg,
            model_profile_two_row_log(cfg),
            profiles=["experimental"],
        )


def test_model_profile_comparison_marks_insufficient_observed_without_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = model_profile_config("default")

    def fail_fit(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fit should not run for insufficient rows")

    monkeypatch.setattr(models_module, "fit_gp_model", fail_fit)

    comparison = model_profile_comparison(cfg, model_profile_log(cfg), profiles=["default"])

    assert comparison.loc[0, "fit_status"] == "insufficient_observed"
    assert "At least two fitting rows" in comparison.loc[0, "fit_message"]
    assert comparison.loc[0, "observed_rows_used_for_fitting"] == 1
    assert pd.isna(comparison.loc[0, "rmse_model_space"])


def test_model_profile_comparison_restores_model_summary_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = model_profile_config("robust")
    df = model_profile_two_row_log(cfg)
    patch_fake_gp(monkeypatch)
    fit_gp_model(cfg, df)

    before_summary = model_summary(cfg, df)
    before = dict(zip(before_summary["field"], before_summary["value"], strict=True))
    comparison = model_profile_comparison(cfg, df, profiles=["smooth"])
    after_summary = model_summary(cfg, df)
    after = dict(zip(after_summary["field"], after_summary["value"], strict=True))

    assert comparison["model_profile"].tolist() == ["smooth"]
    assert before["model_profile"] == "robust"
    assert before["last_fit_status"] == "ok"
    assert after["model_profile"] == "robust"
    assert after["last_fit_status"] == "ok"


def test_model_profile_comparison_restores_metadata_after_profile_fit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = model_profile_config("robust")
    df = model_profile_two_row_log(cfg)
    patch_fake_gp(monkeypatch)
    fit_gp_model(cfg, df)

    before_summary = model_summary(cfg, df)
    before = dict(zip(before_summary["field"], before_summary["value"], strict=True))
    training = dataframe_to_training_tensors(model_profile_config("smooth"), df)

    def fail_after_metadata(profile_config: CampaignConfig, _observed: pd.DataFrame) -> object:
        models_module._record_fit_metadata(
            profile_config,
            training,
            model_class="SingleTaskGP",
            covariance_profile="RBF/ARD",
            fit_status="failed",
            fit_warnings=[],
            fallback_status="raised",
        )
        raise RuntimeError("synthetic profile fit failure")

    monkeypatch.setattr(models_module, "fit_gp_model", fail_after_metadata)

    comparison = model_profile_comparison(cfg, df, profiles=["smooth"])
    after_summary = model_summary(cfg, df)
    after = dict(zip(after_summary["field"], after_summary["value"], strict=True))

    assert comparison.loc[0, "model_profile"] == "smooth"
    assert comparison.loc[0, "fit_status"] == "failed"
    assert comparison.loc[0, "fit_message"] == "synthetic profile fit failure"
    assert before["model_profile"] == "robust"
    assert before["last_fit_status"] == "ok"
    assert after["model_profile"] == "robust"
    assert after["last_fit_status"] == "ok"


def test_model_profile_comparison_reports_replicate_train_yvar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replicate_config()
    df = pd.concat(
        [
            replicate_log(cfg),
            pd.DataFrame(
                [
                    {
                        "row_id": "rep_1a",
                        "iteration": 1,
                        "status": "observed",
                        "source": "manual",
                        "replicate_group": "group_1",
                        "replicate_index": 0,
                        "x": 0.8,
                        "activity": 2.0,
                        "predicted_mean": "",
                        "predicted_std": "",
                        "acquisition": "",
                    },
                    {
                        "row_id": "rep_1b",
                        "iteration": 1,
                        "status": "observed",
                        "source": "manual",
                        "replicate_group": "group_1",
                        "replicate_index": 1,
                        "x": 0.8,
                        "activity": 2.4,
                        "predicted_mean": "",
                        "predicted_std": "",
                        "acquisition": "",
                    },
                ],
                columns=canonical_columns(cfg),
            ),
        ],
        ignore_index=True,
    )
    patch_fake_gp(monkeypatch)

    comparison = model_profile_comparison(cfg, df, profiles=["default"])

    assert comparison.loc[0, "fit_status"] == "ok"
    assert comparison.loc[0, "observed_rows_used_for_fitting"] == 2
    assert bool(comparison.loc[0, "train_yvar_used"]) is True


def test_model_profile_comparison_rejects_unsupported_campaign_kinds() -> None:
    cfg = multi_fidelity_config()

    with pytest.raises(ConfigError, match="multi-fidelity"):
        model_profile_comparison(cfg, multi_fidelity_log(cfg))


def test_fit_multi_fidelity_gp_model_uses_fidelity_feature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = multi_fidelity_config()
    captured: dict[str, object] = {}

    class FakeModel:
        likelihood = object()

    def fake_multi_fidelity_gp(*args: object, **kwargs: object) -> FakeModel:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeModel()

    monkeypatch.setattr(
        models_module,
        "SingleTaskMultiFidelityGP",
        fake_multi_fidelity_gp,
    )
    monkeypatch.setattr(models_module, "ExactMarginalLogLikelihood", lambda *_args: object())
    monkeypatch.setattr(models_module, "fit_gpytorch_mll", lambda *_args: None)

    fit_multi_fidelity_gp_model(cfg, multi_fidelity_log(cfg))

    assert captured["kwargs"]["data_fidelities"] == [1]
    assert "train_Yvar" not in captured["kwargs"]
    assert captured["args"][0].shape == torch.Size([2, 2])
    assert captured["args"][1].shape == torch.Size([2, 1])
