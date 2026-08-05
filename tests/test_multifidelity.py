from dataclasses import replace
from typing import get_type_hints

import pandas as pd
import pytest
import torch

import bo_forge.acquisition as acquisition_module
from bo_forge.acquisition import _extract_qmf_kg_candidates
from bo_forge.config import (
    BOConfig,
    CampaignConfig,
    FidelityConfig,
    ObjectiveConfig,
    ReviewConfig,
    VariableConfig,
)
from bo_forge.multifidelity import (
    affine_fidelity_cost_model,
    fidelity_feature_index,
    fidelity_level_fixed_features,
    fidelity_level_unit_values,
    fidelity_summary,
    fidelity_variable_index,
    map_initial_fidelity_to_levels,
    target_fidelities,
    target_fidelity_projection,
    target_fidelity_unit_value,
)
from bo_forge.validation import canonical_columns


def test_target_fidelity_optimizer_forwards_maxiter_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config()
    assert cfg.fidelity is not None
    cfg = replace(
        cfg,
        fidelity=replace(
            cfg.fidelity,
            optimizer_maxiter=37,
            optimizer_timeout_seconds=9.0,
        ),
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(acquisition_module, "PosteriorMean", lambda model: object())

    def fake_optimize(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        captured.update(kwargs)
        return torch.zeros((1, 2), dtype=torch.double), torch.tensor(1.0)

    monkeypatch.setattr(acquisition_module, "optimize_acqf", fake_optimize)

    result = acquisition_module.optimize_posterior_mean_at_target_fidelity(
        cfg,
        object(),
        model_dim=2,
        timeout_seconds=4.25,
    )

    assert result.tolist() == pytest.approx([1.0])
    assert captured["options"] == {"batch_limit": 5, "maxiter": 37}
    assert captured["timeout_sec"] == pytest.approx(4.25)


def test_multifidelity_runtime_type_hints_resolve_after_lazy_import_hardening() -> None:
    mapping_hints = get_type_hints(map_initial_fidelity_to_levels)
    projection_hints = get_type_hints(target_fidelity_projection)
    cost_hints = get_type_hints(affine_fidelity_cost_model)

    assert mapping_hints["config"] is CampaignConfig
    assert "x_unit" in mapping_hints
    assert "return" in mapping_hints
    assert projection_hints["config"] is CampaignConfig
    assert "return" in projection_hints
    assert cost_hints["config"] is CampaignConfig
    assert "return" in cost_hints


@pytest.mark.parametrize("discrete", [False, True])
def test_qmfkg_optimizer_forwards_maxiter_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
    discrete: bool,
) -> None:
    cfg = discrete_config() if discrete else config()
    assert cfg.fidelity is not None
    cfg = replace(cfg, fidelity=replace(cfg.fidelity, optimizer_maxiter=41))
    captured: dict[str, object] = {}

    class FakeAcquisition:
        def extract_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
            return candidates

    monkeypatch.setattr(
        acquisition_module,
        "qMultiFidelityKnowledgeGradient",
        lambda **_kwargs: FakeAcquisition(),
    )
    monkeypatch.setattr(
        acquisition_module,
        "InverseCostWeightedUtility",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        acquisition_module,
        "affine_fidelity_cost_model",
        lambda _config: object(),
    )
    monkeypatch.setattr(
        acquisition_module,
        "target_fidelity_projection",
        lambda _config: object(),
    )

    def fake_optimize(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        captured.update(kwargs)
        return torch.zeros((2, 2), dtype=torch.double), torch.tensor(0.5)

    optimizer_name = "optimize_acqf_mixed" if discrete else "optimize_acqf"
    monkeypatch.setattr(acquisition_module, optimizer_name, fake_optimize)

    acquisition_module.optimize_qmf_kg(
        cfg,
        object(),
        torch.tensor([0.0]),
        batch_size=2,
        model_dim=2,
        fixed_features_list=[{1: 0.0}, {1: 1.0}] if discrete else None,
        timeout_seconds=3.5,
    )

    assert captured["options"] == {"batch_limit": 5, "maxiter": 41}
    assert captured["timeout_sec"] == pytest.approx(3.5)


def config() -> CampaignConfig:
    return CampaignConfig(
        campaign_name="multi_fidelity",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("fidelity", "continuous", 0.2, 1.0),
        ),
        bo=BOConfig(batch_size=1, initial_design_size=3, acquisition="qmf_kg"),
        fidelity=FidelityConfig(
            variable="fidelity",
            target=1.0,
            fixed_cost=0.01,
            fidelity_cost_weight=2.0,
        ),
    )


def discrete_config() -> CampaignConfig:
    cfg = config()
    return CampaignConfig(
        campaign_name=cfg.campaign_name,
        objective=cfg.objective,
        variables=cfg.variables,
        bo=BOConfig(batch_size=4, initial_design_size=4, acquisition="qmf_kg"),
        fidelity=FidelityConfig(
            variable="fidelity",
            target=1.0,
            fixed_cost=0.01,
            fidelity_cost_weight=2.0,
            levels=(0.25, 0.5, 0.75, 1.0),
        ),
    )


def observed_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "low_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "x": 0.2,
                "fidelity": 0.4,
                "activity": 0.9,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "target_0",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "x": 0.5,
                "fidelity": 1.0 - 1e-10,
                "activity": 1.2,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "target_1",
                "iteration": 2,
                "status": "observed",
                "source": "manual",
                "x": 0.8,
                "fidelity": 1.0,
                "activity": 1.8,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "pending_0",
                "iteration": 3,
                "status": "suggested",
                "source": "qmf_kg",
                "x": 0.6,
                "fidelity": 0.7,
                "activity": "",
                "predicted_mean": 1.4,
                "predicted_std": 0.1,
                "acquisition": 0.2,
            },
        ],
        columns=canonical_columns(cfg),
    )


def value(summary: pd.DataFrame, field: str) -> object:
    return summary.loc[summary["field"] == field, "value"].iloc[0]


def test_fidelity_indices_and_target_mapping() -> None:
    cfg = config()

    assert fidelity_variable_index(cfg) == 1
    assert fidelity_feature_index(cfg) == 1
    assert target_fidelity_unit_value(cfg) == 1.0
    assert target_fidelities(cfg) == {1: 1.0}


def test_discrete_fidelity_levels_map_to_model_space_and_fixed_features() -> None:
    cfg = discrete_config()

    assert fidelity_level_unit_values(cfg) == pytest.approx(
        (0.0625, 0.375, 0.6875, 1.0)
    )
    assert fidelity_level_fixed_features(cfg) == [
        {1: pytest.approx(0.0625)},
        {1: pytest.approx(0.375)},
        {1: pytest.approx(0.6875)},
        {1: pytest.approx(1.0)},
    ]


def test_initial_fidelity_coordinates_map_to_equal_width_level_bins() -> None:
    cfg = discrete_config()
    unit = torch.tensor(
        [[0.1, 0.0], [0.2, 0.249], [0.3, 0.25], [0.4, 0.999]],
        dtype=torch.double,
    )

    mapped = map_initial_fidelity_to_levels(cfg, unit)

    assert torch.equal(mapped[:, 0], unit[:, 0])
    assert mapped[:, 1].tolist() == pytest.approx([0.0625, 0.0625, 0.375, 1.0])


def test_target_fidelity_projection_sets_fidelity_feature() -> None:
    cfg = config()
    projection = target_fidelity_projection(cfg)
    x = torch.tensor([[[0.25, 0.3], [0.75, 0.6]]], dtype=torch.double)

    projected = projection(x)

    assert torch.allclose(projected[..., 0], x[..., 0])
    assert torch.allclose(projected[..., 1], torch.ones_like(projected[..., 1]))


def test_affine_fidelity_cost_model_uses_unit_fidelity_feature() -> None:
    cfg = config()
    cost_model = affine_fidelity_cost_model(cfg)
    x = torch.tensor([[[0.25, 0.5], [0.75, 1.0]]], dtype=torch.double)

    costs = cost_model(x)

    assert costs.shape == torch.Size([1, 2, 1])
    assert float(costs[0, 0, 0]) == pytest.approx(1.01)
    assert float(costs[0, 1, 0]) == pytest.approx(2.01)


def test_fidelity_summary_reports_counts_best_rows_and_pending_qmfkg() -> None:
    cfg = config()

    summary = fidelity_summary(cfg, observed_log(cfg))

    assert value(summary, "fidelity_variable") == "fidelity"
    assert value(summary, "target_fidelity") == pytest.approx(1.0)
    assert value(summary, "observed_rows") == 3
    assert value(summary, "lower_fidelity_observed_rows") == 1
    assert value(summary, "target_fidelity_observed_rows") == 2
    assert value(summary, "min_observed_fidelity") == pytest.approx(0.4)
    assert value(summary, "max_observed_fidelity") == pytest.approx(1.0)
    assert value(summary, "pending_qmfkg_suggestions") == 1
    assert value(summary, "best_observed_row_id") == "target_1"
    assert value(summary, "best_observed_objective") == pytest.approx(1.8)
    assert value(summary, "best_target_fidelity_row_id") == "target_1"
    assert value(summary, "best_target_fidelity_objective") == pytest.approx(1.8)
    assert value(summary, "fidelity_mode") == "continuous"
    assert value(summary, "configured_fidelity_levels") is None
    assert value(summary, "configured_fidelity_level_count") is None
    assert value(summary, "observed_fidelity_level_count") is None


def test_discrete_fidelity_summary_appends_level_fields_without_reordering() -> None:
    cfg = discrete_config()
    df = observed_log(cfg)
    df["fidelity"] = [0.25, 0.5, 1.0, 0.75]

    summary = fidelity_summary(cfg, df)

    assert summary["field"].tolist()[:12] == [
        "fidelity_variable",
        "target_fidelity",
        "observed_rows",
        "lower_fidelity_observed_rows",
        "target_fidelity_observed_rows",
        "min_observed_fidelity",
        "max_observed_fidelity",
        "pending_qmfkg_suggestions",
        "best_observed_row_id",
        "best_observed_objective",
        "best_target_fidelity_row_id",
        "best_target_fidelity_objective",
    ]
    assert value(summary, "fidelity_mode") == "discrete"
    assert value(summary, "configured_fidelity_levels") == "0.25, 0.5, 0.75, 1"
    assert value(summary, "configured_fidelity_level_count") == 4
    assert value(summary, "observed_fidelity_level_count") == 3


def test_fidelity_summary_counts_only_blocking_review_qmfkg_suggestions() -> None:
    cfg = config()
    cfg = CampaignConfig(
        campaign_name=cfg.campaign_name,
        objective=cfg.objective,
        variables=cfg.variables,
        bo=cfg.bo,
        fidelity=cfg.fidelity,
        review=ReviewConfig(enabled=True),
    )
    rows = [
        {
            "row_id": "target_0",
            "iteration": 0,
            "status": "observed",
            "source": "manual",
            "review_status": "accepted",
            "review_note": "",
            "x": 0.5,
            "fidelity": 1.0,
            "activity": 1.2,
            "predicted_mean": "",
            "predicted_std": "",
            "acquisition": "",
        },
        {
            "row_id": "pending_qmfkg",
            "iteration": 1,
            "status": "suggested",
            "source": "qmf_kg",
            "review_status": "pending",
            "review_note": "",
            "x": 0.2,
            "fidelity": 0.8,
            "activity": "",
            "predicted_mean": 1.1,
            "predicted_std": 0.1,
            "acquisition": 0.2,
        },
        {
            "row_id": "accepted_qmfkg",
            "iteration": 2,
            "status": "suggested",
            "source": "qmf_kg",
            "review_status": "accepted",
            "review_note": "",
            "x": 0.3,
            "fidelity": 0.8,
            "activity": "",
            "predicted_mean": 1.2,
            "predicted_std": 0.1,
            "acquisition": 0.3,
        },
        {
            "row_id": "rejected_qmfkg",
            "iteration": 3,
            "status": "suggested",
            "source": "qmf_kg",
            "review_status": "rejected",
            "review_note": "",
            "x": 0.4,
            "fidelity": 0.8,
            "activity": "",
            "predicted_mean": 1.3,
            "predicted_std": 0.1,
            "acquisition": 0.4,
        },
        {
            "row_id": "deferred_qmfkg",
            "iteration": 4,
            "status": "suggested",
            "source": "qmf_kg",
            "review_status": "deferred",
            "review_note": "",
            "x": 0.6,
            "fidelity": 0.8,
            "activity": "",
            "predicted_mean": 1.4,
            "predicted_std": 0.1,
            "acquisition": 0.5,
        },
        {
            "row_id": "pending_sobol",
            "iteration": 5,
            "status": "suggested",
            "source": "sobol",
            "review_status": "pending",
            "review_note": "",
            "x": 0.7,
            "fidelity": 0.8,
            "activity": "",
            "predicted_mean": "",
            "predicted_std": "",
            "acquisition": "",
        },
    ]
    df = pd.DataFrame(rows, columns=canonical_columns(cfg))

    summary = fidelity_summary(cfg, df)

    assert value(summary, "pending_qmfkg_suggestions") == 2


def test_fidelity_summary_is_direction_aware_for_minimization() -> None:
    cfg = config()
    cfg = CampaignConfig(
        campaign_name=cfg.campaign_name,
        objective=ObjectiveConfig(name="activity", direction="minimize"),
        variables=cfg.variables,
        bo=cfg.bo,
        fidelity=cfg.fidelity,
    )

    summary = fidelity_summary(cfg, observed_log(cfg))

    assert value(summary, "best_observed_row_id") == "low_0"
    assert value(summary, "best_observed_objective") == pytest.approx(0.9)


def test_fidelity_summary_handles_empty_observed_logs() -> None:
    cfg = config()
    df = pd.DataFrame(columns=canonical_columns(cfg))

    summary = fidelity_summary(cfg, df)

    assert value(summary, "observed_rows") == 0
    assert value(summary, "target_fidelity_observed_rows") == 0
    assert value(summary, "min_observed_fidelity") is None
    assert value(summary, "best_observed_row_id") is None


def test_fidelity_summary_rejects_non_fidelity_config() -> None:
    cfg = CampaignConfig(
        campaign_name="plain",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=BOConfig(),
    )
    df = pd.DataFrame(columns=canonical_columns(cfg))

    with pytest.raises(ValueError, match="requires a config with a fidelity section"):
        fidelity_summary(cfg, df)


def test_extract_qmfkg_candidates_accepts_already_extracted_result() -> None:
    class FakeAcquisition:
        calls = 0

        def extract_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return candidates

    acquisition = FakeAcquisition()
    candidates = torch.tensor([[0.25, 0.5]], dtype=torch.double)

    extracted = _extract_qmf_kg_candidates(
        acquisition,
        candidates,
        q=1,
        num_fantasies=4,
    )

    assert torch.equal(extracted, candidates)
    assert acquisition.calls == 0


def test_extract_qmfkg_candidates_extracts_full_one_shot_result() -> None:
    class FakeAcquisition:
        calls = 0

        def extract_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return candidates[..., :1, :]

    acquisition = FakeAcquisition()
    candidates = torch.arange(10, dtype=torch.double).reshape(5, 2)

    extracted = _extract_qmf_kg_candidates(
        acquisition,
        candidates,
        q=1,
        num_fantasies=4,
    )

    assert extracted.shape == torch.Size([1, 2])
    assert torch.equal(extracted, candidates[:1])
    assert acquisition.calls == 1


def test_extract_qmfkg_candidates_extracts_batch_from_full_one_shot_result() -> None:
    class FakeAcquisition:
        def extract_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
            return candidates[..., :2, :]

    candidates = torch.arange(12, dtype=torch.double).reshape(6, 2)

    extracted = _extract_qmf_kg_candidates(
        FakeAcquisition(),
        candidates,
        q=2,
        num_fantasies=4,
    )

    assert extracted.shape == torch.Size([2, 2])
    assert torch.equal(extracted, candidates[:2])


def test_extract_qmfkg_candidates_rejects_unexpected_result_shape() -> None:
    class FakeAcquisition:
        def extract_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
            return candidates

    candidates = torch.arange(4, dtype=torch.double).reshape(2, 2)

    with pytest.raises(RuntimeError, match="unexpected candidate count"):
        _extract_qmf_kg_candidates(
            FakeAcquisition(),
            candidates,
            q=1,
            num_fantasies=4,
        )
