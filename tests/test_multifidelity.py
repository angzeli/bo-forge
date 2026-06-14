import pandas as pd
import pytest
import torch

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
    fidelity_summary,
    fidelity_variable_index,
    target_fidelities,
    target_fidelity_projection,
    target_fidelity_unit_value,
)
from bo_forge.validation import canonical_columns


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
