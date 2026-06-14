import pytest
import torch

from bo_forge.acquisition import _extract_qmf_kg_candidates
from bo_forge.config import (
    BOConfig,
    CampaignConfig,
    FidelityConfig,
    ObjectiveConfig,
    VariableConfig,
)
from bo_forge.multifidelity import (
    affine_fidelity_cost_model,
    fidelity_feature_index,
    fidelity_variable_index,
    target_fidelities,
    target_fidelity_projection,
    target_fidelity_unit_value,
)


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
