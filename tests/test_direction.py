import torch

from bo_forge.config import BOConfig, CampaignConfig, ObjectiveConfig, VariableConfig
from bo_forge.transforms import objective_from_model_space, objective_to_model_space


def config(direction: str) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="direction",
        objective=ObjectiveConfig(name="loss", direction=direction),
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=BOConfig(),
    )


def test_maximize_direction_leaves_objective_unchanged() -> None:
    cfg = config("maximize")
    values = torch.tensor([[1.0], [2.0]], dtype=torch.double)

    model_values = objective_to_model_space(cfg, values)
    user_values = objective_from_model_space(cfg, model_values)

    assert torch.equal(model_values, values)
    assert torch.equal(user_values, values)


def test_minimize_direction_negates_for_model_and_converts_back() -> None:
    cfg = config("minimize")
    values = torch.tensor([[1.0], [2.0]], dtype=torch.double)

    model_values = objective_to_model_space(cfg, values)
    user_values = objective_from_model_space(cfg, model_values)

    assert torch.equal(model_values, -values)
    assert torch.equal(user_values, values)

