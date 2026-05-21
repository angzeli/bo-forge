import pandas as pd
import pytest
import torch

from bo_forge.config import BOConfig, CampaignConfig, ObjectiveConfig, VariableConfig
from bo_forge.transforms import dataframe_to_unit_cube, unit_cube_to_user_values
from bo_forge.validation import canonical_columns


def mixed_config() -> CampaignConfig:
    return CampaignConfig(
        campaign_name="mixed",
        objective=ObjectiveConfig(name="score", direction="maximize"),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("repeats", "integer", 1.0, 3.0),
            VariableConfig("dose", "discrete", values=(0.1, 0.2, 0.5)),
            VariableConfig("solvent", "categorical", values=("MeCN", "EtOH")),
        ),
        bo=BOConfig(batch_size=1, initial_design_size=3),
    )


def mixed_df() -> pd.DataFrame:
    cfg = mixed_config()
    return pd.DataFrame(
        [
            {
                "row_id": "obs_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "x": 0.5,
                "repeats": 2.0,
                "dose": "0.20",
                "solvent": "EtOH",
                "score": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )


def test_mixed_dataframe_encodes_to_unit_cube() -> None:
    unit = dataframe_to_unit_cube(mixed_config(), mixed_df())

    assert unit.shape == (1, 4)
    assert torch.all((unit >= 0.0) & (unit <= 1.0))
    assert unit[0, 0].item() == pytest.approx(0.5)


def test_latent_values_decode_to_user_space_and_clip_boundaries() -> None:
    decoded = unit_cube_to_user_values(
        mixed_config(),
        torch.tensor([[-0.1, 0.99, 1.0, 0.0]], dtype=torch.double),
    )

    assert decoded == [(0.0, 3, 0.5, "MeCN")]
