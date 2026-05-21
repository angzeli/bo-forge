import pandas as pd
import pytest
import torch

from bo_forge.config import BOConfig, CampaignConfig, ObjectiveConfig, VariableConfig
from bo_forge.transforms import (
    dataframe_to_unit_cube,
    encoded_dimension,
    encoded_feature_indices,
    encoded_feature_names,
    unit_cube_to_design_values,
    unit_cube_to_user_values,
)
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

    assert unit.shape == (1, 5)
    assert torch.all((unit >= 0.0) & (unit <= 1.0))
    assert unit[0, 0].item() == pytest.approx(0.5)


def test_categorical_values_encode_to_one_hot_columns() -> None:
    cfg = mixed_config()
    unit = dataframe_to_unit_cube(cfg, mixed_df())
    solvent_indices = encoded_feature_indices(cfg)["solvent"]
    solvent_block = unit[0, list(solvent_indices)]

    assert solvent_block.tolist() == [0.0, 1.0]
    assert solvent_block.sum().item() == pytest.approx(1.0)


def test_encoded_feature_metadata_is_stable_and_explicit() -> None:
    cfg = mixed_config()

    assert encoded_feature_names(cfg) == [
        "x",
        "repeats",
        "dose",
        "solvent=MeCN",
        "solvent=EtOH",
    ]
    assert encoded_dimension(cfg) == 5
    assert encoded_feature_indices(cfg) == {
        "x": (0,),
        "repeats": (1,),
        "dose": (2,),
        "solvent": (3, 4),
    }


def test_categorical_one_hot_distances_do_not_encode_ordinal_order() -> None:
    cfg = CampaignConfig(
        campaign_name="categories",
        objective=ObjectiveConfig(name="score", direction="maximize"),
        variables=(VariableConfig("solvent", "categorical", values=("MeCN", "DMF", "THF")),),
        bo=BOConfig(),
    )
    reordered = CampaignConfig(
        campaign_name="categories",
        objective=ObjectiveConfig(name="score", direction="maximize"),
        variables=(VariableConfig("solvent", "categorical", values=("THF", "MeCN", "DMF")),),
        bo=BOConfig(),
    )

    for candidate_config in (cfg, reordered):
        encoded = dataframe_to_unit_cube(
            candidate_config,
            pd.DataFrame(
                {"solvent": list(candidate_config.variables[0].values)}
            ),
        )
        distances = torch.pdist(encoded)
        assert torch.allclose(
            distances,
            torch.full_like(distances, 2.0**0.5),
        )


def test_latent_values_decode_to_user_space_and_clip_boundaries() -> None:
    decoded = unit_cube_to_user_values(
        mixed_config(),
        torch.tensor([[-0.1, 0.99, 1.0, 1.0, 0.0]], dtype=torch.double),
    )

    assert decoded == [(0.0, 3, 0.5, "MeCN")]


def test_design_values_decode_from_one_scalar_per_user_variable() -> None:
    decoded = unit_cube_to_design_values(
        mixed_config(),
        torch.tensor([[-0.1, 0.99, 1.0, 0.0]], dtype=torch.double),
    )

    assert decoded == [(0.0, 3, 0.5, "MeCN")]
