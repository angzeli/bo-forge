from pathlib import Path

import pytest

from bo_forge.config import CampaignConfig
from bo_forge.errors import ConfigError


def write_yaml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_config_from_yaml_parses_valid_config(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: photocatalyst_loading
objective:
  name: activity
  direction: maximize
variables:
  - name: precursor_ratio
    type: continuous
    lower: 0.0
    upper: 1.0
bo:
  batch_size: 2
  initial_design_size: 4
  acquisition: log_ei
""",
    )

    config = CampaignConfig.from_yaml(path)

    assert config.campaign_name == "photocatalyst_loading"
    assert config.objective.name == "activity"
    assert config.objective.direction == "maximize"
    assert config.variable_names == ["precursor_ratio"]
    assert config.bo.batch_size == 2


def test_example_minimize_config_parses() -> None:
    config = CampaignConfig.from_yaml("configs/simple_2d_minimize.yaml")

    assert config.campaign_name == "process_defect_minimisation"
    assert config.objective.name == "defect_rate"
    assert config.objective.direction == "minimize"
    assert config.direction_sign == -1.0
    assert config.variable_names == ["catalyst_loading", "cure_temperature"]
    assert config.bo.batch_size == 2


def test_config_rejects_invalid_bounds(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_bounds
objective:
  name: activity
  direction: maximize
variables:
  - name: temperature
    type: continuous
    lower: 800
    upper: 300
""",
    )

    with pytest.raises(ConfigError, match="Variable 'temperature' has lower >= upper"):
        CampaignConfig.from_yaml(path)


def test_config_rejects_unsupported_variable_type(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: mixed
objective:
  name: activity
  direction: maximize
variables:
  - name: catalyst
    type: categorical
    lower: 0
    upper: 1
""",
    )

    with pytest.raises(ConfigError, match="unsupported type 'categorical'"):
        CampaignConfig.from_yaml(path)


def test_config_rejects_duplicate_variable_names(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: duplicate
objective:
  name: activity
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
  - name: x
    type: continuous
    lower: 0
    upper: 1
""",
    )

    with pytest.raises(ConfigError, match="Duplicate variable name 'x'"):
        CampaignConfig.from_yaml(path)


def test_config_rejects_invalid_objective_direction(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_direction
objective:
  name: activity
  direction: largest
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
""",
    )

    with pytest.raises(ConfigError, match="invalid direction 'largest'"):
        CampaignConfig.from_yaml(path)
