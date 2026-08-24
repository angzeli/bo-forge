"""Configuration tests for noisy acquisition scope boundaries."""

from tests._config_support import (
    CampaignConfig,
    ConfigError,
    Path,
    pytest,
    write_yaml,
)


def test_config_rejects_qlog_nei_multi_objective_with_specific_message(
    tmp_path: Path,
) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_nei_multi
objectives:
  - name: yield
    direction: maximize
    reference_point: 0
  - name: waste
    direction: minimize
    reference_point: 10
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
bo:
  acquisition: qlog_nei
""",
    )

    with pytest.raises(ConfigError, match="single-objective only"):
        CampaignConfig.from_yaml(path)

def test_config_rejects_qlog_nei_fidelity_with_specific_message(
    tmp_path: Path,
) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_nei_fidelity
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
  - name: fidelity
    type: continuous
    lower: 0.1
    upper: 1.0
fidelity:
  variable: fidelity
  target: 1.0
bo:
  acquisition: qlog_nei
""",
    )

    with pytest.raises(ConfigError, match="cannot be combined with fidelity"):
        CampaignConfig.from_yaml(path)

def test_config_accepts_qlog_nei_replicates_new_only(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: noisy_replicates
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
replicates:
  enabled: true
  suggestion_policy: new_only
bo:
  acquisition: qlog_nei
""",
    )

    config = CampaignConfig.from_yaml(path)

    assert config.bo.acquisition == "qlog_nei"
    assert config.replicates.enabled
    assert config.replicates.suggestion_policy == "new_only"

def test_qlog_nei_example_config_loads() -> None:
    config = CampaignConfig.from_yaml("configs/18_noisy_pending_qlognei.yaml")

    assert config.bo.acquisition == "qlog_nei"
    assert config.review.enabled
