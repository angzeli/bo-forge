from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch

import bo_forge.suggestions as suggestions_module
from bo_forge.config import CampaignConfig
from bo_forge.io import empty_campaign_log
from bo_forge.logs import load_campaign_log
from bo_forge.multifidelity import FIDELITY_COVERAGE_COLUMNS
from bo_forge.session import CampaignSession
from bo_forge.suggestions import suggest_next
from bo_forge.transforms import values_to_unit_cube
from bo_forge.validation import canonical_columns

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("config_name", "log_name", "expected_levels"),
    [
        (
            "15_multi_fidelity_qmfkg.yaml",
            "15_multi_fidelity_qmfkg_campaign_log.csv",
            None,
        ),
        (
            "22_discrete_multi_fidelity_qmfkg.yaml",
            "22_discrete_multi_fidelity_qmfkg_campaign_log.csv",
            (0.25, 0.5, 0.75, 1.0),
        ),
    ],
)
def test_v243_fidelity_assets_preserve_schema_and_read_only_contracts(
    config_name: str,
    log_name: str,
    expected_levels: tuple[float, ...] | None,
) -> None:
    config_path = PROJECT_ROOT / "configs" / config_name
    log_path = PROJECT_ROOT / "examples" / log_name
    before = log_path.read_bytes()
    campaign = CampaignSession.from_files(config_path, log_path)

    campaign.validate()
    summary = campaign.fidelity_summary()
    coverage = campaign.fidelity_coverage()
    report = campaign.report()

    assert list(campaign.df.columns) == canonical_columns(campaign.config)
    assert campaign.config.fidelity is not None
    assert campaign.config.fidelity.levels == expected_levels
    assert list(summary.columns) == ["field", "value"]
    assert list(coverage.columns) == FIDELITY_COVERAGE_COLUMNS
    assert tuple(report)[-2:] == ("fidelity_summary", "fidelity_coverage")
    assert log_path.read_bytes() == before


@pytest.mark.parametrize("config_name", [
    "15_multi_fidelity_qmfkg.yaml",
    "22_discrete_multi_fidelity_qmfkg.yaml",
])
@pytest.mark.parametrize("initial_design_method", ["sobol", "random"])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_v243_initial_design_matrix_preserves_deterministic_batch_contract(
    config_name: str,
    initial_design_method: str,
    batch_size: int,
) -> None:
    base = CampaignConfig.from_yaml(PROJECT_ROOT / "configs" / config_name)
    config = replace(
        base,
        bo=replace(
            base.bo,
            batch_size=batch_size,
            initial_design_size=12,
            initial_design_method=initial_design_method,
        ),
    )
    df = empty_campaign_log(config)

    first = suggest_next(config, df, batch_size=batch_size)
    second = suggest_next(config, df, batch_size=batch_size)

    pd.testing.assert_frame_equal(df, empty_campaign_log(config))
    pd.testing.assert_frame_equal(
        first.drop(columns="row_id").reset_index(drop=True),
        second.drop(columns="row_id").reset_index(drop=True),
    )
    assert len(first) == batch_size
    assert first["source"].eq(initial_design_method).all()
    assert first["iteration"].eq(0).all()
    assert first["predicted_mean"].eq("").all()
    assert first["predicted_std"].eq("").all()
    assert first["acquisition"].eq("").all()
    if config.fidelity is not None and config.fidelity.levels is not None:
        assert set(first[config.fidelity.variable].astype(float)).issubset(
            set(config.fidelity.levels)
        )


@pytest.mark.parametrize(
    ("config_name", "log_name", "candidate_values"),
    [
        (
            "15_multi_fidelity_qmfkg.yaml",
            "15_multi_fidelity_qmfkg_campaign_log.csv",
            [
                (0.12, 0.30),
                (0.20, 0.45),
                (0.28, 0.70),
                (0.38, 0.90),
            ],
        ),
        (
            "22_discrete_multi_fidelity_qmfkg.yaml",
            "22_discrete_multi_fidelity_qmfkg_campaign_log.csv",
            [
                (0.12, 55.0, 0.25),
                (0.20, 70.0, 0.50),
                (0.28, 100.0, 0.75),
                (0.38, 112.0, 1.00),
            ],
        ),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_v243_mocked_model_batch_matrix_preserves_public_row_contract(
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
    log_name: str,
    candidate_values: list[tuple[float, ...]],
    batch_size: int,
) -> None:
    config = CampaignConfig.from_yaml(PROJECT_ROOT / "configs" / config_name)
    df = load_campaign_log(PROJECT_ROOT / "examples" / log_name, config)
    before = df.copy(deep=True)
    selected_values = candidate_values[:batch_size]
    candidates = values_to_unit_cube(config, selected_values)

    class FakePosterior:
        def __init__(self, count: int) -> None:
            self.mean = torch.arange(1, count + 1, dtype=torch.double).reshape(-1, 1)
            self.variance = torch.full((count, 1), 0.04, dtype=torch.double)

    class FakeModel:
        def posterior(self, x_unit: torch.Tensor) -> FakePosterior:
            return FakePosterior(x_unit.shape[0])

    monkeypatch.setattr(
        suggestions_module,
        "fit_multi_fidelity_gp_model",
        lambda *_args, **_kwargs: FakeModel(),
    )
    monkeypatch.setattr(
        suggestions_module,
        "optimize_posterior_mean_at_target_fidelity",
        lambda **_kwargs: torch.tensor([1.0], dtype=torch.double),
    )
    monkeypatch.setattr(
        suggestions_module,
        "optimize_qmf_kg",
        lambda **_kwargs: (candidates, torch.tensor(0.75), "qmf_kg"),
    )

    suggestions = suggest_next(config, df, batch_size=batch_size)

    pd.testing.assert_frame_equal(df, before)
    assert list(suggestions.columns) == canonical_columns(config)
    assert len(suggestions) == batch_size
    assert suggestions["source"].eq("qmf_kg").all()
    assert suggestions["iteration"].nunique() == 1
    assert suggestions["predicted_mean"].astype(float).tolist() == pytest.approx(
        list(range(1, batch_size + 1))
    )
    assert suggestions["predicted_std"].astype(float).tolist() == pytest.approx(
        [0.2] * batch_size
    )
    assert suggestions["acquisition"].astype(float).tolist() == pytest.approx(
        [0.75] * batch_size
    )
    expected_fidelity = [values[-1] for values in selected_values]
    assert suggestions[config.fidelity.variable].astype(float).tolist() == pytest.approx(
        expected_fidelity
    )
