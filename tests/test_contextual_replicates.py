from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import torch

import bo_forge.suggestions as suggestions_module
from bo_forge.config import CampaignConfig, parse_campaign_config
from bo_forge.errors import ConfigError, LogValidationError, LogWriteError
from bo_forge.replicates import modeling_observed_data_with_variance
from bo_forge.session import CampaignSession
from bo_forge.suggestions import suggest_next
from bo_forge.validation import canonical_columns, validate_campaign_data


def contextual_replicate_raw() -> dict[str, object]:
    return {
        "campaign_name": "contextual_replicates",
        "objective": {"name": "yield_score", "direction": "maximize"},
        "variables": [
            {"name": "loading", "type": "continuous", "lower": 0.0, "upper": 1.0},
            {"name": "feedstock", "type": "categorical", "values": ["A", "B", "C"]},
        ],
        "context": {"variables": ["feedstock"], "default_values": {"feedstock": "A"}},
        "replicates": {
            "enabled": True,
            "suggestion_policy": "uncertain_best",
            "replicate_threshold": 0.1,
            "min_repeats_at_best": 2,
            "max_repeats_per_group": 4,
            "noise_floor": 1.0e-8,
        },
        "bo": {
            "batch_size": 1,
            "initial_design_size": 2,
            "acquisition": "log_ei",
            "random_seed": 7,
            "raw_samples": 8,
            "num_restarts": 1,
            "mc_samples": 8,
            "min_normalized_distance": 0.0,
        },
    }


def contextual_replicate_config(
    *, review: bool = False, cost: bool = False
) -> CampaignConfig:
    raw = contextual_replicate_raw()
    if review:
        raw["review"] = {"enabled": True}
    if cost:
        raw["cost"] = {
            "expression": "1.0 + loading",
            "weight": 0.25,
            "budget": 20.0,
            "candidate_pool_size": 8,
            "top_k": 4,
        }
    return parse_campaign_config(raw)


def contextual_replicate_log(config: CampaignConfig) -> pd.DataFrame:
    rows = [
        ("a0", "group_a", 0, 0.2, "A", 0.70),
        ("a1", "group_a", 1, 0.2, "A", 0.90),
        ("b0", "group_b", 0, 0.8, "B", 1.50),
    ]
    records: list[dict[str, object]] = []
    for iteration, (row_id, group, replicate_index, loading, feedstock, score) in enumerate(rows):
        row: dict[str, object] = {
            "row_id": row_id,
            "iteration": iteration,
            "status": "observed",
            "source": "manual",
            "replicate_group": group,
            "replicate_index": replicate_index,
            "loading": loading,
            "feedstock": feedstock,
            "yield_score": score,
            "predicted_mean": "",
            "predicted_std": "",
            "acquisition": "",
        }
        if config.review.enabled:
            row.update({"review_status": "accepted", "review_note": ""})
        if config.cost is not None:
            row.update(
                {
                    "cost_estimate": 1.0 + loading,
                    "cost_actual": 1.0 + loading,
                    "utility": "",
                }
            )
        records.append(row)
    return pd.DataFrame(records, columns=canonical_columns(config))


@pytest.mark.parametrize(
    ("review", "cost"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_contextual_replicate_combinations_parse(review: bool, cost: bool) -> None:
    config = contextual_replicate_config(review=review, cost=cost)

    assert config.context is not None
    assert config.replicates.enabled
    assert config.review.enabled is review
    assert (config.cost is not None) is cost


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {
                "objectives": [
                    {
                        "name": "yield_score",
                        "direction": "maximize",
                        "reference_point": 0.0,
                    },
                    {
                        "name": "waste",
                        "direction": "minimize",
                        "reference_point": 1.0,
                    },
                ]
            },
            "single-objective",
        ),
        ({"stages": [{"name": "screen", "variables": ["loading"]}]}, "structured campaign stages"),
        ({"fidelity": {"variable": "loading", "target": 1.0}}, "fidelity campaigns"),
    ],
)
def test_contextual_replicates_keep_existing_unsupported_scope(
    change: dict[str, object], message: str
) -> None:
    raw = deepcopy(contextual_replicate_raw())
    raw.update(change)
    if "objectives" in change:
        raw.pop("objective")
        raw["bo"]["acquisition"] = "qlog_ehvi"  # type: ignore[index]
    if "fidelity" in change:
        raw["bo"]["acquisition"] = "qmf_kg"  # type: ignore[index]

    with pytest.raises(ConfigError, match=message):
        parse_campaign_config(raw)


@pytest.mark.parametrize("acquisition", ["qlog_nei", "qlog_nehvi"])
def test_contextual_replicates_reject_noisy_acquisitions(acquisition: str) -> None:
    raw = contextual_replicate_raw()
    raw["bo"]["acquisition"] = acquisition  # type: ignore[index]
    if acquisition == "qlog_nehvi":
        raw.pop("objective")
        raw["objectives"] = [
            {"name": "yield_score", "direction": "maximize", "reference_point": 0.0},
            {"name": "waste", "direction": "minimize", "reference_point": 1.0},
        ]

    with pytest.raises(ConfigError, match="cannot be combined with context|single-objective"):
        parse_campaign_config(raw)


def test_contextual_replicate_schema_and_group_context_validation() -> None:
    config = contextual_replicate_config(review=True, cost=True)
    assert canonical_columns(config) == [
        "row_id",
        "iteration",
        "status",
        "source",
        "review_status",
        "review_note",
        "replicate_group",
        "replicate_index",
        "loading",
        "feedstock",
        "yield_score",
        "cost_estimate",
        "cost_actual",
        "predicted_mean",
        "predicted_std",
        "acquisition",
        "utility",
    ]
    df = contextual_replicate_log(config)
    validate_campaign_data(config, df)

    invalid = df.copy()
    invalid.loc[1, "feedstock"] = "B"
    with pytest.raises(LogValidationError, match="Replicate group 'group_a'"):
        validate_campaign_data(config, invalid)


def test_contextual_replicates_fit_group_means_and_variance_across_contexts() -> None:
    config = contextual_replicate_config()
    model_df, yvar_df = modeling_observed_data_with_variance(
        config, contextual_replicate_log(config)
    )

    assert model_df[["loading", "feedstock"]].to_dict("records") == [
        {"loading": 0.2, "feedstock": "A"},
        {"loading": 0.8, "feedstock": "B"},
    ]
    assert model_df["yield_score"].astype(float).tolist() == pytest.approx([0.8, 1.5])
    assert yvar_df is not None
    assert yvar_df["yield_score"].astype(float).gt(0).all()


def test_uncertain_best_repeat_is_restricted_to_requested_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = contextual_replicate_config()
    df = contextual_replicate_log(config)

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0]], dtype=torch.double)
        variance = torch.tensor([[0.04], [0.04]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())

    suggestions = suggest_next(config, df, context_values={"feedstock": "A"})

    assert suggestions.loc[0, "replicate_group"] == "group_a"
    assert int(suggestions.loc[0, "replicate_index"]) == 2
    assert suggestions.loc[0, "feedstock"] == "A"


def test_contextual_repeat_selection_does_not_depend_on_aggregate_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = contextual_replicate_config()
    df = contextual_replicate_log(config)
    aggregate = suggestions_module.aggregate_observed_replicates(config, df)
    aggregate.index = [10, 20]

    class FakePosterior:
        mean = torch.tensor([[2.0], [10.0]], dtype=torch.double)
        variance = torch.full((2, 1), 0.04, dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    monkeypatch.setattr(
        suggestions_module,
        "aggregate_observed_replicates",
        lambda *_args: aggregate,
    )
    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())

    suggestions = suggest_next(config, df, context_values={"feedstock": "A"})

    assert suggestions.loc[0, "replicate_group"] == "group_a"
    assert suggestions.loc[0, "feedstock"] == "A"


def test_context_matching_uses_normalized_numeric_values() -> None:
    campaign = CampaignSession.from_files(
        Path("configs/21_contextual_replicate_logei.yaml"),
        Path("examples/21_contextual_replicate_campaign_log.csv"),
    )
    df = campaign.df.astype(object)
    df.loc[df["feedstock_acidity"].astype(float) == 0.25, "feedstock_acidity"] = "0.25"

    matching = suggestions_module._rows_matching_context(
        campaign.config,
        df,
        {"feedstock_acidity": "0.25"},
    )

    assert set(matching["replicate_group"]) == {
        "group_acid25_best",
        "group_acid25_other",
    }


def test_no_matching_replicate_group_falls_back_to_context_fixed_exploration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = contextual_replicate_config()
    df = contextual_replicate_log(config)

    def fake_model_based(**kwargs):
        assert kwargs["context_values"] == {"feedstock": "C"}
        row = {
            "row_id": "new_c",
            "iteration": 4,
            "status": "suggested",
            "source": "log_ei",
            "replicate_group": "new_c",
            "replicate_index": 0,
            "loading": 0.4,
            "feedstock": "C",
            "yield_score": "",
            "predicted_mean": 0.9,
            "predicted_std": 0.2,
            "acquisition": 0.1,
        }
        return pd.DataFrame([row], columns=canonical_columns(config))

    monkeypatch.setattr(suggestions_module, "_suggest_model_based", fake_model_based)
    suggestions = suggest_next(config, df, context_values={"feedstock": "C"})

    assert suggestions.loc[0, "feedstock"] == "C"
    assert suggestions.loc[0, "replicate_group"] == "new_c"


def test_repeat_and_exploration_batch_share_requested_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = contextual_replicate_config()
    df = contextual_replicate_log(config)

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0]], dtype=torch.double)
        variance = torch.tensor([[0.04], [0.04]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    def fake_model_based(**kwargs):
        assert kwargs["context_values"] == {"feedstock": "A"}
        row = {
            "row_id": "new_a",
            "iteration": 99,
            "status": "suggested",
            "source": "log_ei",
            "replicate_group": "new_a",
            "replicate_index": 0,
            "loading": 0.6,
            "feedstock": "A",
            "yield_score": "",
            "predicted_mean": 1.0,
            "predicted_std": 0.1,
            "acquisition": 0.2,
        }
        return pd.DataFrame([row], columns=canonical_columns(config))

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    monkeypatch.setattr(suggestions_module, "_suggest_model_based", fake_model_based)

    suggestions = suggest_next(
        config,
        df,
        batch_size=2,
        context_values={"feedstock": "A"},
    )

    assert suggestions["feedstock"].tolist() == ["A", "A"]
    assert suggestions["replicate_group"].tolist() == ["group_a", "new_a"]
    assert suggestions["iteration"].astype(int).nunique() == 1


def test_contextual_cost_repeat_fills_cost_without_utility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = contextual_replicate_config(cost=True)
    df = contextual_replicate_log(config)

    class FakePosterior:
        mean = torch.tensor([[1.0], [2.0]], dtype=torch.double)
        variance = torch.tensor([[0.04], [0.04]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    suggestions = suggest_next(config, df, context_values={"feedstock": "A"})

    assert suggestions.loc[0, "replicate_group"] == "group_a"
    assert float(suggestions.loc[0, "cost_estimate"]) == pytest.approx(1.2)
    assert suggestions.loc[0, "utility"] == ""


def test_contextual_cost_repeat_reserves_budget_before_context_fixed_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = contextual_replicate_config(cost=True)
    df = contextual_replicate_log(config)

    class FakePosterior:
        mean = torch.tensor([[1.0], [2.0]], dtype=torch.double)
        variance = torch.tensor([[0.04], [0.04]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    def fake_cost_fill(**kwargs):
        assert kwargs["context_values"] == {"feedstock": "A"}
        assert kwargs["config"].cost.budget == pytest.approx(18.8)
        row = {
            "row_id": "new_a",
            "iteration": 99,
            "status": "suggested",
            "source": "cost_log_ei",
            "replicate_group": "new_a",
            "replicate_index": 0,
            "loading": 0.5,
            "feedstock": "A",
            "yield_score": "",
            "cost_estimate": 1.5,
            "cost_actual": "",
            "predicted_mean": 1.0,
            "predicted_std": 0.1,
            "acquisition": 0.2,
            "utility": -0.175,
        }
        return pd.DataFrame([row], columns=canonical_columns(config))

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    monkeypatch.setattr(
        suggestions_module,
        "_suggest_cost_aware_model_based",
        fake_cost_fill,
    )

    suggestions = suggest_next(
        config,
        df,
        batch_size=2,
        context_values={"feedstock": "A"},
    )

    assert suggestions["feedstock"].tolist() == ["A", "A"]
    assert suggestions["replicate_group"].tolist() == ["group_a", "new_a"]


def test_example_21_validates_and_report_keeps_context_and_group_tables() -> None:
    campaign = CampaignSession.from_files(
        Path("configs/21_contextual_replicate_logei.yaml"),
        Path("examples/21_contextual_replicate_campaign_log.csv"),
    )

    campaign.validate()
    report = campaign.report()

    assert "context_summary" in report
    assert "replicate_summary" in report
    assert int(report["context_summary"]["observed_rows"].sum()) == 5
    assert len(report["replicate_summary"]) == 4


def test_example_21_session_round_trip_and_invalid_observation_are_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = Path("configs/21_contextual_replicate_logei.yaml")
    log_path = tmp_path / "contextual_replicates.csv"
    log_path.write_bytes(
        Path("examples/21_contextual_replicate_campaign_log.csv").read_bytes()
    )
    campaign = CampaignSession.from_files(config_path, log_path)

    class FakePosterior:
        mean = torch.tensor([[2.0], [1.0], [10.0], [0.0]], dtype=torch.double)
        variance = torch.full((4, 1), 0.04, dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    before_dry_run = log_path.read_bytes()

    suggestions = campaign.suggest_next(
        batch_size=1,
        context_values={"feedstock_acidity": 0.25},
    )

    assert log_path.read_bytes() == before_dry_run
    assert suggestions.loc[0, "replicate_group"] == "group_acid25_best"
    assert int(suggestions.loc[0, "replicate_index"]) == 2
    assert suggestions.loc[0, "review_status"] == "pending"
    assert float(suggestions.loc[0, "cost_estimate"]) == pytest.approx(3.9)

    campaign.append_suggestions(suggestions)
    row_id = str(suggestions.loc[0, "row_id"])
    campaign.review_suggestion(row_id, "accept", "approved")
    before_invalid_observation = log_path.read_bytes()

    with pytest.raises(LogWriteError, match="actual_cost.*finite and >= 0"):
        campaign.mark_observed(
            row_id,
            objective_value=0.91,
            actual_cost=float("nan"),
        )

    assert log_path.read_bytes() == before_invalid_observation
    campaign.mark_observed(row_id, objective_value=0.91, actual_cost=4.0)

    reloaded = CampaignSession.from_files(config_path, log_path)
    reloaded.validate()
    observed = reloaded.df.loc[reloaded.df["row_id"] == row_id].iloc[0]
    report = reloaded.report()
    assert observed["status"] == "observed"
    assert observed["review_note"] == "approved"
    assert float(observed["yield_score"]) == pytest.approx(0.91)
    assert float(observed["cost_actual"]) == pytest.approx(4.0)
    assert "context_summary" in report
    assert "replicate_summary" in report
    assert "cost_summary" in report
