from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from bo_forge.config import (
    BOConfig,
    CampaignConfig,
    ConstraintConfig,
    ContextConfig,
    ObjectiveConfig,
    ReviewConfig,
    VariableConfig,
)
from bo_forge.contextual import (
    context_summary,
    contextual_fixed_feature_assignments,
    resolve_context_values,
)
from bo_forge.errors import ConfigError, SuggestionError
from bo_forge.io import empty_campaign_log
from bo_forge.session import CampaignSession
from bo_forge.suggestions import suggest_next
from bo_forge.transforms import encoded_feature_indices
from bo_forge.validation import canonical_columns, validate_campaign_data


def contextual_config(*, default: object | None = 0.5) -> CampaignConfig:
    defaults = {} if default is None else {"feedstock_acidity": default}
    return CampaignConfig(
        campaign_name="contextual_test",
        objective=ObjectiveConfig("yield_score", "maximize"),
        variables=(
            VariableConfig("catalyst_loading", "continuous", 0.0, 1.0),
            VariableConfig("feedstock_acidity", "continuous", 0.0, 1.0),
        ),
        bo=BOConfig(
            batch_size=1,
            initial_design_size=4,
            random_seed=16,
            raw_samples=16,
            num_restarts=2,
            mc_samples=16,
        ),
        context=ContextConfig(
            variables=("feedstock_acidity",),
            default_values=defaults,
        ),
    )


def observed_contextual_log(cfg: CampaignConfig) -> pd.DataFrame:
    rows = [
        ("obs_0", 0.15, 0.2, 0.50),
        ("obs_1", 0.65, 0.2, 0.80),
        ("obs_2", 0.25, 0.8, 0.55),
        ("obs_3", 0.75, 0.8, 0.72),
    ]
    return pd.DataFrame(
        [
            {
                "row_id": row_id,
                "iteration": index,
                "status": "observed",
                "source": "manual",
                "catalyst_loading": loading,
                "feedstock_acidity": acidity,
                "yield_score": score,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
            for index, (row_id, loading, acidity, score) in enumerate(rows)
        ],
        columns=canonical_columns(cfg),
    )


def test_valid_context_config_parses_and_has_decision_variables() -> None:
    cfg = CampaignConfig.from_yaml("configs/16_contextual_logei.yaml")

    assert cfg.context is not None
    assert cfg.context_variable_names == ["feedstock_acidity"]
    assert cfg.decision_variable_names == [
        "catalyst_loading",
        "reaction_temperature",
        "solvent",
    ]
    assert cfg.context.default_values["feedstock_acidity"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "yaml_text, message",
    [
        (
            """
campaign_name: bad_context
objective: {name: score, direction: maximize}
variables:
  - {name: x, type: continuous, lower: 0, upper: 1}
context:
  variables: [missing]
""",
            "unknown variable",
        ),
        (
            """
campaign_name: bad_context
objective: {name: score, direction: maximize}
variables:
  - {name: x, type: continuous, lower: 0, upper: 1}
context:
  variables: [x, x]
""",
            "Duplicate context variable",
        ),
        (
            """
campaign_name: bad_context
objective: {name: score, direction: maximize}
variables:
  - {name: x, type: continuous, lower: 0, upper: 1}
context:
  variables: [x]
""",
            "at least one non-context decision variable",
        ),
        (
            """
campaign_name: bad_context
objective: {name: score, direction: maximize}
variables:
  - {name: x, type: continuous, lower: 0, upper: 1}
  - {name: c, type: continuous, lower: 0, upper: 1}
context:
  variables: [c]
  default_values: {c: 2.0}
""",
            "outside variable",
        ),
    ],
)
def test_context_config_rejects_invalid_definitions(
    tmp_path,
    yaml_text: str,
    message: str,
) -> None:
    path = tmp_path / "context.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        CampaignConfig.from_yaml(path)


@pytest.mark.parametrize(
    "extra_yaml, message",
    [
        (
            "objectives: [{name: a, direction: maximize, reference_point: 0}, "
            "{name: b, direction: maximize, reference_point: 0}]",
            "single-objective",
        ),
        ("stages: [{name: screen, variables: [x]}]", "structured"),
        ("fidelity: {variable: x, target: 1.0}", "fidelity"),
        ("replicates: {enabled: true}", "replicate"),
    ],
)
def test_context_config_rejects_unsupported_combinations(
    tmp_path,
    extra_yaml: str,
    message: str,
) -> None:
    objective = (
        ""
        if extra_yaml.startswith("objectives:")
        else "objective: {name: score, direction: maximize}"
    )
    path = tmp_path / "context.yaml"
    path.write_text(
        f"""
campaign_name: bad_context_combo
{objective}
variables:
  - {{name: x, type: continuous, lower: 0, upper: 1}}
  - {{name: c, type: continuous, lower: 0, upper: 1}}
context:
  variables: [c]
{extra_yaml}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match=message):
        CampaignConfig.from_yaml(path)


def test_contextual_canonical_columns_match_normal_single_objective_schema() -> None:
    cfg = CampaignConfig.from_yaml("configs/16_contextual_logei.yaml")

    assert canonical_columns(cfg) == [
        "row_id",
        "iteration",
        "status",
        "source",
        "catalyst_loading",
        "reaction_temperature",
        "solvent",
        "feedstock_acidity",
        "yield_score",
        "predicted_mean",
        "predicted_std",
        "acquisition",
    ]
    validate_campaign_data(
        cfg,
        pd.read_csv("examples/16_contextual_logei_campaign_log.csv", keep_default_na=False),
    )


def test_contextual_fixed_features_do_not_enumerate_categorical_context() -> None:
    cfg = CampaignConfig(
        campaign_name="categorical_context",
        objective=ObjectiveConfig("score", "maximize"),
        variables=(
            VariableConfig("solvent", "categorical", values=("MeCN", "EtOH")),
            VariableConfig("feedstock", "categorical", values=("A", "B")),
        ),
        bo=BOConfig(),
        context=ContextConfig(variables=("feedstock",), default_values={"feedstock": "B"}),
    )

    fixed = contextual_fixed_feature_assignments(cfg, resolve_context_values(cfg))
    feedstock_indices = encoded_feature_indices(cfg)["feedstock"]

    assert len(fixed) == 2
    assert {tuple(item[index] for index in feedstock_indices) for item in fixed} == {
        (0.0, 1.0)
    }


def test_initial_design_suggestions_fill_supplied_context_values() -> None:
    cfg = contextual_config(default=None)
    log = empty_campaign_log(cfg)

    suggestions = suggest_next(
        cfg,
        log,
        batch_size=2,
        context_values={"feedstock_acidity": 0.75},
    )

    assert len(suggestions) == 2
    assert set(suggestions["feedstock_acidity"].astype(float)) == {0.75}
    assert set(suggestions["source"]) == {"sobol"}
    assert log.empty


def test_initial_design_finite_space_counts_only_matching_context() -> None:
    cfg = CampaignConfig(
        campaign_name="finite_context",
        objective=ObjectiveConfig("score", "maximize"),
        variables=(
            VariableConfig("x", "discrete", values=(0.0, 1.0)),
            VariableConfig("ctx", "discrete", values=(0.0, 1.0, 2.0)),
        ),
        bo=BOConfig(batch_size=1, initial_design_size=10, random_seed=1),
        context=ContextConfig(variables=("ctx",), default_values={}),
    )
    log = pd.DataFrame(
        [
            {
                "row_id": "obs_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "x": 0.0,
                "ctx": 0.0,
                "score": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "obs_1",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "x": 1.0,
                "ctx": 1.0,
                "score": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )

    suggestions = suggest_next(cfg, log, batch_size=1, context_values={"ctx": 2.0})

    assert suggestions["ctx"].astype(float).tolist() == [pytest.approx(2.0)]


def test_model_based_suggestions_fix_context_values_and_are_non_mutating() -> None:
    cfg = contextual_config()
    log = observed_contextual_log(cfg)
    before = log.copy(deep=True)

    suggestions = suggest_next(
        cfg,
        log,
        batch_size=1,
        context_values={"feedstock_acidity": 0.25},
    )

    assert suggestions["feedstock_acidity"].astype(float).tolist() == [pytest.approx(0.25)]
    assert suggestions["source"].isin({"log_ei", "qlog_ei"}).all()
    assert pd.to_numeric(suggestions["predicted_mean"]).map(pd.notna).all()
    pd.testing.assert_frame_equal(log, before)


def test_context_summary_reports_context_counts_best_rows_and_pending() -> None:
    cfg = contextual_config()
    log = observed_contextual_log(cfg)
    pending = {
        "row_id": "pending_0",
        "iteration": 2,
        "status": "suggested",
        "source": "log_ei",
        "catalyst_loading": 0.4,
        "feedstock_acidity": 0.2,
        "yield_score": "",
        "predicted_mean": 0.7,
        "predicted_std": 0.1,
        "acquisition": 0.01,
    }
    log.loc[len(log)] = [pending[column] for column in canonical_columns(cfg)]

    summary = context_summary(cfg, log)

    assert summary["context_key"].tolist() == [
        "feedstock_acidity=0.2",
        "feedstock_acidity=0.8",
    ]
    first = summary.loc[summary["feedstock_acidity"] == "0.2"].iloc[0]
    second = summary.loc[summary["feedstock_acidity"] == "0.8"].iloc[0]
    assert int(first["observed_rows"]) == 2
    assert int(first["pending_suggestions"]) == 1
    assert first["best_row_id"] == "obs_1"
    assert float(first["best_objective"]) == pytest.approx(0.8)
    assert int(second["observed_rows"]) == 2
    assert int(second["pending_suggestions"]) == 0
    assert second["best_row_id"] == "obs_3"


def test_context_summary_handles_empty_contextual_log() -> None:
    cfg = contextual_config()

    summary = context_summary(cfg, empty_campaign_log(cfg))

    assert summary.empty
    assert summary.columns.tolist() == [
        "context_key",
        "feedstock_acidity",
        "observed_rows",
        "pending_suggestions",
        "best_row_id",
        "best_objective",
    ]


def test_context_summary_handles_pending_only_contextual_log() -> None:
    cfg = contextual_config()
    pending = {
        "row_id": "pending_0",
        "iteration": 0,
        "status": "suggested",
        "source": "sobol",
        "catalyst_loading": 0.4,
        "feedstock_acidity": 0.25,
        "yield_score": "",
        "predicted_mean": "",
        "predicted_std": "",
        "acquisition": "",
    }
    log = pd.DataFrame(
        [[pending[column] for column in canonical_columns(cfg)]],
        columns=canonical_columns(cfg),
    )

    summary = context_summary(cfg, log)

    assert summary["context_key"].tolist() == ["feedstock_acidity=0.25"]
    row = summary.iloc[0]
    assert row["feedstock_acidity"] == "0.25"
    assert int(row["observed_rows"]) == 0
    assert int(row["pending_suggestions"]) == 1
    assert row["best_row_id"] is None
    assert row["best_objective"] is None


def test_context_summary_is_direction_aware_for_minimization() -> None:
    cfg = replace(
        contextual_config(),
        objective=ObjectiveConfig("yield_score", "minimize"),
    )
    log = observed_contextual_log(cfg)

    summary = context_summary(cfg, log)

    first = summary.loc[summary["feedstock_acidity"] == "0.2"].iloc[0]
    assert first["best_row_id"] == "obs_0"
    assert float(first["best_objective"]) == pytest.approx(0.5)


def test_context_summary_counts_only_review_blocking_pending_suggestions() -> None:
    cfg = replace(contextual_config(), review=ReviewConfig(enabled=True))
    rows = []
    for row in observed_contextual_log(contextual_config()).to_dict("records"):
        row.update({"review_status": "accepted", "review_note": ""})
        rows.append(row)
    for index, (status, acidity) in enumerate(
        [
            ("pending", 0.2),
            ("accepted", 0.2),
            ("rejected", 0.2),
            ("deferred", 0.8),
        ],
        start=1,
    ):
        rows.append(
            {
                "row_id": f"{status}_suggestion",
                "iteration": 2,
                "status": "suggested",
                "source": "log_ei",
                "review_status": status,
                "review_note": "",
                "catalyst_loading": 0.35 + 0.05 * index,
                "feedstock_acidity": acidity,
                "yield_score": "",
                "predicted_mean": 0.7,
                "predicted_std": 0.1,
                "acquisition": 0.01,
            }
        )
    log = pd.DataFrame(rows, columns=canonical_columns(cfg))

    summary = context_summary(cfg, log)

    first = summary.loc[summary["feedstock_acidity"] == "0.2"].iloc[0]
    second = summary.loc[summary["feedstock_acidity"] == "0.8"].iloc[0]
    assert int(first["pending_suggestions"]) == 2
    assert int(second["pending_suggestions"]) == 0


def test_context_summary_rejects_non_contextual_config() -> None:
    cfg = replace(contextual_config(), context=None)
    df = pd.DataFrame(columns=canonical_columns(cfg))

    with pytest.raises(ValueError, match="requires a config with a context section"):
        context_summary(cfg, df)


def test_missing_context_values_fail_before_suggestion() -> None:
    cfg = contextual_config(default=None)

    with pytest.raises(SuggestionError, match="missing"):
        suggest_next(cfg, empty_campaign_log(cfg), batch_size=1)


def test_pending_suggestions_block_before_missing_context_values() -> None:
    cfg = contextual_config(default=None)
    log = empty_campaign_log(cfg)
    log.loc[len(log)] = [
        "pending_0",
        0,
        "suggested",
        "sobol",
        0.25,
        0.5,
        "",
        "",
        "",
        "",
    ]

    with pytest.raises(SuggestionError, match="unresolved status='suggested'"):
        suggest_next(cfg, log, batch_size=1)


def test_same_decision_in_different_contexts_is_not_duplicate() -> None:
    cfg = contextual_config()
    log = observed_contextual_log(cfg)
    log.loc[1, "catalyst_loading"] = log.loc[0, "catalyst_loading"]
    log.loc[1, "feedstock_acidity"] = 0.8

    validate_campaign_data(cfg, log)


def test_constraints_are_evaluated_with_fixed_context_values() -> None:
    cfg = replace(
        contextual_config(default=None),
        constraints=(
            ConstraintConfig(
                name="loading_context_limit",
                expression="catalyst_loading + feedstock_acidity <= 1.0",
            ),
        ),
    )

    suggestions = suggest_next(
        cfg,
        empty_campaign_log(cfg),
        batch_size=2,
        context_values={"feedstock_acidity": 0.9},
    )

    assert (suggestions["catalyst_loading"].astype(float) <= 0.1 + 1e-12).all()


def test_contextual_session_summary_and_next_action_include_context_values(tmp_path) -> None:
    cfg = contextual_config(default=None)
    config_path = tmp_path / "context.yaml"
    config_path.write_text(
        """
campaign_name: contextual_test
objective: {name: yield_score, direction: maximize}
variables:
  - {name: catalyst_loading, type: continuous, lower: 0, upper: 1}
  - {name: feedstock_acidity, type: continuous, lower: 0, upper: 1}
context:
  variables: [feedstock_acidity]
bo: {batch_size: 1, initial_design_size: 4, acquisition: log_ei}
""",
        encoding="utf-8",
    )
    log_path = tmp_path / "context.csv"
    empty_campaign_log(cfg).to_csv(log_path, index=False)
    campaign = CampaignSession.from_files(config_path, log_path)

    summary = campaign.summary()
    next_action = campaign.next_action()

    assert summary.loc[summary["field"] == "contextual_campaign", "value"].iloc[0] is True
    assert "feedstock_acidity" in summary.loc[
        summary["field"] == "context_variables",
        "value",
    ].iloc[0]
    assert "context_values={...}" in next_action["suggested_call"].iloc[0]
