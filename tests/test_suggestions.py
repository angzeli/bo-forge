import math
from dataclasses import replace

import pandas as pd
import pytest
import torch

import bo_forge.suggestions as suggestions_module
from bo_forge.config import (
    BOConfig,
    CampaignConfig,
    ConstraintConfig,
    ContextConfig,
    CostConfig,
    FidelityConfig,
    ModelConfig,
    ObjectiveConfig,
    ReplicateConfig,
    ReviewConfig,
    StageConfig,
    VariableConfig,
)
from bo_forge.costs import evaluate_cost
from bo_forge.errors import SuggestionError
from bo_forge.io import empty_campaign_log
from bo_forge.logs import append_suggestions, load_campaign_log, mark_observed
from bo_forge.multi_objective import reference_point_to_model_space
from bo_forge.suggestions import (
    MAX_DECODE_RETRIES,
    suggest_next,
    suggestion_quality_summary,
)
from bo_forge.transforms import values_to_unit_cube
from bo_forge.validation import canonical_columns


def config(batch_size: int = 2, initial_design_size: int = 3) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="test",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("temperature", "continuous", 300.0, 800.0),
        ),
        bo=BOConfig(
            batch_size=batch_size,
            initial_design_size=initial_design_size,
            random_seed=3,
            raw_samples=16,
            num_restarts=2,
            mc_samples=16,
        ),
    )


def qlog_nei_config(*, review: bool = False, initial_design_size: int = 3) -> CampaignConfig:
    base = config(batch_size=1, initial_design_size=initial_design_size)
    return CampaignConfig(
        campaign_name="qlog_nei_test",
        objective=base.objective,
        variables=base.variables,
        bo=BOConfig(
            batch_size=1,
            initial_design_size=initial_design_size,
            acquisition="qlog_nei",
            random_seed=3,
            raw_samples=8,
            num_restarts=1,
            mc_samples=8,
        ),
        review=ReviewConfig(enabled=review),
    )


def qlog_nei_log(cfg: CampaignConfig) -> pd.DataFrame:
    rows = []
    for index, (x_value, temperature, activity) in enumerate(
        [
            (0.1, 350.0, 0.5),
            (0.3, 500.0, 1.1),
            (0.6, 650.0, 1.8),
            (0.9, 780.0, 1.2),
        ]
    ):
        row = {
            "row_id": f"obs_{index}",
            "iteration": index,
            "status": "observed",
            "source": "manual",
            "x": x_value,
            "temperature": temperature,
            "activity": activity,
            "predicted_mean": "",
            "predicted_std": "",
            "acquisition": "",
        }
        if cfg.review.enabled:
            row["review_status"] = "accepted"
            row["review_note"] = ""
        rows.append(row)
    return pd.DataFrame(rows, columns=canonical_columns(cfg))


def qlog_nehvi_config(*, review: bool = False, initial_design_size: int = 4) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="qlog_nehvi_test",
        objective=ObjectiveConfig("yield_score", "maximize", 40.0),
        objectives=(
            ObjectiveConfig("yield_score", "maximize", 40.0),
            ObjectiveConfig("waste_score", "minimize", 25.0),
        ),
        variables=(
            VariableConfig("temperature", "continuous", 20.0, 100.0),
            VariableConfig("solvent", "categorical", values=("MeCN", "Water")),
        ),
        bo=BOConfig(
            batch_size=1,
            initial_design_size=initial_design_size,
            acquisition="qlog_nehvi",
            random_seed=11,
            raw_samples=8,
            num_restarts=1,
            mc_samples=8,
        ),
        review=ReviewConfig(enabled=review),
    )


def qlog_nehvi_log(cfg: CampaignConfig) -> pd.DataFrame:
    rows = []
    for index, (temperature, solvent, yield_score, waste_score) in enumerate(
        [
            (30.0, "MeCN", 51.0, 22.0),
            (45.0, "Water", 62.0, 19.0),
            (65.0, "MeCN", 66.0, 15.0),
            (82.0, "Water", 70.0, 17.0),
        ]
    ):
        row = {
            "row_id": f"obs_{index}",
            "iteration": index,
            "status": "observed",
            "source": "manual",
            "temperature": temperature,
            "solvent": solvent,
            "yield_score": yield_score,
            "waste_score": waste_score,
            "predicted_mean_yield_score": "",
            "predicted_std_yield_score": "",
            "predicted_mean_waste_score": "",
            "predicted_std_waste_score": "",
            "acquisition": "",
        }
        if cfg.review.enabled:
            row["review_status"] = "accepted"
            row["review_note"] = ""
        rows.append(row)
    return pd.DataFrame(rows, columns=canonical_columns(cfg))


def qlog_nehvi_pending_row(
    cfg: CampaignConfig,
    *,
    row_id: str,
    review_status: str | None,
    temperature: float = 55.0,
    solvent: str = "Water",
    source: str = "qlog_nehvi",
) -> dict[str, object]:
    row: dict[str, object] = {
        "row_id": row_id,
        "iteration": 5,
        "status": "suggested",
        "source": source,
        "temperature": temperature,
        "solvent": solvent,
        "yield_score": "",
        "waste_score": "",
        "predicted_mean_yield_score": "",
        "predicted_std_yield_score": "",
        "predicted_mean_waste_score": "",
        "predicted_std_waste_score": "",
        "acquisition": "",
    }
    if cfg.review.enabled:
        row["review_status"] = review_status
        row["review_note"] = ""
    return row


def structured_config() -> CampaignConfig:
    cfg = config(batch_size=1, initial_design_size=1)
    return CampaignConfig(
        campaign_name="structured_test",
        objective=cfg.objective,
        variables=cfg.variables,
        bo=cfg.bo,
        stages=(
            StageConfig("screen", ("x",)),
            StageConfig("refine", ("x", "temperature")),
        ),
    )


def observed_log(cfg: CampaignConfig) -> pd.DataFrame:
    rows = []
    for index, (x_value, temperature, activity) in enumerate(
        [
            (0.1, 350.0, 0.5),
            (0.3, 500.0, 1.1),
            (0.6, 650.0, 1.8),
            (0.9, 780.0, 1.2),
        ]
    ):
        rows.append(
            {
                "row_id": f"obs_{index}",
                "iteration": index,
                "status": "observed",
                "source": "manual",
                "x": x_value,
                "temperature": temperature,
                "activity": activity,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        )
    return pd.DataFrame(rows)


def mixed_config(
    *,
    batch_size: int = 2,
    initial_design_size: int = 3,
    initial_design_method: str = "sobol",
) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="mixed",
        objective=ObjectiveConfig(name="score", direction="maximize"),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("repeats", "integer", 1.0, 3.0),
            VariableConfig("dose", "discrete", values=(0.1, 0.2, 0.5)),
            VariableConfig("solvent", "categorical", values=("MeCN", "EtOH")),
        ),
        bo=BOConfig(
            batch_size=batch_size,
            initial_design_size=initial_design_size,
            initial_design_method=initial_design_method,
            random_seed=3,
            raw_samples=16,
            num_restarts=2,
            mc_samples=16,
        ),
    )


def constrained_mixed_config(
    *,
    batch_size: int = 2,
    initial_design_size: int = 3,
    min_normalized_distance: float = 0.0,
) -> CampaignConfig:
    cfg = mixed_config(batch_size=batch_size, initial_design_size=initial_design_size)
    return CampaignConfig(
        campaign_name=cfg.campaign_name,
        objective=cfg.objective,
        variables=cfg.variables,
        bo=BOConfig(
            batch_size=cfg.bo.batch_size,
            initial_design_size=cfg.bo.initial_design_size,
            initial_design_method=cfg.bo.initial_design_method,
            random_seed=cfg.bo.random_seed,
            raw_samples=cfg.bo.raw_samples,
            num_restarts=cfg.bo.num_restarts,
            mc_samples=cfg.bo.mc_samples,
            min_normalized_distance=min_normalized_distance,
        ),
        constraints=(
            ConstraintConfig(
                name="no_etoh_high_dose",
                expression="not (solvent == 'EtOH' and dose >= 0.5)",
            ),
        ),
    )


def review_mixed_config(
    *,
    batch_size: int = 2,
    initial_design_size: int = 3,
) -> CampaignConfig:
    cfg = mixed_config(batch_size=batch_size, initial_design_size=initial_design_size)
    return CampaignConfig(
        campaign_name=cfg.campaign_name,
        objective=cfg.objective,
        variables=cfg.variables,
        bo=cfg.bo,
        review=ReviewConfig(enabled=True),
    )


def cost_review_mixed_config(
    *,
    batch_size: int = 2,
    initial_design_size: int = 3,
    budget: float | None = 50.0,
    weight: float = 0.5,
) -> CampaignConfig:
    cfg = mixed_config(batch_size=batch_size, initial_design_size=initial_design_size)
    return CampaignConfig(
        campaign_name=cfg.campaign_name,
        objective=cfg.objective,
        variables=cfg.variables,
        bo=cfg.bo,
        cost=CostConfig(
            expression="1.0 + 0.2 * repeats + 2.0 * (solvent == 'EtOH')",
            weight=weight,
            budget=budget,
            candidate_pool_size=16,
            top_k=8,
        ),
        review=ReviewConfig(enabled=True),
    )


def contextual_cost_review_config(
    *,
    batch_size: int = 1,
    initial_design_size: int = 4,
    budget: float | None = 90.0,
) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="contextual_cost_review",
        objective=ObjectiveConfig(name="yield_score", direction="maximize"),
        variables=(
            VariableConfig("catalyst_loading", "continuous", 0.0, 1.0),
            VariableConfig("reaction_temperature", "integer", 60, 120),
            VariableConfig("solvent", "categorical", values=("MeCN", "EtOH", "Water")),
            VariableConfig("feedstock_acidity", "continuous", 0.0, 1.0),
        ),
        bo=BOConfig(
            batch_size=batch_size,
            initial_design_size=initial_design_size,
            acquisition="log_ei",
            random_seed=23,
            raw_samples=8,
            num_restarts=1,
            mc_samples=8,
        ),
        cost=CostConfig(
            expression=(
                "1.0 + 0.03 * reaction_temperature + "
                "1.5 * (solvent == 'Water') + 0.8 * feedstock_acidity"
            ),
            weight=0.35,
            budget=budget,
            candidate_pool_size=12,
            top_k=6,
        ),
        review=ReviewConfig(enabled=True),
        context=ContextConfig(
            variables=("feedstock_acidity",),
            default_values={"feedstock_acidity": 0.5},
        ),
    )


def replicate_config(
    initial_design_size: int = 3,
    *,
    suggestion_policy: str = "uncertain_best",
    replicate_threshold: float = 0.10,
    min_repeats_at_best: int = 3,
    max_repeats_per_group: int = 5,
) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="replicate_test",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("temperature", "continuous", 300.0, 800.0),
        ),
        bo=BOConfig(
            batch_size=1,
            initial_design_size=initial_design_size,
            random_seed=3,
            raw_samples=16,
            num_restarts=2,
            mc_samples=16,
        ),
        replicates=ReplicateConfig(
            enabled=True,
            suggestion_policy=suggestion_policy,
            replicate_threshold=replicate_threshold,
            min_repeats_at_best=min_repeats_at_best,
            max_repeats_per_group=max_repeats_per_group,
        ),
    )


def multi_fidelity_config(initial_design_size: int = 3) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="multi_fidelity_test",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("fidelity", "continuous", 0.2, 1.0),
        ),
        bo=BOConfig(
            batch_size=1,
            initial_design_size=initial_design_size,
            acquisition="qmf_kg",
            random_seed=5,
            raw_samples=8,
            num_restarts=1,
            mc_samples=8,
        ),
        fidelity=FidelityConfig(
            variable="fidelity",
            target=1.0,
            num_fantasies=8,
        ),
    )


def multi_fidelity_observed_log(cfg: CampaignConfig) -> pd.DataFrame:
    rows = []
    for index, (x_value, fidelity, activity) in enumerate(
        [
            (0.10, 0.25, 0.7),
            (0.30, 0.50, 1.1),
            (0.60, 0.75, 1.4),
            (0.85, 1.00, 1.3),
        ]
    ):
        rows.append(
            {
                "row_id": f"mf_obs_{index}",
                "iteration": index,
                "status": "observed",
                "source": "manual",
                "x": x_value,
                "fidelity": fidelity,
                "activity": activity,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        )
    return pd.DataFrame(rows, columns=canonical_columns(cfg))


def mixed_observed_log(cfg: CampaignConfig) -> pd.DataFrame:
    rows = []
    for index, (x_value, repeats, dose, solvent, score) in enumerate(
        [
            (0.1, 1, 0.1, "MeCN", 1.0),
            (0.3, 2, 0.2, "EtOH", 1.4),
            (0.8, 3, 0.5, "MeCN", 1.2),
            (0.6, 2, 0.2, "MeCN", 1.8),
        ]
    ):
        rows.append(
            {
                "row_id": f"mixed_obs_{index}",
                "iteration": index,
                "status": "observed",
                "source": "manual",
                "x": x_value,
                "repeats": repeats,
                "dose": dose,
                "solvent": solvent,
                "score": score,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        )
    return pd.DataFrame(rows, columns=canonical_columns(cfg))


def cost_review_mixed_observed_log(cfg: CampaignConfig) -> pd.DataFrame:
    rows = []
    for index, (x_value, repeats, dose, solvent, score, cost_estimate, cost_actual) in enumerate(
        [
            (0.1, 1, 0.1, "MeCN", 1.0, 1.2, 1.1),
            (0.3, 2, 0.2, "EtOH", 1.4, 3.4, ""),
            (0.8, 3, 0.5, "MeCN", 1.2, 1.6, 1.7),
            (0.6, 2, 0.2, "MeCN", 1.8, 1.4, ""),
        ]
    ):
        rows.append(
            {
                "row_id": f"mixed_obs_{index}",
                "iteration": index,
                "status": "observed",
                "source": "manual",
                "review_status": "accepted",
                "review_note": "",
                "x": x_value,
                "repeats": repeats,
                "dose": dose,
                "solvent": solvent,
                "score": score,
                "cost_estimate": cost_estimate,
                "cost_actual": cost_actual,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
                "utility": "",
            }
        )
    return pd.DataFrame(rows, columns=canonical_columns(cfg))


def contextual_cost_review_observed_log(cfg: CampaignConfig) -> pd.DataFrame:
    rows = []
    for index, (
        loading,
        temperature,
        solvent,
        acidity,
        yield_score,
        cost_estimate,
        cost_actual,
    ) in enumerate(
        [
            (0.20, 70, "MeCN", 0.25, 0.64, 3.3, 3.4),
            (0.55, 90, "EtOH", 0.25, 0.83, 3.9, 3.8),
            (0.35, 100, "Water", 0.65, 0.60, 6.02, 6.1),
            (0.75, 110, "MeCN", 0.65, 0.78, 4.82, 4.9),
        ]
    ):
        rows.append(
            {
                "row_id": f"ctx_cost_obs_{index}",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "review_status": "accepted",
                "review_note": "",
                "catalyst_loading": loading,
                "reaction_temperature": temperature,
                "solvent": solvent,
                "feedstock_acidity": acidity,
                "yield_score": yield_score,
                "cost_estimate": cost_estimate,
                "cost_actual": cost_actual,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
                "utility": "",
            }
        )
    return pd.DataFrame(rows, columns=canonical_columns(cfg))


def replicate_observed_log(cfg: CampaignConfig) -> pd.DataFrame:
    rows = []
    for row_id, iteration, group, replicate_index, x_value, temperature, activity in [
        ("rep_0a", 0, "group_0", 0, 0.1, 350.0, 0.5),
        ("rep_0b", 0, "group_0", 1, 0.1, 350.0, 0.9),
        ("rep_1a", 1, "group_1", 0, 0.4, 550.0, 1.4),
        ("rep_2a", 2, "group_2", 0, 0.8, 720.0, 1.2),
    ]:
        rows.append(
            {
                "row_id": row_id,
                "iteration": iteration,
                "status": "observed",
                "source": "manual",
                "replicate_group": group,
                "replicate_index": replicate_index,
                "x": x_value,
                "temperature": temperature,
                "activity": activity,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        )
    return pd.DataFrame(rows, columns=canonical_columns(cfg))


def test_suggest_next_returns_sobol_initial_suggestions() -> None:
    cfg = config(batch_size=2, initial_design_size=3)
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 2
    assert set(suggestions["status"]) == {"suggested"}
    assert set(suggestions["source"]) == {"sobol"}
    assert suggestions["x"].astype(float).between(0.0, 1.0).all()
    assert suggestions["temperature"].astype(float).between(300.0, 800.0).all()
    assert suggestions["activity"].astype(str).eq("").all()


def test_multi_fidelity_initial_design_includes_fidelity_values() -> None:
    cfg = multi_fidelity_config(initial_design_size=3)
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(cfg, df, batch_size=2)

    assert len(suggestions) == 2
    assert set(suggestions["source"]) == {"sobol"}
    assert suggestions["x"].astype(float).between(0.0, 1.0).all()
    assert suggestions["fidelity"].astype(float).between(0.2, 1.0).all()
    assert list(suggestions.columns) == canonical_columns(cfg)


def test_structured_suggest_requires_stage_when_ambiguous() -> None:
    cfg = structured_config()
    df = empty_campaign_log(cfg)

    with pytest.raises(SuggestionError, match="require an explicit stage"):
        suggest_next(cfg, df)


def test_structured_suggest_rejects_unknown_stage() -> None:
    cfg = structured_config()
    df = empty_campaign_log(cfg)

    with pytest.raises(SuggestionError, match="Unknown structured campaign stage 'unknown'"):
        suggest_next(cfg, df, stage="unknown")


def test_structured_suggest_rejects_stage_with_no_active_variables() -> None:
    cfg = CampaignConfig(
        campaign_name="bad_structured",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=BOConfig(batch_size=1, initial_design_size=1),
        stages=(StageConfig("empty", ()),),
    )
    df = empty_campaign_log(cfg)

    with pytest.raises(SuggestionError, match="has no active variables"):
        suggest_next(cfg, df, stage="empty")


def test_structured_suggest_with_cost_fails_with_current_version_message() -> None:
    base = structured_config()
    cfg = CampaignConfig(
        campaign_name=base.campaign_name,
        objective=base.objective,
        variables=base.variables,
        bo=base.bo,
        cost=CostConfig(expression="1.0"),
        stages=base.stages,
    )
    df = empty_campaign_log(cfg)

    with pytest.raises(
        SuggestionError,
        match="Structured campaign suggestions with cost are not supported in v1.4.0",
    ):
        suggest_next(cfg, df, stage="screen")


def test_structured_suggest_populates_stage_and_only_active_variables() -> None:
    cfg = structured_config()
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(cfg, df, stage="screen")

    assert len(suggestions) == 1
    assert suggestions.loc[0, "stage"] == "screen"
    assert suggestions.loc[0, "x"] != ""
    assert suggestions.loc[0, "temperature"] == ""
    assert suggestions.loc[0, "activity"] == ""


def test_structured_model_based_suggest_uses_selected_stage() -> None:
    cfg = structured_config()
    df = pd.DataFrame(
        [
            {
                "row_id": "screen_obs_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "stage": "screen",
                "x": 0.2,
                "temperature": "",
                "activity": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "screen_obs_1",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "stage": "screen",
                "x": 0.8,
                "temperature": "",
                "activity": 1.3,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )

    suggestions = suggest_next(cfg, df, stage="screen")

    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "log_ei"
    assert suggestions.loc[0, "stage"] == "screen"
    assert suggestions.loc[0, "x"] != ""
    assert suggestions.loc[0, "temperature"] == ""
    assert math.isfinite(float(suggestions.loc[0, "predicted_mean"]))
    assert float(suggestions.loc[0, "predicted_std"]) >= 0.0
    assert math.isfinite(float(suggestions.loc[0, "acquisition"]))


def test_structured_single_stage_can_infer_stage() -> None:
    base = structured_config()
    cfg = CampaignConfig(
        campaign_name=base.campaign_name,
        objective=base.objective,
        variables=base.variables,
        bo=base.bo,
        stages=(StageConfig("screen", ("x",)),),
    )
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert suggestions.loc[0, "stage"] == "screen"
    assert suggestions.loc[0, "temperature"] == ""


def test_structured_suggest_ignores_constraints_with_inactive_variables() -> None:
    base = structured_config()
    cfg = CampaignConfig(
        campaign_name=base.campaign_name,
        objective=base.objective,
        variables=base.variables,
        bo=base.bo,
        constraints=(ConstraintConfig("temperature_low", "temperature <= 300"),),
        stages=base.stages,
    )
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(cfg, df, stage="screen")

    assert suggestions.loc[0, "stage"] == "screen"
    assert suggestions.loc[0, "temperature"] == ""


def test_structured_suggest_applies_constraints_with_active_variables() -> None:
    cfg = CampaignConfig(
        campaign_name="structured_active_constraint",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(VariableConfig("solvent", "categorical", values=("A",)),),
        bo=BOConfig(batch_size=1, initial_design_size=1, random_seed=3),
        constraints=(ConstraintConfig("only_a", "solvent == 'A'"),),
        stages=(StageConfig("screen", ("solvent",)),),
    )
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(cfg, df, stage="screen")

    assert suggestions.loc[0, "solvent"] == "A"


def test_structured_duplicate_checks_are_stage_aware() -> None:
    cfg = CampaignConfig(
        campaign_name="structured_duplicate",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(VariableConfig("solvent", "categorical", values=("A",)),),
        bo=BOConfig(batch_size=1, initial_design_size=1, random_seed=3),
        stages=(
            StageConfig("screen", ("solvent",)),
            StageConfig("refine", ("solvent",)),
        ),
    )
    df = pd.DataFrame(
        [
            {
                "row_id": "refine_obs",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "stage": "refine",
                "solvent": "A",
                "activity": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )

    suggestions = suggest_next(cfg, df, stage="screen")

    assert suggestions.loc[0, "stage"] == "screen"
    assert suggestions.loc[0, "solvent"] == "A"


def test_replicate_suggestions_set_group_to_row_id_and_start_at_zero() -> None:
    cfg = replicate_config(initial_design_size=4)
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 1
    assert suggestions.loc[0, "replicate_group"] == suggestions.loc[0, "row_id"]
    assert int(suggestions.loc[0, "replicate_index"]) == 0


def test_replicate_initial_design_counts_groups_not_raw_rows() -> None:
    cfg = replicate_config(initial_design_size=4)
    df = replicate_observed_log(cfg).astype(object)

    suggestions = suggest_next(cfg, df)

    assert suggestions.loc[0, "source"] == "sobol"


def test_replicate_model_based_suggestions_use_aggregated_observations() -> None:
    cfg = replicate_config(initial_design_size=3, suggestion_policy="new_only")
    df = replicate_observed_log(cfg).astype(object)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "log_ei"
    assert suggestions.loc[0, "replicate_group"] == suggestions.loc[0, "row_id"]
    existing_designs = {
        (float(row["x"]), float(row["temperature"]))
        for _, row in df.iterrows()
    }
    suggested_design = (
        float(suggestions.loc[0, "x"]),
        float(suggestions.loc[0, "temperature"]),
    )
    assert suggested_design not in existing_designs


def test_replicate_uncertain_best_policy_suggests_same_group_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replicate_config(initial_design_size=3, replicate_threshold=0.10)
    df = replicate_observed_log(cfg)

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0], [1.0]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04], [0.01]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "log_ei"
    assert suggestions.loc[0, "replicate_group"] == "group_1"
    assert int(suggestions.loc[0, "replicate_index"]) == 1
    assert float(suggestions.loc[0, "x"]) == pytest.approx(0.4)
    assert float(suggestions.loc[0, "temperature"]) == pytest.approx(550.0)
    assert float(suggestions.loc[0, "predicted_mean"]) == pytest.approx(2.0)
    assert float(suggestions.loc[0, "predicted_std"]) == pytest.approx(0.2)


def test_uncertain_best_uses_next_replicate_index_from_all_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = replicate_config(initial_design_size=3, replicate_threshold=0.10)
    cfg = CampaignConfig(
        campaign_name=base.campaign_name,
        objective=base.objective,
        variables=base.variables,
        bo=base.bo,
        review=ReviewConfig(enabled=True),
        replicates=base.replicates,
    )
    df = replicate_observed_log(cfg).astype(object)
    df.loc[:, "review_status"] = "accepted"
    df.loc[:, "review_note"] = ""
    rejected = df.loc[df["replicate_group"] == "group_1"].iloc[[0]].copy()
    rejected.loc[:, "row_id"] = "rejected_repeat"
    rejected.loc[:, "status"] = "suggested"
    rejected.loc[:, "review_status"] = "rejected"
    rejected.loc[:, "review_note"] = "do not run"
    rejected.loc[:, "replicate_index"] = 1
    rejected.loc[:, "activity"] = ""
    df = pd.concat([df, rejected], ignore_index=True)

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0], [1.0]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04], [0.01]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())

    suggestions = suggest_next(cfg, df)

    assert suggestions.loc[0, "replicate_group"] == "group_1"
    assert int(suggestions.loc[0, "replicate_index"]) == 2


def test_replicate_repeat_suggestion_round_trips_as_same_group_duplicate(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replicate_config(initial_design_size=3, replicate_threshold=0.10)
    df = replicate_observed_log(cfg)

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0], [1.0]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04], [0.01]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    suggestions = suggest_next(cfg, df)
    log_path = tmp_path / "campaign.csv"
    df.to_csv(log_path, index=False)

    append_suggestions(log_path, suggestions, config=cfg)
    mark_observed(log_path, str(suggestions.loc[0, "row_id"]), objective_value=1.45)

    written = load_campaign_log(log_path, cfg)
    repeated = written.loc[written["replicate_group"] == "group_1"]
    assert len(repeated) == 2
    assert sorted(repeated["replicate_index"].astype(int).tolist()) == [0, 1]


def test_suggestion_quality_marks_intentional_replicate_without_duplicate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replicate_config(initial_design_size=3, replicate_threshold=0.10)
    df = replicate_observed_log(cfg)

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0], [1.0]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04], [0.01]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    suggestions = suggest_next(cfg, df)

    summary = suggestion_quality_summary(cfg, df, suggestions)

    assert bool(summary.loc[0, "duplicate_allowed_by_replicates"])
    assert not bool(summary.loc[0, "is_exact_duplicate"])
    assert summary.loc[0, "nearest_existing_distance"] == pytest.approx(0.0)
    assert bool(summary.loc[0, "passes_distance_threshold"])


def test_suggestion_quality_allows_intentional_replicate_batch_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replicate_config(
        initial_design_size=3,
        replicate_threshold=0.10,
        min_repeats_at_best=3,
    )
    df = replicate_observed_log(cfg)

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0], [1.0]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04], [0.01]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    suggestions = suggest_next(cfg, df, batch_size=2)

    summary = suggestion_quality_summary(cfg, df, suggestions)

    assert suggestions["replicate_index"].astype(int).tolist() == [1, 2]
    assert summary["duplicate_allowed_by_replicates"].astype(bool).all()
    assert not summary["is_exact_duplicate"].astype(bool).any()
    assert summary["passes_distance_threshold"].astype(bool).all()


def test_cost_replicate_uncertain_best_fills_cost_without_utility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = replicate_config(initial_design_size=3, replicate_threshold=0.10)
    cfg = CampaignConfig(
        campaign_name=base.campaign_name,
        objective=base.objective,
        variables=base.variables,
        bo=base.bo,
        cost=CostConfig(
            expression="1.0 + x",
            weight=0.5,
            budget=100.0,
            candidate_pool_size=16,
            top_k=8,
        ),
        replicates=base.replicates,
    )
    df = replicate_observed_log(cfg)

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0], [1.0]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04], [0.01]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())

    suggestions = suggest_next(cfg, df)

    assert suggestions.loc[0, "source"] == "log_ei"
    assert float(suggestions.loc[0, "cost_estimate"]) == pytest.approx(1.4)
    assert str(suggestions.loc[0, "utility"]) == ""


def test_uncertain_best_fills_remaining_batch_with_exploration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replicate_config(
        initial_design_size=3,
        replicate_threshold=0.10,
        min_repeats_at_best=2,
    )
    df = replicate_observed_log(cfg)
    captured: dict[str, pd.DataFrame] = {}

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0], [1.0]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04], [0.01]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    def fake_model_based(**kwargs):
        captured["df"] = kwargs["df"]
        row = {
            "row_id": "new_suggestion",
            "iteration": 99,
            "status": "suggested",
            "source": "log_ei",
            "replicate_group": "new_suggestion",
            "replicate_index": 0,
            "x": 0.9,
            "temperature": 700.0,
            "activity": "",
            "predicted_mean": 1.5,
            "predicted_std": 0.1,
            "acquisition": 0.2,
        }
        return pd.DataFrame([row], columns=canonical_columns(cfg))

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    monkeypatch.setattr(suggestions_module, "_suggest_model_based", fake_model_based)

    suggestions = suggest_next(cfg, df, batch_size=2)

    assert len(suggestions) == 2
    assert suggestions.loc[0, "replicate_group"] == "group_1"
    assert int(suggestions.loc[0, "replicate_index"]) == 1
    assert suggestions.loc[1, "replicate_group"] == "new_suggestion"
    assert suggestions["iteration"].astype(int).nunique() == 1
    staged = captured["df"].loc[captured["df"]["replicate_group"] == "group_1"]
    assert sorted(staged["replicate_index"].astype(int).tolist()) == [0, 1]


def test_uncertain_best_batch_fill_avoids_duplicate_repeat_design(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replicate_config(
        initial_design_size=3,
        replicate_threshold=0.10,
        min_repeats_at_best=2,
    )
    df = replicate_observed_log(cfg)

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0], [1.0]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04], [0.01]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    def fake_model_based(**kwargs):
        staged_df = kwargs["df"]
        repeat_design = staged_df.loc[
            staged_df["replicate_group"] == "group_1",
            ["x", "temperature"],
        ].iloc[-1]
        assert float(repeat_design["x"]) == pytest.approx(0.4)
        assert float(repeat_design["temperature"]) == pytest.approx(550.0)
        row = {
            "row_id": "new_suggestion",
            "iteration": 99,
            "status": "suggested",
            "source": "log_ei",
            "replicate_group": "new_suggestion",
            "replicate_index": 0,
            "x": 0.95,
            "temperature": 730.0,
            "activity": "",
            "predicted_mean": 1.5,
            "predicted_std": 0.1,
            "acquisition": 0.2,
        }
        return pd.DataFrame([row], columns=canonical_columns(cfg))

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    monkeypatch.setattr(suggestions_module, "_suggest_model_based", fake_model_based)

    suggestions = suggest_next(cfg, df, batch_size=2)

    designs = {
        (float(row["x"]), float(row["temperature"]))
        for _, row in suggestions.iterrows()
    }
    assert designs == {(0.4, 550.0), (0.95, 730.0)}


def test_cost_aware_uncertain_best_repeat_then_cost_exploration_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = replicate_config(
        initial_design_size=3,
        replicate_threshold=0.10,
        min_repeats_at_best=2,
    )
    cfg = CampaignConfig(
        campaign_name=base.campaign_name,
        objective=base.objective,
        variables=base.variables,
        bo=base.bo,
        cost=CostConfig(
            expression="1.0 + x",
            weight=0.5,
            budget=10.0,
            candidate_pool_size=16,
            top_k=8,
        ),
        replicates=base.replicates,
    )
    df = replicate_observed_log(cfg)
    captured: dict[str, object] = {}

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0], [1.0]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04], [0.01]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    def fake_cost_aware(**kwargs):
        captured["config"] = kwargs["config"]
        captured["df"] = kwargs["df"]
        row = {
            "row_id": "cost_suggestion",
            "iteration": 99,
            "status": "suggested",
            "source": "cost_log_ei",
            "replicate_group": "cost_suggestion",
            "replicate_index": 0,
            "x": 0.9,
            "temperature": 700.0,
            "activity": "",
            "cost_estimate": 1.9,
            "cost_actual": "",
            "predicted_mean": 1.5,
            "predicted_std": 0.1,
            "acquisition": 0.8,
            "utility": -0.15,
        }
        return pd.DataFrame([row], columns=canonical_columns(cfg))

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    monkeypatch.setattr(suggestions_module, "_suggest_cost_aware_model_based", fake_cost_aware)

    suggestions = suggest_next(cfg, df, batch_size=2)

    assert len(suggestions) == 2
    assert suggestions.loc[0, "source"] == "log_ei"
    assert str(suggestions.loc[0, "utility"]) == ""
    assert suggestions.loc[1, "source"] == "cost_log_ei"
    assert captured["config"].cost.budget == pytest.approx(8.6)
    staged = captured["df"].loc[captured["df"]["replicate_group"] == "group_1"]
    assert sorted(staged["replicate_index"].astype(int).tolist()) == [0, 1]


def test_repeat_batch_fill_underfills_on_candidate_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = replicate_config(
        initial_design_size=3,
        replicate_threshold=0.10,
        min_repeats_at_best=2,
    )
    cfg = CampaignConfig(
        campaign_name=base.campaign_name,
        objective=base.objective,
        variables=base.variables,
        bo=base.bo,
        cost=CostConfig(
            expression="1.0 + x",
            weight=0.5,
            budget=6.81,
            candidate_pool_size=16,
            top_k=8,
        ),
        replicates=base.replicates,
    )
    df = replicate_observed_log(cfg)

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0], [1.0]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04], [0.01]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    def exhausted_cost_aware(**_kwargs):
        raise suggestions_module._CandidateGenerationExhausted(
            "remaining budget exhausted"
        )

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    monkeypatch.setattr(
        suggestions_module,
        "_suggest_cost_aware_model_based",
        exhausted_cost_aware,
    )

    suggestions = suggest_next(cfg, df, batch_size=2)

    assert len(suggestions) == 1
    assert suggestions.loc[0, "replicate_group"] == "group_1"
    assert int(suggestions.loc[0, "replicate_index"]) == 1
    assert float(suggestions.loc[0, "cost_estimate"]) == pytest.approx(1.4)


def test_repeat_batch_fill_reraises_unexpected_suggestion_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replicate_config(
        initial_design_size=3,
        replicate_threshold=0.10,
        min_repeats_at_best=2,
    )
    df = replicate_observed_log(cfg)

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0], [1.0]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04], [0.01]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    def unexpected_model_based(**_kwargs):
        raise SuggestionError("unexpected internal issue")

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    monkeypatch.setattr(
        suggestions_module,
        "_suggest_model_based",
        unexpected_model_based,
    )

    with pytest.raises(SuggestionError) as exc_info:
        suggest_next(cfg, df, batch_size=2)
    message = str(exc_info.value)
    assert "Repeat suggestions were generated, but exploration fill failed" in message
    assert "unexpected internal issue" in message


def test_replicate_uncertain_best_policy_respects_max_repeat_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replicate_config(
        initial_design_size=3,
        replicate_threshold=0.10,
        max_repeats_per_group=5,
    )
    df = replicate_observed_log(cfg)
    extra_rows = []
    for replicate_index in range(1, 5):
        row = df.loc[df["replicate_group"] == "group_1"].iloc[0].copy()
        row["row_id"] = f"rep_1_extra_{replicate_index}"
        row["replicate_index"] = replicate_index
        row["activity"] = 1.4 + 0.01 * replicate_index
        extra_rows.append(row)
    df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0], [1.0]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04], [0.01]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    fallback = pd.DataFrame(
        [
            {
                "row_id": "new_suggestion",
                "iteration": 99,
                "status": "suggested",
                "source": "log_ei",
                "replicate_group": "new_suggestion",
                "replicate_index": 0,
                "x": 0.9,
                "temperature": 700.0,
                "activity": "",
                "predicted_mean": 1.5,
                "predicted_std": 0.1,
                "acquisition": 0.2,
            }
        ],
        columns=canonical_columns(cfg),
    )
    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    monkeypatch.setattr(suggestions_module, "_suggest_model_based", lambda **_kwargs: fallback)

    suggestions = suggest_next(cfg, df)

    assert suggestions.loc[0, "replicate_group"] == "new_suggestion"


def test_replicate_new_only_policy_skips_active_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replicate_config(initial_design_size=3, suggestion_policy="new_only")
    df = replicate_observed_log(cfg)
    fallback = pd.DataFrame(
        [
            {
                "row_id": "new_suggestion",
                "iteration": 99,
                "status": "suggested",
                "source": "log_ei",
                "replicate_group": "new_suggestion",
                "replicate_index": 0,
                "x": 0.9,
                "temperature": 700.0,
                "activity": "",
                "predicted_mean": 1.5,
                "predicted_std": 0.1,
                "acquisition": 0.2,
            }
        ],
        columns=canonical_columns(cfg),
    )
    monkeypatch.setattr(suggestions_module, "_suggest_model_based", lambda **_kwargs: fallback)

    suggestions = suggest_next(cfg, df)

    assert suggestions.loc[0, "replicate_group"] == "new_suggestion"


def test_suggest_next_refuses_pending_suggestions() -> None:
    cfg = config(batch_size=1, initial_design_size=3)
    df = empty_campaign_log(cfg)
    pending = suggest_next(cfg, df)

    with pytest.raises(SuggestionError, match="unresolved status='suggested'"):
        suggest_next(cfg, pending)


def test_suggest_next_returns_model_based_single_suggestion() -> None:
    cfg = config(batch_size=1, initial_design_size=3)
    df = observed_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 1
    assert suggestions.loc[0, "status"] == "suggested"
    assert suggestions.loc[0, "source"] == "log_ei"
    assert float(suggestions.loc[0, "predicted_std"]) >= 0.0
    assert float(suggestions.loc[0, "x"]) >= 0.0
    assert float(suggestions.loc[0, "x"]) <= 1.0


def test_suggest_next_supports_non_default_model_profile_without_mutating() -> None:
    base = config(batch_size=1, initial_design_size=3)
    cfg = CampaignConfig(
        campaign_name=base.campaign_name,
        objective=base.objective,
        variables=base.variables,
        bo=base.bo,
        model=ModelConfig(profile="rough"),
    )
    df = observed_log(cfg)
    before = df.copy(deep=True)

    suggestions = suggest_next(cfg, df, batch_size=1)

    pd.testing.assert_frame_equal(df, before)
    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "log_ei"
    assert math.isfinite(float(suggestions.loc[0, "predicted_mean"]))
    assert float(suggestions.loc[0, "predicted_std"]) >= 0.0


def test_qlog_nei_suggestions_are_non_mutating_and_use_qlog_nei_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = qlog_nei_config()
    df = qlog_nei_log(cfg)
    before = df.copy(deep=True)
    candidate = values_to_unit_cube(cfg, [(0.45, 610.0)])

    def fake_optimizer(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        assert kwargs["x_baseline"].shape[0] == 4
        assert kwargs["x_pending"] is None
        return candidate, torch.tensor(0.25, dtype=torch.double), "qlog_nei"

    monkeypatch.setattr(suggestions_module, "optimize_qlog_nei", fake_optimizer)

    suggestions = suggest_next(cfg, df, batch_size=1)

    pd.testing.assert_frame_equal(df, before)
    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "qlog_nei"
    assert math.isfinite(float(suggestions.loc[0, "predicted_mean"]))
    assert float(suggestions.loc[0, "predicted_std"]) >= 0.0
    assert float(suggestions.loc[0, "acquisition"]) == pytest.approx(0.25)


def test_qlog_nei_passes_accepted_review_suggestions_as_x_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = qlog_nei_config(review=True)
    df = qlog_nei_log(cfg)
    pending = {
        "row_id": "pending_0",
        "iteration": 4,
        "status": "suggested",
        "source": "qlog_nei",
        "review_status": "accepted",
        "review_note": "",
        "x": 0.75,
        "temperature": 700.0,
        "activity": "",
        "predicted_mean": 1.4,
        "predicted_std": 0.2,
        "acquisition": 0.1,
    }
    df = pd.concat([df, pd.DataFrame([pending], columns=canonical_columns(cfg))], ignore_index=True)
    candidate = values_to_unit_cube(cfg, [(0.45, 610.0)])
    captured: dict[str, object] = {}

    def fake_optimizer(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        captured["x_pending"] = kwargs["x_pending"]
        return candidate, torch.tensor(0.25, dtype=torch.double), "qlog_nei"

    monkeypatch.setattr(suggestions_module, "optimize_qlog_nei", fake_optimizer)

    suggestions = suggest_next(cfg, df, batch_size=1)

    x_pending = captured["x_pending"]
    assert isinstance(x_pending, torch.Tensor)
    assert x_pending.shape == (1, 2)
    assert suggestions.loc[0, "source"] == "qlog_nei"


def test_qlog_nei_review_pending_rows_block_suggestions() -> None:
    cfg = qlog_nei_config(review=True)
    df = qlog_nei_log(cfg)
    pending = {
        "row_id": "review_pending",
        "iteration": 4,
        "status": "suggested",
        "source": "qlog_nei",
        "review_status": "pending",
        "review_note": "",
        "x": 0.75,
        "temperature": 700.0,
        "activity": "",
        "predicted_mean": 1.4,
        "predicted_std": 0.2,
        "acquisition": 0.1,
    }
    df = pd.concat([df, pd.DataFrame([pending], columns=canonical_columns(cfg))], ignore_index=True)

    with pytest.raises(SuggestionError, match="review_status='pending'"):
        suggest_next(cfg, df, batch_size=1)


@pytest.mark.parametrize("review_status", ["rejected", "deferred"])
def test_qlog_nei_rejected_and_deferred_review_rows_do_not_enter_x_pending(
    review_status: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = qlog_nei_config(review=True)
    df = qlog_nei_log(cfg)
    ignored = {
        "row_id": f"review_{review_status}",
        "iteration": 4,
        "status": "suggested",
        "source": "qlog_nei",
        "review_status": review_status,
        "review_note": "",
        "x": 0.75,
        "temperature": 700.0,
        "activity": "",
        "predicted_mean": 1.4,
        "predicted_std": 0.2,
        "acquisition": 0.1,
    }
    df = pd.concat([df, pd.DataFrame([ignored], columns=canonical_columns(cfg))], ignore_index=True)
    candidate = values_to_unit_cube(cfg, [(0.45, 610.0)])
    captured: dict[str, object] = {}

    def fake_optimizer(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        captured["x_pending"] = kwargs["x_pending"]
        return candidate, torch.tensor(0.25, dtype=torch.double), "qlog_nei"

    monkeypatch.setattr(suggestions_module, "optimize_qlog_nei", fake_optimizer)

    suggestions = suggest_next(cfg, df, batch_size=1)

    assert captured["x_pending"] is None
    assert suggestions.loc[0, "source"] == "qlog_nei"


def test_qlog_nei_waits_for_pending_initial_design_rows() -> None:
    cfg = qlog_nei_config(review=True, initial_design_size=4)
    df = qlog_nei_log(cfg).iloc[:3].copy()
    pending_initial = {
        "row_id": "initial_pending",
        "iteration": 3,
        "status": "suggested",
        "source": "sobol",
        "review_status": "accepted",
        "review_note": "",
        "x": 0.75,
        "temperature": 700.0,
        "activity": "",
        "predicted_mean": "",
        "predicted_std": "",
        "acquisition": "",
    }
    df = pd.concat(
        [df, pd.DataFrame([pending_initial], columns=canonical_columns(cfg))],
        ignore_index=True,
    )

    with pytest.raises(SuggestionError, match="observe accepted pending initial suggestions"):
        suggest_next(cfg, df, batch_size=1)


def test_qlog_nei_can_fill_remaining_initial_design_with_pending_initial_rows() -> None:
    cfg = qlog_nei_config(review=True, initial_design_size=4)
    df = qlog_nei_log(cfg).iloc[:2].copy()
    pending_initial = {
        "row_id": "initial_pending",
        "iteration": 2,
        "status": "suggested",
        "source": "sobol",
        "review_status": "accepted",
        "review_note": "",
        "x": 0.75,
        "temperature": 700.0,
        "activity": "",
        "predicted_mean": "",
        "predicted_std": "",
        "acquisition": "",
    }
    df = pd.concat(
        [df, pd.DataFrame([pending_initial], columns=canonical_columns(cfg))],
        ignore_index=True,
    )

    suggestions = suggest_next(cfg, df, batch_size=2)

    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "sobol"


def test_qlog_nehvi_suggestions_are_non_mutating_and_use_design_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = qlog_nehvi_config()
    df = qlog_nehvi_log(cfg)
    before = df.copy(deep=True)
    candidate = values_to_unit_cube(cfg, [(72.0, "MeCN")])

    def fake_optimizer(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        assert kwargs["x_baseline"].shape == (4, candidate.shape[1])
        assert kwargs["x_pending"] is None
        assert torch.equal(kwargs["ref_point"], reference_point_to_model_space(cfg))
        return candidate, torch.tensor(0.35, dtype=torch.double), "qlog_nehvi"

    monkeypatch.setattr(suggestions_module, "optimize_qlog_nehvi", fake_optimizer)

    suggestions = suggest_next(cfg, df, batch_size=1)

    pd.testing.assert_frame_equal(df, before)
    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "qlog_nehvi"
    assert suggestions[cfg.objective_names].map(lambda value: value == "").all().all()
    assert math.isfinite(float(suggestions.loc[0, "predicted_mean_yield_score"]))
    assert math.isfinite(float(suggestions.loc[0, "predicted_mean_waste_score"]))
    assert float(suggestions.loc[0, "acquisition"]) == pytest.approx(0.35)


def test_qlog_nehvi_non_review_pending_rows_enter_x_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = qlog_nehvi_config(review=False)
    df = qlog_nehvi_log(cfg)
    pending = qlog_nehvi_pending_row(
        cfg,
        row_id="pending_no_review",
        review_status=None,
        temperature=58.0,
        solvent="Water",
    )
    df = pd.concat([df, pd.DataFrame([pending], columns=canonical_columns(cfg))], ignore_index=True)
    candidate = values_to_unit_cube(cfg, [(72.0, "MeCN")])
    captured: dict[str, object] = {}

    def fake_optimizer(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        captured["x_pending"] = kwargs["x_pending"]
        return candidate, torch.tensor(0.35, dtype=torch.double), "qlog_nehvi"

    monkeypatch.setattr(suggestions_module, "optimize_qlog_nehvi", fake_optimizer)

    suggestions = suggest_next(cfg, df, batch_size=1)

    x_pending = captured["x_pending"]
    assert isinstance(x_pending, torch.Tensor)
    assert x_pending.shape == (1, candidate.shape[1])
    assert suggestions.loc[0, "source"] == "qlog_nehvi"


def test_qlog_nehvi_review_accepted_rows_enter_x_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = qlog_nehvi_config(review=True)
    df = qlog_nehvi_log(cfg)
    pending = qlog_nehvi_pending_row(
        cfg,
        row_id="accepted_pending",
        review_status="accepted",
        temperature=58.0,
        solvent="Water",
    )
    df = pd.concat([df, pd.DataFrame([pending], columns=canonical_columns(cfg))], ignore_index=True)
    candidate = values_to_unit_cube(cfg, [(72.0, "MeCN")])
    captured: dict[str, object] = {}

    def fake_optimizer(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        captured["x_pending"] = kwargs["x_pending"]
        return candidate, torch.tensor(0.35, dtype=torch.double), "qlog_nehvi"

    monkeypatch.setattr(suggestions_module, "optimize_qlog_nehvi", fake_optimizer)

    suggestions = suggest_next(cfg, df, batch_size=1)

    x_pending = captured["x_pending"]
    assert isinstance(x_pending, torch.Tensor)
    assert x_pending.shape == (1, candidate.shape[1])
    assert suggestions.loc[0, "source"] == "qlog_nehvi"


def test_qlog_nehvi_review_pending_rows_block_suggestions() -> None:
    cfg = qlog_nehvi_config(review=True)
    df = qlog_nehvi_log(cfg)
    pending = qlog_nehvi_pending_row(
        cfg,
        row_id="review_pending",
        review_status="pending",
        temperature=58.0,
        solvent="Water",
    )
    df = pd.concat([df, pd.DataFrame([pending], columns=canonical_columns(cfg))], ignore_index=True)

    with pytest.raises(SuggestionError, match="review_status='pending'"):
        suggest_next(cfg, df, batch_size=1)


@pytest.mark.parametrize("review_status", ["rejected", "deferred"])
def test_qlog_nehvi_rejected_and_deferred_rows_do_not_enter_x_pending(
    review_status: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = qlog_nehvi_config(review=True)
    df = qlog_nehvi_log(cfg)
    ignored = qlog_nehvi_pending_row(
        cfg,
        row_id=f"review_{review_status}",
        review_status=review_status,
        temperature=58.0,
        solvent="Water",
    )
    df = pd.concat([df, pd.DataFrame([ignored], columns=canonical_columns(cfg))], ignore_index=True)
    candidate = values_to_unit_cube(cfg, [(72.0, "MeCN")])
    captured: dict[str, object] = {}

    def fake_optimizer(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        captured["x_pending"] = kwargs["x_pending"]
        return candidate, torch.tensor(0.35, dtype=torch.double), "qlog_nehvi"

    monkeypatch.setattr(suggestions_module, "optimize_qlog_nehvi", fake_optimizer)

    suggestions = suggest_next(cfg, df, batch_size=1)

    assert captured["x_pending"] is None
    assert suggestions.loc[0, "source"] == "qlog_nehvi"


def test_qlog_nehvi_waits_for_pending_initial_design_rows() -> None:
    cfg = qlog_nehvi_config(review=True, initial_design_size=4)
    df = qlog_nehvi_log(cfg).iloc[:3].copy()
    pending_initial = qlog_nehvi_pending_row(
        cfg,
        row_id="initial_pending",
        review_status="accepted",
        source="sobol",
        temperature=58.0,
        solvent="Water",
    )
    df = pd.concat(
        [df, pd.DataFrame([pending_initial], columns=canonical_columns(cfg))],
        ignore_index=True,
    )

    with pytest.raises(SuggestionError, match="observe accepted pending initial suggestions"):
        suggest_next(cfg, df, batch_size=1)


def test_qlog_nei_mixed_variables_use_fixed_feature_assignments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = mixed_config(batch_size=1, initial_design_size=3)
    cfg = replace(
        base,
        bo=replace(
            base.bo,
            acquisition="qlog_nei",
            raw_samples=8,
            num_restarts=1,
            mc_samples=8,
        ),
    )
    df = mixed_observed_log(cfg)
    candidate = values_to_unit_cube(cfg, [(0.45, 2, 0.2, "EtOH")])
    captured: dict[str, object] = {}

    def fake_optimizer(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        captured["model_dim"] = kwargs["model_dim"]
        captured["fixed_features_list"] = kwargs["fixed_features_list"]
        return candidate, torch.tensor(0.25, dtype=torch.double), "qlog_nei"

    monkeypatch.setattr(suggestions_module, "optimize_qlog_nei", fake_optimizer)

    suggestions = suggest_next(cfg, df, batch_size=1)

    assert captured["model_dim"] == 5
    fixed_features = captured["fixed_features_list"]
    assert isinstance(fixed_features, list)
    assert len(fixed_features) == 2
    assert suggestions.loc[0, "source"] == "qlog_nei"
    assert suggestions.loc[0, "solvent"] in {"MeCN", "EtOH"}
    assert int(suggestions.loc[0, "repeats"]) in {1, 2, 3}
    assert float(suggestions.loc[0, "dose"]) in {0.1, 0.2, 0.5}


def test_qlog_nei_replicate_new_only_uses_replicate_train_yvar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = replicate_config(initial_design_size=3, suggestion_policy="new_only")
    cfg = replace(
        base,
        bo=replace(
            base.bo,
            acquisition="qlog_nei",
            raw_samples=8,
            num_restarts=1,
            mc_samples=8,
        ),
    )
    df = replicate_observed_log(cfg)
    candidate = values_to_unit_cube(cfg, [(0.55, 600.0)])
    captured: dict[str, object] = {}

    class FakePosterior:
        def __init__(self, x_unit: torch.Tensor) -> None:
            self.mean = torch.full((x_unit.shape[0], 1), 1.25, dtype=torch.double)
            self.variance = torch.full((x_unit.shape[0], 1), 0.04, dtype=torch.double)

    class FakeModel:
        def posterior(self, x_unit: torch.Tensor) -> FakePosterior:
            return FakePosterior(x_unit)

    def fake_fit_gp_model(config: CampaignConfig, observed_df: pd.DataFrame) -> FakeModel:
        training = suggestions_module.dataframe_to_training_tensors(config, observed_df)
        captured["train_yvar_used"] = training.train_yvar is not None
        return FakeModel()

    def fake_optimizer(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        assert kwargs["x_pending"] is None
        return candidate, torch.tensor(0.25, dtype=torch.double), "qlog_nei"

    monkeypatch.setattr(suggestions_module, "fit_gp_model", fake_fit_gp_model)
    monkeypatch.setattr(suggestions_module, "optimize_qlog_nei", fake_optimizer)

    suggestions = suggest_next(cfg, df, batch_size=1)

    assert captured["train_yvar_used"] is True
    assert suggestions.loc[0, "source"] == "qlog_nei"
    assert suggestions.loc[0, "replicate_group"] == suggestions.loc[0, "row_id"]
    assert int(suggestions.loc[0, "replicate_index"]) == 0


def test_multi_fidelity_qmfkg_returns_one_valid_non_mutating_suggestion() -> None:
    cfg = multi_fidelity_config(initial_design_size=3)
    df = multi_fidelity_observed_log(cfg)
    before = df.copy(deep=True)

    suggestions = suggest_next(cfg, df, batch_size=1)

    pd.testing.assert_frame_equal(df, before)
    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "qmf_kg"
    assert suggestions["x"].astype(float).between(0.0, 1.0).all()
    assert suggestions["fidelity"].astype(float).between(0.2, 1.0).all()
    assert suggestions.loc[0, "activity"] == ""
    assert math.isfinite(float(suggestions.loc[0, "predicted_mean"]))
    assert float(suggestions.loc[0, "predicted_std"]) >= 0.0
    assert math.isfinite(float(suggestions.loc[0, "acquisition"]))
    assert list(suggestions.columns) == canonical_columns(cfg)


def test_multi_fidelity_qmfkg_rejects_model_based_batch_size_above_one() -> None:
    cfg = multi_fidelity_config(initial_design_size=3)
    df = multi_fidelity_observed_log(cfg)

    with pytest.raises(SuggestionError, match="batch_size=1"):
        suggest_next(cfg, df, batch_size=2)


def test_multi_fidelity_qmfkg_wraps_optimizer_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = multi_fidelity_config(initial_design_size=3)
    df = multi_fidelity_observed_log(cfg)

    monkeypatch.setattr(
        suggestions_module,
        "fit_multi_fidelity_gp_model",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        suggestions_module,
        "optimize_posterior_mean_at_target_fidelity",
        lambda **_kwargs: torch.tensor([0.0], dtype=torch.double),
    )

    def fail_qmfkg(*_args: object, **_kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        raise RuntimeError("optimizer exploded")

    monkeypatch.setattr(suggestions_module, "optimize_qmf_kg", fail_qmfkg)

    with pytest.raises(
        SuggestionError,
        match="Could not generate qMFKG suggestion: optimizer exploded",
    ):
        suggest_next(cfg, df, batch_size=1)


def test_suggest_next_returns_model_based_batch_suggestions() -> None:
    cfg = config(batch_size=2, initial_design_size=3)
    df = observed_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 2
    assert set(suggestions["source"]) == {"qlog_ei"}
    assert suggestions["temperature"].astype(float).between(300.0, 800.0).all()


def test_one_by_one_sobol_suggestions_do_not_repeat_after_csv_round_trip(tmp_path) -> None:
    cfg = config(batch_size=1, initial_design_size=4)
    log_path = tmp_path / "campaign.csv"
    df = empty_campaign_log(cfg)
    seen: set[tuple[float, float]] = set()

    for index in range(4):
        suggestions = suggest_next(cfg, df, batch_size=1)
        candidate = (
            float(suggestions.loc[0, "x"]),
            float(suggestions.loc[0, "temperature"]),
        )
        assert candidate not in seen
        seen.add(candidate)

        append_suggestions(log_path, suggestions, config=cfg)
        mark_observed(log_path, str(suggestions.loc[0, "row_id"]), float(index))
        df = load_campaign_log(log_path, cfg)


def test_mixed_sobol_initial_suggestions_are_valid_and_duplicate_free() -> None:
    cfg = mixed_config(batch_size=3, initial_design_size=4)
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 3
    assert set(suggestions["source"]) == {"sobol"}
    assert suggestions["x"].astype(float).between(0.0, 1.0).all()
    assert set(suggestions["repeats"].astype(int)).issubset({1, 2, 3})
    assert set(suggestions["dose"].astype(float)).issubset({0.1, 0.2, 0.5})
    assert set(suggestions["solvent"]).issubset({"MeCN", "EtOH"})
    assert len(suggestions[["x", "repeats", "dose", "solvent"]].drop_duplicates()) == 3


def test_mixed_random_initial_suggestions_are_seeded() -> None:
    cfg = mixed_config(batch_size=2, initial_design_size=4, initial_design_method="random")
    df = empty_campaign_log(cfg)

    first = suggest_next(cfg, df)
    second = suggest_next(cfg, df)

    pd.testing.assert_series_equal(first["x"], second["x"])
    pd.testing.assert_series_equal(first["repeats"], second["repeats"])
    pd.testing.assert_series_equal(first["dose"], second["dose"])
    pd.testing.assert_series_equal(first["solvent"], second["solvent"])
    assert set(first["source"]) == {"random"}


def test_constrained_mixed_initial_suggestions_are_feasible() -> None:
    cfg = constrained_mixed_config(batch_size=4, initial_design_size=4)
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert not (
        (suggestions["solvent"] == "EtOH")
        & (suggestions["dose"].astype(float) >= 0.5)
    ).any()


def test_mixed_model_based_single_suggestion() -> None:
    cfg = mixed_config(batch_size=1, initial_design_size=3)
    df = mixed_observed_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "log_ei"
    assert suggestions.loc[0, "solvent"] in {"MeCN", "EtOH"}
    assert int(suggestions.loc[0, "repeats"]) in {1, 2, 3}
    assert float(suggestions.loc[0, "dose"]) in {0.1, 0.2, 0.5}
    assert float(suggestions.loc[0, "predicted_std"]) >= 0.0


def test_mixed_model_based_batch_suggestions() -> None:
    cfg = mixed_config(batch_size=2, initial_design_size=3)
    df = mixed_observed_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 2
    assert set(suggestions["source"]) == {"qlog_ei"}
    assert set(suggestions["dose"].astype(float)).issubset({0.1, 0.2, 0.5})
    assert set(suggestions["repeats"].astype(int)).issubset({1, 2, 3})
    assert set(suggestions["solvent"]).issubset({"MeCN", "EtOH"})
    assert suggestions["solvent"].nunique() == 1


def test_constrained_mixed_model_based_suggestions_are_feasible() -> None:
    cfg = constrained_mixed_config(batch_size=2, initial_design_size=3)
    df = mixed_observed_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert not (
        (suggestions["solvent"] == "EtOH")
        & (suggestions["dose"].astype(float) >= 0.5)
    ).any()


def test_cost_aware_initial_suggestions_fill_cost_but_not_utility() -> None:
    cfg = cost_review_mixed_config(batch_size=2, initial_design_size=4)
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 2
    assert set(suggestions["source"]) == {"sobol"}
    assert set(suggestions["review_status"]) == {"pending"}
    assert suggestions["cost_estimate"].astype(float).gt(0).all()
    assert suggestions["cost_actual"].astype(str).eq("").all()
    assert suggestions["utility"].astype(str).eq("").all()


def test_contextual_cost_review_initial_suggestion_fills_context_cost_and_review() -> None:
    cfg = contextual_cost_review_config(initial_design_size=4)
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(
        cfg,
        df,
        context_values={"feedstock_acidity": 0.75},
    )

    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "sobol"
    assert suggestions.loc[0, "review_status"] == "pending"
    assert float(suggestions.loc[0, "feedstock_acidity"]) == pytest.approx(0.75)
    candidate = tuple(suggestions.loc[0, cfg.variable_names])
    assert float(suggestions.loc[0, "cost_estimate"]) == pytest.approx(
        evaluate_cost(cfg, candidate)
    )
    assert suggestions.loc[0, "cost_actual"] == ""
    assert suggestions.loc[0, "utility"] == ""


def test_cost_aware_model_suggestions_fill_cost_and_utility() -> None:
    cfg = cost_review_mixed_config(batch_size=1, initial_design_size=3)
    df = cost_review_mixed_observed_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "cost_log_ei"
    assert suggestions.loc[0, "review_status"] == "pending"
    acquisition = float(suggestions.loc[0, "acquisition"])
    cost_estimate = float(suggestions.loc[0, "cost_estimate"])
    utility = float(suggestions.loc[0, "utility"])
    assert utility == pytest.approx(acquisition - cfg.cost.weight * cost_estimate)


def test_contextual_cost_review_model_suggestion_uses_fixed_context_and_cost() -> None:
    cfg = contextual_cost_review_config(initial_design_size=4)
    df = contextual_cost_review_observed_log(cfg)
    before = df.copy(deep=True)

    suggestions = suggest_next(
        cfg,
        df,
        context_values={"feedstock_acidity": 0.5},
    )

    pd.testing.assert_frame_equal(df, before)
    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "cost_log_ei"
    assert suggestions.loc[0, "review_status"] == "pending"
    assert float(suggestions.loc[0, "feedstock_acidity"]) == pytest.approx(0.5)
    candidate = tuple(suggestions.loc[0, cfg.variable_names])
    cost_estimate = float(suggestions.loc[0, "cost_estimate"])
    acquisition = float(suggestions.loc[0, "acquisition"])
    utility = float(suggestions.loc[0, "utility"])
    assert cost_estimate == pytest.approx(evaluate_cost(cfg, candidate))
    assert utility == pytest.approx(acquisition - cfg.cost.weight * cost_estimate)


def test_contextual_cost_aware_model_path_requires_resolved_context_values() -> None:
    cfg = contextual_cost_review_config(initial_design_size=4)
    df = contextual_cost_review_observed_log(cfg)

    with pytest.raises(
        SuggestionError,
        match="Contextual cost-aware suggestions require resolved context values",
    ):
        suggestions_module._suggest_cost_aware_model_based(
            config=cfg,
            df=df,
            observed_df=df,
            batch_size=1,
        )


def test_cost_aware_candidate_pool_falls_back_to_sobol_when_optimizer_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = cost_review_mixed_config(batch_size=1, initial_design_size=3)
    df = cost_review_mixed_observed_log(cfg)

    def failing_optimizer(*args, **kwargs):
        raise RuntimeError("optimizer failed")

    monkeypatch.setattr(suggestions_module, "optimize_log_ei", failing_optimizer)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "cost_log_ei"
    assert suggestions.loc[0, "review_status"] == "pending"
    assert float(suggestions.loc[0, "cost_estimate"]) > 0
    assert suggestions.loc[0, "solvent"] in {"MeCN", "EtOH"}


def test_contextual_cost_duplicate_detection_uses_full_context_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = contextual_cost_review_config(initial_design_size=4)
    df = contextual_cost_review_observed_log(cfg)
    same_decision_different_context = (0.20, 70, "MeCN", 0.75)

    def candidate_pool(*args, **kwargs):
        return [
            (0.20, 70, "MeCN", 0.25),
            same_decision_different_context,
        ]

    def score_candidate(*, config, model, acquisition, candidate, cost_estimate):
        return {
            "candidate": candidate,
            "cost_estimate": cost_estimate,
            "acquisition": 2.0,
            "utility": 2.0 - config.cost.weight * cost_estimate,
            "predicted_mean": 0.8,
            "predicted_std": 0.1,
        }

    monkeypatch.setattr(suggestions_module, "_cost_aware_candidate_pool", candidate_pool)
    monkeypatch.setattr(suggestions_module, "_score_cost_aware_candidate", score_candidate)

    suggestions = suggest_next(
        cfg,
        df,
        context_values={"feedstock_acidity": 0.75},
    )

    assert tuple(suggestions.loc[0, cfg.variable_names]) == same_decision_different_context


@pytest.mark.parametrize(
    ("review_status", "blocks"),
    [
        ("pending", True),
        ("accepted", True),
        ("rejected", False),
        ("deferred", False),
    ],
)
def test_review_status_controls_suggestion_blocking(
    review_status: str,
    blocks: bool,
) -> None:
    cfg = review_mixed_config(batch_size=1, initial_design_size=2)
    df = pd.DataFrame(
        [
            {
                "row_id": "reviewed_0",
                "iteration": 0,
                "status": "suggested",
                "source": "sobol",
                "review_status": review_status,
                "review_note": "",
                "x": 0.1,
                "repeats": 1,
                "dose": 0.1,
                "solvent": "MeCN",
                "score": "",
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )

    if blocks:
        with pytest.raises(SuggestionError, match="unresolved status='suggested'"):
            suggest_next(cfg, df)
    else:
        suggestions = suggest_next(cfg, df)
        assert len(suggestions) == 1


def test_rejected_suggestions_do_not_block_but_remain_duplicate_protected() -> None:
    cfg = CampaignConfig(
        campaign_name="review_categories",
        objective=ObjectiveConfig(name="score", direction="maximize"),
        variables=(VariableConfig("solvent", "categorical", values=("MeCN", "EtOH")),),
        bo=BOConfig(batch_size=1, initial_design_size=2, random_seed=3),
        review=ReviewConfig(enabled=True),
    )
    df = pd.DataFrame(
        [
            {
                "row_id": "rejected_0",
                "iteration": 0,
                "status": "suggested",
                "source": "sobol",
                "review_status": "rejected",
                "review_note": "not practical",
                "solvent": "MeCN",
                "score": "",
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )

    suggestions = suggest_next(cfg, df)

    assert suggestions.loc[0, "solvent"] == "EtOH"


def test_cost_budget_exhaustion_raises_clear_error() -> None:
    cfg = cost_review_mixed_config(batch_size=1, initial_design_size=3, budget=0.1)
    df = cost_review_mixed_observed_log(cfg)

    with pytest.raises(SuggestionError, match="remaining budget may be too small"):
        suggest_next(cfg, df)


def test_initial_cost_budget_exhaustion_raises_clear_error() -> None:
    cfg = cost_review_mixed_config(batch_size=1, initial_design_size=2, budget=0.0)
    df = empty_campaign_log(cfg)

    with pytest.raises(SuggestionError, match="budget-feasible initial suggestions"):
        suggest_next(cfg, df)


def test_cost_penalty_can_choose_cheaper_lower_acquisition_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = cost_review_mixed_config(
        batch_size=1,
        initial_design_size=3,
        budget=50.0,
        weight=1.0,
    )
    df = cost_review_mixed_observed_log(cfg)
    cheap = (0.2, 1, 0.1, "MeCN")
    costly = (0.9, 3, 0.5, "EtOH")

    def candidate_pool(*args, **kwargs):
        return [costly, cheap]

    def score_candidate(*, config, model, acquisition, candidate, cost_estimate):
        acquisition_value = 3.0 if candidate == costly else 2.0
        return {
            "candidate": candidate,
            "cost_estimate": cost_estimate,
            "acquisition": acquisition_value,
            "utility": acquisition_value - config.cost.weight * cost_estimate,
            "predicted_mean": 1.0,
            "predicted_std": 0.1,
        }

    monkeypatch.setattr(suggestions_module, "_cost_aware_candidate_pool", candidate_pool)
    monkeypatch.setattr(suggestions_module, "_score_cost_aware_candidate", score_candidate)

    suggestions = suggest_next(cfg, df)

    assert tuple(suggestions.loc[0, cfg.variable_names]) == cheap
    assert float(suggestions.loc[0, "acquisition"]) < 3.0


def test_suggestion_quality_summary_reports_constraints_duplicates_and_distances() -> None:
    cfg = constrained_mixed_config(
        batch_size=2,
        initial_design_size=3,
        min_normalized_distance=0.05,
    )
    df = mixed_observed_log(cfg)
    suggestions = pd.DataFrame(
        [
            {
                "row_id": "suggested_duplicate",
                "iteration": 4,
                "status": "suggested",
                "source": "log_ei",
                "x": 0.1,
                "repeats": 1,
                "dose": 0.1,
                "solvent": "MeCN",
                "score": "",
                "predicted_mean": 1.1,
                "predicted_std": 0.1,
                "acquisition": 0.01,
            },
            {
                "row_id": "suggested_infeasible",
                "iteration": 4,
                "status": "suggested",
                "source": "log_ei",
                "x": 0.2,
                "repeats": 2,
                "dose": 0.5,
                "solvent": "EtOH",
                "score": "",
                "predicted_mean": 1.2,
                "predicted_std": 0.1,
                "acquisition": 0.02,
            },
        ],
        columns=canonical_columns(cfg),
    )

    summary = suggestion_quality_summary(cfg, df, suggestions)

    assert list(summary.columns) == [
        "row_id",
        "is_feasible",
        "violated_constraints",
        "is_exact_duplicate",
        "duplicate_allowed_by_replicates",
        "nearest_existing_distance",
        "nearest_batch_distance",
        "passes_distance_threshold",
    ]
    assert bool(summary.loc[0, "is_exact_duplicate"])
    assert summary.loc[1, "violated_constraints"] == "no_etoh_high_dose"
    assert not bool(summary.loc[1, "is_feasible"])
    assert summary["nearest_existing_distance"].notna().all()


def test_categorical_combination_threshold_is_enforced() -> None:
    cfg = CampaignConfig(
        campaign_name="many_categories",
        objective=ObjectiveConfig(name="score", direction="maximize"),
        variables=tuple(
            VariableConfig(f"cat_{index}", "categorical", values=("a", "b"))
            for index in range(7)
        ),
        bo=BOConfig(batch_size=1, initial_design_size=1),
    )
    row = {
        "row_id": "obs_0",
        "iteration": 0,
        "status": "observed",
        "source": "manual",
        "score": 1.0,
        "predicted_mean": "",
        "predicted_std": "",
        "acquisition": "",
    }
    for variable in cfg.variables:
        row[variable.name] = "a"
    df = pd.DataFrame([row], columns=canonical_columns(cfg))

    with pytest.raises(SuggestionError, match="at most 64 categorical combinations"):
        suggest_next(cfg, df)


def test_duplicate_decoded_candidates_retry_then_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = mixed_config(batch_size=1, initial_design_size=3)
    df = mixed_observed_log(cfg)
    duplicate_x = values_to_unit_cube(cfg, [(0.1, 1, 0.1, "MeCN")])
    calls = 0

    def duplicate_optimizer(*args, **kwargs):
        nonlocal calls
        calls += 1
        return duplicate_x, torch.tensor(0.0, dtype=torch.double), "log_ei"

    monkeypatch.setattr(suggestions_module, "optimize_log_ei", duplicate_optimizer)

    with pytest.raises(SuggestionError, match=f"{MAX_DECODE_RETRIES} retries"):
        suggest_next(cfg, df)

    assert calls == MAX_DECODE_RETRIES


def test_near_duplicate_threshold_failure_has_clear_message() -> None:
    cfg = CampaignConfig(
        campaign_name="too_restrictive",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=BOConfig(
            batch_size=1,
            initial_design_size=1,
            random_seed=3,
            raw_samples=16,
            num_restarts=2,
            mc_samples=16,
            min_normalized_distance=1.1,
        ),
    )
    df = pd.DataFrame(
        [
            {
                "row_id": "obs_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "x": 0.5,
                "activity": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )

    with pytest.raises(SuggestionError, match="constraints may be too restrictive"):
        suggest_next(cfg, df)


def test_mixed_append_round_trip_validates(tmp_path) -> None:
    cfg = mixed_config(batch_size=1, initial_design_size=3)
    log_path = tmp_path / "mixed.csv"
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(cfg, df, batch_size=1)
    append_suggestions(log_path, suggestions)
    mark_observed(log_path, str(suggestions.loc[0, "row_id"]), 1.2)
    reloaded = load_campaign_log(log_path, cfg)

    assert len(reloaded) == 1
    assert reloaded.loc[0, "status"] == "observed"
    assert reloaded.loc[0, "solvent"] in {"MeCN", "EtOH"}


def test_finite_mixed_initial_space_exhaustion_raises() -> None:
    cfg = CampaignConfig(
        campaign_name="tiny",
        objective=ObjectiveConfig(name="score", direction="maximize"),
        variables=(VariableConfig("solvent", "categorical", values=("MeCN",)),),
        bo=BOConfig(batch_size=2, initial_design_size=2, random_seed=3),
    )
    df = empty_campaign_log(cfg)

    with pytest.raises(SuggestionError, match="finite design space is exhausted"):
        suggest_next(cfg, df)
