import math
import warnings
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


def patch_multi_fidelity_test_model(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePosterior:
        mean = torch.tensor([[1.2]], dtype=torch.double)
        variance = torch.tensor([[0.04]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x_unit: torch.Tensor) -> FakePosterior:
            return FakePosterior()

    monkeypatch.setattr(
        suggestions_module,
        "fit_multi_fidelity_gp_model",
        lambda *_args, **_kwargs: FakeModel(),
    )


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

__all__ = [
    'BOConfig',
    'CampaignConfig',
    'ConstraintConfig',
    'ContextConfig',
    'CostConfig',
    'FidelityConfig',
    'MAX_DECODE_RETRIES',
    'ModelConfig',
    'ObjectiveConfig',
    'ReplicateConfig',
    'ReviewConfig',
    'StageConfig',
    'SuggestionError',
    'VariableConfig',
    'append_suggestions',
    'canonical_columns',
    'config',
    'constrained_mixed_config',
    'contextual_cost_review_config',
    'contextual_cost_review_observed_log',
    'cost_review_mixed_config',
    'cost_review_mixed_observed_log',
    'empty_campaign_log',
    'evaluate_cost',
    'load_campaign_log',
    'mark_observed',
    'math',
    'mixed_config',
    'mixed_observed_log',
    'multi_fidelity_config',
    'multi_fidelity_observed_log',
    'observed_log',
    'patch_multi_fidelity_test_model',
    'pd',
    'pytest',
    'qlog_nehvi_config',
    'qlog_nehvi_log',
    'qlog_nehvi_pending_row',
    'qlog_nei_config',
    'qlog_nei_log',
    'reference_point_to_model_space',
    'replace',
    'replicate_config',
    'replicate_observed_log',
    'review_mixed_config',
    'structured_config',
    'suggest_next',
    'suggestion_quality_summary',
    'suggestions_module',
    'torch',
    'values_to_unit_cube',
    'warnings',
]
