"""Static consumers only: these functions are checked, never executed."""

from pathlib import Path
from typing import Any, assert_type

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

import bo_forge as bo
from bo_forge.config import CampaignConfig as ConfigViaModule
from bo_forge.logs import append_suggestions as append_via_module
from bo_forge.models import model_summary as summary_via_module
from bo_forge.predictive import PredictiveEvaluationResult as ResultViaModule
from bo_forge.provenance import adopt_provenance as adopt_via_module
from bo_forge.session import CampaignSession as SessionViaModule
from bo_forge.suggestions import suggest_next as suggest_via_module
from bo_forge.validation import validate_campaign_data as validate_via_module


def configurations(path: Path) -> bo.CampaignConfig:
    variable = assert_type(bo.VariableConfig("x", "continuous", 0, 1), bo.VariableConfig)
    objective = assert_type(bo.ObjectiveConfig("score", "maximize"), bo.ObjectiveConfig)
    bo_config = assert_type(bo.BOConfig(batch_size=2), bo.BOConfig)
    constraint = assert_type(bo.ConstraintConfig("bound", "x < 1"), bo.ConstraintConfig)
    cost = assert_type(bo.CostConfig("1 + x"), bo.CostConfig)
    fidelity = assert_type(bo.FidelityConfig("s", 1), bo.FidelityConfig)
    context = assert_type(bo.ContextConfig(("temperature",)), bo.ContextConfig)
    model = assert_type(bo.ModelConfig("smooth"), bo.ModelConfig)
    review = assert_type(bo.ReviewConfig(True), bo.ReviewConfig)
    replicates = assert_type(bo.ReplicateConfig(True), bo.ReplicateConfig)
    stage = assert_type(bo.StageConfig("first", ("x",)), bo.StageConfig)
    config = assert_type(bo.CampaignConfig("example", objective, (variable,), bo_config),
                         bo.CampaignConfig)
    assert_type(ConfigViaModule.from_yaml(path), bo.CampaignConfig)
    assert_type(config.objective, bo.ObjectiveConfig | None)
    assert_type(config.cost, bo.CostConfig | None)
    assert_type(config.fidelity, bo.FidelityConfig | None)
    assert_type(config.context, bo.ContextConfig | None)
    assert_type(config.variables, tuple[bo.VariableConfig, ...])
    assert_type(config.objectives, tuple[bo.ObjectiveConfig, ...])
    assert_type(config.constraints, tuple[bo.ConstraintConfig, ...])
    assert_type(config.variable_names, list[str])
    assert_type(config.objective_names, list[str])
    assert_type(config.stage_names, list[str])
    assert_type(config.context_variable_names, list[str])
    assert_type(config.decision_variable_names, list[str])
    assert_type(config.direction_sign, float)
    assert_type(config.is_multi_objective, bool)
    assert_type(config.is_contextual_campaign, bool)
    assert_type(config.is_structured_campaign, bool)
    assert_type(config.active_variable_names_for_stage("first"), list[str])
    assert_type(variable.lower, float | None)
    assert_type(variable.upper, float | None)
    assert_type(variable.values, tuple[str | float, ...])
    assert_type(objective.reference_point, float | None)
    assert_type(constraint.expression, str)
    assert_type(cost.budget, float | None)
    assert_type(fidelity.levels, tuple[float, ...] | None)
    assert_type(fidelity.optimizer_timeout_seconds, float | None)
    assert_type(context.default_values, dict[str, object])
    assert_type(model.profile, str)
    assert_type(review.enabled, bool)
    assert_type(replicates.noise_floor, float)
    assert_type(stage.variables, tuple[str, ...])
    return config


def core_exports(config: bo.CampaignConfig, df: pd.DataFrame, path: Path) -> None:
    assert_type(bo.__version__, str)
    assert_type(bo.active_variables_for_stage(config, "first"), list[str])
    assert_type(bo.configured_stage_names(config), list[str])
    assert_type(bo.is_structured_campaign(config), bool)
    assert_type(bo.evaluate_cost(config, {"x": 0.5}), float)
    assert_type(bo.evaluate_cost(config, [0.5]), float)
    assert_type(bo.evaluate_cost(config, pd.Series({"x": 0.5})), float)
    for _, row in df.iterrows():
        assert_type(bo.evaluate_cost(config, row), float)
    assert_type(bo.hypervolume(config, df), float)
    assert_type(bo.hypervolume_progress(config, df), pd.DataFrame)
    assert_type(bo.aggregate_observed_replicates(config, df), pd.DataFrame)
    assert_type(bo.best_replicate_group(config, df), pd.DataFrame)
    assert_type(bo.context_summary(config, df), pd.DataFrame)
    assert_type(bo.fidelity_coverage(config, df), pd.DataFrame)
    assert_type(bo.fidelity_summary(config, df), pd.DataFrame)
    assert_type(bo.get_observed_data(config, df), pd.DataFrame)
    assert_type(bo.load_campaign_log(path, config), pd.DataFrame)
    assert_type(bo.model_summary(config, df), pd.DataFrame)
    assert_type(bo.model_profile_comparison(config, df, ("rough", "default")), pd.DataFrame)
    assert_type(bo.pareto_front(config, df), pd.DataFrame)
    assert_type(bo.pareto_summary(config, df), pd.DataFrame)
    assert_type(bo.qlog_nei_summary(config, df), pd.DataFrame)
    assert_type(bo.replicate_summary(config, df), pd.DataFrame)
    assert_type(bo.stage_summary(config, df), pd.DataFrame)
    suggestions = assert_type(bo.suggest_next(config, df, batch_size=2, stage="first",
                                             context_values={"temperature": 25}), pd.DataFrame)
    assert_type(bo.suggestion_quality_summary(config, df, suggestions), pd.DataFrame)
    assert_type(bo.append_suggestions(path, suggestions, config), None)
    assert_type(bo.mark_observed(path, "row", 2.0, config=config), None)
    assert_type(
        bo.mark_observed(path, "row", objective_values={"a": 1, "b": 2}, config=config), None,
    )
    assert_type(bo.review_suggestion(path, "row", "accept", config=config), None)
    assert_type(bo.validate_campaign_data(config, df), None)
    assert_type(append_via_module(path, df, config), None)
    assert_type(suggest_via_module(config, df), pd.DataFrame)
    assert_type(summary_via_module(config, df), pd.DataFrame)
    assert_type(validate_via_module(config, df), None)


def sessions(config: bo.CampaignConfig, df: pd.DataFrame, path: Path) -> None:
    assert_type(bo.CampaignSession(path, path, config, df), bo.CampaignSession)
    assert_type(bo.CampaignSession.initialize(path, path), bo.CampaignSession)
    session = assert_type(SessionViaModule.from_files(path, path, provenance_policy="required"),
                          bo.CampaignSession)
    assert_type(session.config, bo.CampaignConfig)
    assert_type(session.df, pd.DataFrame)
    assert_type(session.log_path, Path)
    assert_type(session.config_path, Path)
    assert_type(session.log_fingerprint, str | None)
    assert_type(session.config_fingerprint, str | None)
    assert_type(session.is_provenance_managed, bool)
    assert_type(session.reload(), pd.DataFrame)
    assert_type(session.validate(), None)
    assert_type(session.summary(), pd.DataFrame)
    assert_type(session.observed_data(), pd.DataFrame)
    assert_type(session.pending_suggestions(), pd.DataFrame)
    assert_type(session.review_queue(), pd.DataFrame)
    assert_type(session.cost_summary(), pd.DataFrame)
    assert_type(session.campaign_status(), str)
    assert_type(session.next_action(), pd.DataFrame)
    assert_type(session.report(), dict[str, pd.DataFrame])
    assert_type(session.provenance_summary(), pd.DataFrame)
    assert_type(session.export_report(path), Path)
    assert_type(session.best_observation(), pd.DataFrame)
    assert_type(session.pareto_front(), pd.DataFrame)
    assert_type(session.pareto_summary(), pd.DataFrame)
    assert_type(session.stage_summary(), pd.DataFrame)
    assert_type(session.fidelity_summary(), pd.DataFrame)
    assert_type(session.fidelity_coverage(), pd.DataFrame)
    assert_type(session.context_summary(), pd.DataFrame)
    assert_type(session.qlog_nei_summary(), pd.DataFrame)
    assert_type(session.model_summary(), pd.DataFrame)
    assert_type(session.model_profile_comparison(profiles=["smooth"]), pd.DataFrame)
    assert_type(session.model_predictive_evaluation(("default",), folds=3, seed=0),
                bo.PredictiveEvaluationResult)
    assert_type(session.replicate_summary(), pd.DataFrame)
    assert_type(session.best_replicate_group(), pd.DataFrame)
    assert_type(
        session.suggest_next(batch_size=1, context_values={"temperature": 20}), pd.DataFrame,
    )
    assert_type(session.suggestion_quality(df), pd.DataFrame)
    assert_type(session.append_suggestions(df, expected_log_fingerprint="snapshot"), pd.DataFrame)
    assert_type(session.mark_observed("row", 2.0, actual_cost=1), pd.DataFrame)
    assert_type(session.review_suggestion("row", "accept"), pd.DataFrame)
    assert_type(session.plot_progress(save_path=path), tuple[Figure, Axes])
    assert_type(session.plot_diagnostics(), tuple[Figure, Axes | NDArray[np.object_]])
    assert_type(session.plot_cost_progress(), tuple[Figure, Axes])
    assert_type(session.plot_replicates(), tuple[Figure, Axes | NDArray[np.object_]])
    assert_type(session.plot_pareto(), tuple[Figure, Axes | NDArray[np.object_]])
    assert_type(session.plot_pareto_parallel(), tuple[Figure, Axes])
    assert_type(session.plot_hypervolume(), tuple[Figure, Axes])
    assert_type(session.plot_stage_diagnostics(), tuple[Figure, NDArray[np.object_]])
    assert_type(session.plot_fidelity_diagnostics(), tuple[Figure, NDArray[np.object_]])
    assert_type(session.plot_fidelity_progress(), tuple[Figure, NDArray[np.object_]])
    assert_type(session.plot_context_diagnostics(), tuple[Figure, NDArray[np.object_]])
    assert_type(session.plot_model_diagnostics(), tuple[Figure, NDArray[np.object_]])
    assert_type(session.plot_model_comparison(), tuple[Figure, NDArray[np.object_]])
    assert_type(session.plot_qlog_nei_diagnostics(), tuple[Figure, NDArray[np.object_]])


def provenance(config: Path, log: Path, destination: Path) -> None:
    assert_type(bo.provenance_summary(config, log), pd.DataFrame)
    assert_type(
        bo.recover_provenance(config, log, expected_log_fingerprint="snapshot"), pd.DataFrame,
    )
    assert_type(bo.adopt_provenance(config, log), dict[str, Any])
    assert_type(adopt_via_module(config, log, apply=True, reason="adopt",
                                expected_identities={"log_sha256": "snapshot"}), dict[str, Any])
    assert_type(bo.migrate_provenance(config, log), dict[str, Any])
    assert_type(bo.accept_provenance_config(config, log), dict[str, Any])
    assert_type(bo.fork_campaign(config, log, destination,
                                 config_changes={"campaign_name": "child"}), dict[str, Any])


def results(config: bo.CampaignConfig, df: pd.DataFrame, path: Path) -> None:
    fit = assert_type(bo.FitMetadata((("fit_status", "success"),)), bo.FitMetadata)
    assert_type(fit.fields, tuple[tuple[str, str | int | bool | None], ...])
    assert_type(fit.as_dict(), dict[str, str | int | bool | None])
    assert_type(bo.model_summary(config, df, metadata=fit), pd.DataFrame)
    result = assert_type(bo.model_predictive_evaluation(config, df, ["smooth"], folds=2),
                         bo.PredictiveEvaluationResult)
    assert_type(ResultViaModule(df, df, df, {"seed": 0}), bo.PredictiveEvaluationResult)
    assert_type(result.summary, pd.DataFrame)
    assert_type(result.predictions, pd.DataFrame)
    assert_type(result.fold_outcomes, pd.DataFrame)
    assert_type(result.metadata, dict[str, Any])
    assert_type(result.export(path), Path)
    assert_type(result.plot_predictions(save_path=path), tuple[Figure, NDArray[np.object_]])
    assert_type(result.plot_residuals(), tuple[Figure, NDArray[np.object_]])


def errors() -> None:
    assert_type(bo.BOForgeError("error"), bo.BOForgeError)
    assert_type(bo.ConfigError("error"), bo.ConfigError)
    assert_type(bo.LogValidationError("error"), bo.LogValidationError)
    assert_type(bo.LogWriteError("error"), bo.LogWriteError)
    assert_type(bo.LogConflictError("error"), bo.LogConflictError)
    assert_type(bo.LogBusyError("error"), bo.LogBusyError)
    assert_type(bo.SuggestionError("error"), bo.SuggestionError)
    error = assert_type(bo.ProvenanceError("error"), bo.ProvenanceError)
    assert_type(error.reason_code, str)
    assert_type(error.recovery_action, str | None)
    recovery = assert_type(bo.ProvenanceRecoveryRequired("error", reason_code="pending",
                                                        recovery_action="recover"),
                           bo.ProvenanceRecoveryRequired)
    assert_type(recovery.reason_code, str)
    assert_type(recovery.recovery_action, str)
