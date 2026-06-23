# 📦 BO Forge Public API

This page lists the stable imports supported from the top-level `bo_forge` package in v1.5.0.

Implementation modules such as `bo_forge.transforms`, `bo_forge.models`, and `bo_forge.diagnostics` remain importable for development, but their private helpers are not part of the stable public surface.

## ✅ Public Package Exports

These names are supported imports from `bo_forge`:

- `BOConfig`
- `BOForgeError`
- `CampaignConfig`
- `CampaignSession`
- `ConfigError`
- `ConstraintConfig`
- `ContextConfig`
- `CostConfig`
- `FidelityConfig`
- `LogValidationError`
- `LogWriteError`
- `ObjectiveConfig`
- `ReplicateConfig`
- `ReviewConfig`
- `StageConfig`
- `SuggestionError`
- `VariableConfig`
- `__version__`
- `active_variables_for_stage`
- `append_suggestions`
- `aggregate_observed_replicates`
- `best_replicate_group`
- `configured_stage_names`
- `evaluate_cost`
- `fidelity_summary`
- `get_observed_data`
- `hypervolume`
- `hypervolume_progress`
- `is_structured_campaign`
- `load_campaign_log`
- `mark_observed`
- `pareto_front`
- `pareto_summary`
- `review_suggestion`
- `replicate_summary`
- `stage_summary`
- `suggest_next`
- `suggestion_quality_summary`
- `validate_campaign_data`

`best_replicate_group` is only defined for single-objective replicate campaigns. For multi-objective replicate campaigns, use `replicate_summary` for group-level statistics and `pareto_front` for group-mean Pareto inspection.

Replicate-enabled model fitting keeps raw CSV rows as the source of truth, but trains on one group-mean row per `replicate_group`. When empirical replicate variance is available, BO Forge passes group-mean observation variance to BoTorch as `train_Yvar`; otherwise it keeps learned-noise GP behavior.

For append safety, prefer `CampaignSession.append_suggestions()` or `append_suggestions(log_path, suggestions, config=config)`. The config-aware path validates the combined CSV log before writing. Calling `append_suggestions(log_path, suggestions)` without a config remains supported for non-replicate, non-structured logs, but replicate and structured logs require config-aware append validation. Structured logs also require config-aware `mark_observed()` and `review_suggestion()` transitions; use the `CampaignSession` methods or pass `config=config` to the low-level helpers.

Structured campaigns expose stage metadata through `StageConfig`,
`is_structured_campaign`, `configured_stage_names`, and
`active_variables_for_stage`. v1.3.1 supports explicit stage-aware suggestions
with `suggest_next(config, df, stage="...")` or
`CampaignSession.suggest_next(stage="...")`. v1.3.2 adds read-only
`stage_summary(config, df)`, `CampaignSession.stage_summary()`, and
`CampaignSession.plot_stage_diagnostics()`. For replicate-enabled structured
campaigns, stage best values use replicate group means; the `best_row_id`
field contains the best `replicate_group`. Cost-aware structured campaigns and
automatic stage transitions remain deferred.

Multi-fidelity campaigns expose `FidelityConfig` and `fidelity_summary` through
the top-level package for config construction and read-only inspection.
v1.4.x multi-fidelity support remains single-objective, continuous-variable
qMFKG only; BoTorch-facing helper functions in `bo_forge.multifidelity` are
implementation details rather than stable public API.

Contextual campaigns expose `ContextConfig` through the top-level package for
config construction and inspection. `CampaignConfig.context_variable_names` and
`CampaignConfig.decision_variable_names` identify fixed-at-suggestion-time
context variables and optimized decision variables. v1.5.0 contextual support
is single-objective LogEI/qLogEI only. Use
`suggest_next(config, df, context_values={...})` or
`CampaignSession.suggest_next(context_values={...})` when context defaults are
not fully declared in YAML.

`hypervolume` returns the current multi-objective hypervolume for the observed state, using replicate group means when replicates are enabled. `hypervolume_progress` returns cumulative best-so-far hypervolume progress with `observation`, `row_id`, `iteration`, and `hypervolume` columns.

## 🧪 Example

```python
from bo_forge import CampaignConfig, CampaignSession, suggest_next

config = CampaignConfig.from_yaml("configs/01_simple_2d_maximise_logei.yaml")
campaign = CampaignSession.from_files(
    config_path="configs/01_simple_2d_maximise_logei.yaml",
    log_path="examples/01_simple_2d_maximise_logei_campaign_log.csv",
)
suggestions = suggest_next(config, campaign.df)
```

## 🚧 Not Public API

The following are intentionally not guaranteed as stable v1.0 public APIs:

- private functions beginning with `_`;
- Streamlit app helper internals;
- matplotlib styling internals;
- exact text formatting of reports beyond documented sections;
- implementation details of latent transforms and acquisition optimisation.
