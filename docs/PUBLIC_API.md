# 📦 BO Forge Public API

This page lists the stable imports supported from the top-level `bo_forge` package in v2.5.3.

Top-level exports are resolved lazily. Names, signatures, `__all__`,
star imports, and `dir(bo_forge)` remain compatible; importing the package alone
does not load optimizer or plotting dependencies.

Implementation modules such as `bo_forge.transforms`, `bo_forge.models`, and `bo_forge.diagnostics` remain importable for development, but their private helpers are not part of the stable public surface.

For supported and intentionally deferred workflow combinations, see
[CAPABILITY_MATRIX.md](CAPABILITY_MATRIX.md).

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
- `LogBusyError`
- `LogConflictError`
- `LogValidationError`
- `LogWriteError`
- `ModelConfig`
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
- `context_summary`
- `evaluate_cost`
- `fidelity_coverage`
- `fidelity_summary`
- `get_observed_data`
- `hypervolume`
- `hypervolume_progress`
- `is_structured_campaign`
- `load_campaign_log`
- `mark_observed`
- `model_summary`
- `model_profile_comparison`
- `pareto_front`
- `pareto_summary`
- `qlog_nei_summary`
- `review_suggestion`
- `replicate_summary`
- `stage_summary`
- `suggest_next`
- `suggestion_quality_summary`
- `validate_campaign_data`

`best_replicate_group` is only defined for single-objective replicate campaigns. For multi-objective replicate campaigns, use `replicate_summary` for group-level statistics and `pareto_front` for group-mean Pareto inspection.

Replicate-enabled model fitting keeps raw CSV rows as the source of truth, but trains on one group-mean row per `replicate_group`. When empirical replicate variance is available, BO Forge passes group-mean observation variance to BoTorch as `train_Yvar`; otherwise it keeps learned-noise GP behavior.

For append safety, prefer `CampaignSession.append_suggestions()` or `append_suggestions(log_path, suggestions, config=config)`. The config-aware path validates the combined CSV log before writing. Calling `append_suggestions(log_path, suggestions)` without a config remains supported for simple non-replicate, non-structured logs, but replicate, structured, qMFKG, and qLogNEHVI generated rows require config-aware append validation. Structured logs also require config-aware `mark_observed()` and `review_suggestion()` transitions; use the `CampaignSession` methods or pass `config=config` to the low-level helpers.

BO Forge serializes append, review, and observation mutations with one
same-machine file lock per canonical resolved log path. `CampaignSession`
captures a log fingerprint at load/reload and passes it to later mutations;
stale sessions raise `LogConflictError`, while lock acquisition timeouts raise
`LogBusyError`. Low-level mutation helpers accept optional keyword-only
`expected_log_fingerprint`; omitting it preserves latest-state serialized
behavior. The lock directory is process-stable even when local processes use
different temporary-directory environment settings. App-service dry-runs bind
their staged payload to the exact config/log fingerprints used before
optimization and fail if either file changes during generation. Multi-host
shared-filesystem coordination is not supported.

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

Multi-fidelity campaigns expose `FidelityConfig`, `fidelity_summary`, and
`fidelity_coverage` through
the top-level package for config construction and read-only inspection.
`FidelityConfig.levels` optionally constrains the continuous fidelity variable
to ordered numeric levels. qMFKG batches from one through four are supported.
`FidelityConfig.optimizer_maxiter` defaults to `200`, while
`optimizer_timeout_seconds` defaults to `None`; omitting both preserves the
v2.4.0 numerical path. The timeout is one acquisition deadline after model
fitting. Candidate batches returned after the deadline are rejected, but
BoTorch initial-condition generation and in-flight calls cannot be cancelled
immediately, so the method can return later than the configured limit. The
timeout is not a candidate-quality guarantee.
`fidelity_summary()` appends the fidelity mode, configured levels and level
counts after its existing fields. Existing continuous summaries keep those
new level-specific values blank.
`fidelity_coverage()` returns deterministic per-fidelity observed statistics,
active suggestion counts, affine modeled evaluation cost, and direction-aware
best rows without fitting a model or mutating the input data. Continuous values
remain exact sorted coverage keys; discrete rows map uniquely to configured
levels. Use
`CampaignSession.plot_fidelity_progress()` for fidelity-by-iteration and
target-fidelity best-so-far progress.
BoTorch-facing helper functions in `bo_forge.multifidelity` remain
implementation details rather than stable public API.

Contextual campaigns expose `ContextConfig` and `context_summary` through the
top-level package for config construction and read-only inspection.
`CampaignConfig.context_variable_names` and
`CampaignConfig.decision_variable_names` identify fixed-at-suggestion-time
context variables and optimized decision variables. Contextual support is
single-objective LogEI/qLogEI only; `bo.acquisition: log_ei` may combine
with `review.enabled: true`, deterministic `cost:`, replicates, or all three. Use
`suggest_next(config, df, context_values={...})` or
`CampaignSession.suggest_next(context_values={...})` when context defaults are
not fully declared in YAML. Use `context_summary(config, df)` or
`CampaignSession.context_summary()` to inspect observed and pending rows by
context combination. For contextual cost campaigns, use the existing
`cost_summary`, `mark_observed(..., actual_cost=...)`, and cost-progress
plotting APIs.
`context_summary()` is row-level by context combination, so observed replicate
rows are counted individually. `replicate_summary()` remains group-level, with
one row per observed replicate group and context variables retained as design
columns.

Model profiles expose `ModelConfig`, `model_summary`, and
`model_profile_comparison` through the top-level package for config construction
and read-only inspection. Supported profiles are `default`, `smooth`, `rough`,
and `robust`; non-default profiles require single-objective configs with
`bo.acquisition: log_ei` or `qlog_nei`.
Use `model_summary(config, df)` or `CampaignSession.model_summary()` to inspect
the configured profile, model class, covariance profile, fitting-row count, and
train-Y variance use. Use `model_profile_comparison(config, df)` or
`CampaignSession.model_profile_comparison()` to compare supported profiles on
the current observed fitting rows without changing the configured profile or
CSV log. Comparison rows include `fit_status` and `fit_message` so failed or
insufficient profile fits stay visible in tables and plots.
`last_fit_*` fields are process-local and report `not_recorded` unless a model
fit has happened in the same Python process for matching current fitting inputs.

qLogNEI pending-state diagnostics expose `qlog_nei_summary` through the
top-level package for read-only inspection. Use `qlog_nei_summary(config, df)`
or `CampaignSession.qlog_nei_summary()` on configs with
`bo.acquisition: qlog_nei` to inspect observed baseline rows, active
`X_pending` rows, review-pending blockers, initial-design readiness,
replicate-derived `train_Yvar` availability, and the configured model profile.
qLogNEHVI uses the existing multi-objective public helpers rather than adding a
new helper. Use `pareto_front`, `pareto_summary`, `hypervolume`,
`hypervolume_progress`, and `CampaignSession.suggest_next()` on supported
configs with `bo.acquisition: qlog_nehvi`.

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

The following are intentionally not guaranteed as stable public APIs:

- private functions beginning with `_`;
- Streamlit app helper internals;
- matplotlib styling internals;
- exact text formatting of reports beyond documented sections;
- implementation details of latent transforms and acquisition optimisation.
