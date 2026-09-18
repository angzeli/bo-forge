# 📦 BO Forge Public API

This page lists the stable imports supported from the top-level `bo_forge` package in v3.3.1.

The v3.3.1 [benchmark harness](BENCHMARKS.md) adds no public `bo_forge` imports,
campaign CLI commands, or HTTP routes. Its `python -m benchmarks` commands run
from a checkout or extracted source archive; benchmark Python helpers are not a
stable public API and are excluded from the runtime wheel.

Top-level exports are resolved lazily. Names, signatures, `__all__`,
star imports, and `dir(bo_forge)` remain compatible; importing the package alone
does not load optimizer or plotting dependencies.

Implementation modules such as `bo_forge.config`, `bo_forge.logs`,
`bo_forge.validation`, `bo_forge.suggestions`, `bo_forge.diagnostics`, and
`bo_forge.session` remain importable through compatibility facades. Their
private helpers and the `_config`, `_campaign`, `_optimization`, and
`_diagnostics` packages are not part of the stable public surface.

`bo_forge.application` is the shared internal non-HTTP workflow layer. The
optional FastAPI implementation belongs to `bo_forge_api`; existing
`bo_forge_app.api`, `api_cli`, `stages`, and `service` imports remain available
for v3 compatibility but are not public BO APIs.

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
- `FitMetadata`
- `LogBusyError`
- `LogConflictError`
- `LogValidationError`
- `LogWriteError`
- `ModelConfig`
- `ObjectiveConfig`
- `PredictiveEvaluationResult`
- `ProvenanceError`
- `ProvenanceRecoveryRequired`
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
- `model_predictive_evaluation`
- `pareto_front`
- `pareto_summary`
- `provenance_summary`
- `recover_provenance`
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
Use `model_summary(config, df, *, metadata=None)` or `CampaignSession.model_summary()` to inspect
the configured profile, model class, covariance profile, fitting-row count, and
train-Y variance use. Use `model_profile_comparison(config, df)` or
`CampaignSession.model_profile_comparison()` to compare supported profiles on
the current observed fitting rows without changing the configured profile or
CSV log. Comparison rows include `fit_status` and `fit_message` so failed or
insufficient profile fits stay visible in tables and plots.
Existing comparison columns (`rmse_model_space`, `mae_model_space`, and
`mean_predicted_std` included) are retained with an added
`evaluation_scope=in_sample`. They describe training-row fit, not held-out
prediction or evidence for selecting a profile.
`FitMetadata` is a top-level exported frozen record of fit evidence, supplied as
`model_summary(config, df, metadata=fit_metadata)`. Its `as_dict()` method returns
a mapping for inspection; it does not own or expose a fitted model.
`model_summary(config, df)` reads no ambient fit history: without explicit
`metadata`, `last_fit_status` and `fallback_status` report `not_recorded`;
`last_fit_warning_count` is `0` and `last_fit_warnings` is empty. `CampaignSession` owns
its matching fit metadata; another session or an unrelated fit cannot populate
that session's history. Fit metadata is not durable campaign provenance.

### Predictive Evaluation

```python
model_predictive_evaluation(config, df, profiles=None, *, folds=5, seed=0)
campaign.model_predictive_evaluation(profiles=None, *, folds=5, seed=0)
```

Both return `PredictiveEvaluationResult(summary, predictions, fold_outcomes,
metadata)`: profile summary and held-out prediction tables, explicit per-fold
outcomes (including failures), and evaluation metadata. Evaluation runs only
when explicitly requested, does not mutate config/CSV inputs, and never selects
a model or generates campaign suggestions.

Summary metrics are `rmse`, `mae`, `mean_nlpd`, `interval_coverage`, and
`mean_interval_width`; summary `fit_status` is `complete` or `incomplete`.
An appended `fit_message` is empty on completion and otherwise explains failed
folds or aggregate numerical failure. Incomplete profiles retain no aggregate metrics.
Prediction rows identify `model_profile`, `fold`, and `row_id` and include
`predicted_mean`, `predicted_variance`, `predicted_std`, `residual`,
`standardized_residual`, `negative_log_predictive_density`, `interval_lower`,
`interval_upper`, and `interval_covered`. Inspect fold failures rather than
treating an incomplete profile as a successful comparison.

The evaluator supports standard single-objective campaigns only. Context,
replicates, structured stages, fidelity, and multi-objective campaigns are
rejected. It requires 5..200 observed rows, 2..5 folds, at least two training
rows in every fold, and no duplicate designs. Use the same profiles, folds, seed,
and input rows for a reproducible comparison. Small adaptive datasets and one
split do not establish generalization, calibrated uncertainty, or a best model.
Omitting `profiles` evaluates the configured profile only. An explicit sequence
must contain distinct supported profile names; `seed` is a nonnegative integer.

Predictive variance includes observation noise and is in original objective
units squared, not latent-function variance or standardized model units.
Means, residuals, and standard deviations use original objective units,
including the original sign for minimization objectives; standardized residuals
are dimensionless. Read fold failures alongside aggregate metrics.

```python
result = campaign.model_predictive_evaluation(
    profiles=["default", "smooth"], folds=3, seed=0,
)
result.summary
result.predictions
result.fold_outcomes
result.metadata
result.export("reports/evaluation")
result.plot_predictions(save_path="reports/evaluation/predictions.png")
result.plot_residuals(save_path="reports/evaluation/residuals.png")
```

`result.export(output_dir)` requires a **new destination directory** and refuses
overwrite. It creates exactly `summary.csv`, `predictions.csv`, `fold_outcomes.csv`,
and `metadata.json`, not plots or campaign state. Do not pre-create `output_dir`.
The example's plots are separate explicit writes after export.
Results obtained through a session retain source-path guards for table and plot
exports, including campaign-file aliases and reserved legacy manifest paths.
Standalone in-memory evaluations cannot identify source files; callers must
choose separate artifact paths. Exported result metadata remains unchanged.
The four files are prepared in a temporary sibling directory and published
atomically without overwrite on macOS/Linux. Failed exports leave no partial
final bundle and can retry from the same result; an existing or concurrently
created destination raises `FileExistsError` and is never removed.
Evaluation owns a config/data snapshot. File-loaded sessions verify config/log/
manifest identity before and after evaluation and reject changes with
`LogConflictError`. Standalone calls remain filesystem-independent, and returned
results remain historical snapshots. See [Predictive Evaluation](PREDICTIVE_EVALUATION.md).
`result.plot_predictions(save_path=None)` and
`result.plot_residuals(save_path=None)` plot the stored held-out results without
refitting; omit `save_path` for an unsaved figure. The existing model-diagnostics
and model-comparison plots remain in-sample. See the self-contained
[20-row tutorial](../notebooks/23_predictive_diagnostics.ipynb).

New campaigns initialized with `CampaignSession.initialize(config_path, log_path)`
receive a versioned provenance manifest beside the CSV log. Use
`provenance_summary(config_path, log_path)` or
`CampaignSession.provenance_summary()` for ordered, read-only identity and
integrity fields. Existing `CampaignSession.from_files()` calls remain valid and use
`provenance_policy="compatible"`; pass `provenance_policy="required"` to reject a
missing sidecar. `reload()` preserves that policy, and any present manifest is enforced
in both modes. Campaigns without a manifest are reported as `legacy` and are not adopted
automatically.

Use `recover_provenance(config_path, log_path,
expected_log_fingerprint=...)` to explicitly finalize or cancel a recoverable pending
transaction. It changes only manifest bytes. Ordinary session operations raise
`ProvenanceRecoveryRequired`, a `LogConflictError` subclass, until recovery completes.
`ProvenanceError` identifies malformed, unreadable, unsupported, required-but-missing,
or path-invalid manifests. Because schema v1 has no marker inside legacy CSVs, callers
must move and back up a managed CSV with its sidecar. See
[PROVENANCE.md](PROVENANCE.md) for reason codes, recovery, and the trust boundary.

The top-level helpers `adopt_provenance(config_path, log_path)`,
`migrate_provenance(config_path, log_path)`, and
`accept_provenance_config(config_path, log_path)` return JSON-compatible preview mappings.
`fork_campaign(config_path, log_path, destination, config_changes=None)` previews a child;
`config_changes` is keyword-only and allows `campaign_name`, `bo`, and `model`.
All four accept keyword-only `apply=False`, `reason=""`, and `expected_identities=None`.
Apply with a reason and the unchanged preview's `expected_identities`; then load a new
session. Existing sessions reject manifest identity changes even when CSV bytes agree.
New campaigns use schema v2; older schema-v1-only BO Forge releases cannot load them.

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

CampaignSession plot methods preserve their existing signatures and one-path,
one-file behavior. v3 figures use scoped Matplotlib settings, a shared semantic
color registry, white opaque backgrounds, and 600 dpi PNG export. Plot data,
direction conventions, and PDF support are unchanged.

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
