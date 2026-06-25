# 🧾 CSV Schema Reference

BO Forge campaign logs are plain CSV files. The schema is deliberately strict so a campaign can be resumed safely after manual edits.

## 📐 Canonical Column Order

```text
row_id,iteration,status,source,<variable columns...>,<objective column>,predicted_mean,predicted_std,acquisition
```

When `stages:` is configured, add `stage` immediately after `source`.

When `review.enabled: true`, add `review_status,review_note` immediately after `source`, or immediately after `stage` when structured stages are configured.

When `replicates.enabled: true`, add `replicate_group,replicate_index` immediately after `source`, after `stage`, or after the review columns when review is also enabled.

When `cost` is configured, add `cost_estimate,cost_actual` immediately after
the objective column and add `utility` immediately after `acquisition`. In
v1.5.0, `stages:` cannot be combined with `cost:`.

When `fidelity:` is configured, no new CSV columns are added. The fidelity
variable stays in the normal variable columns and stores the user-facing
fidelity value measured for that row.

When `context:` is configured, no new CSV columns are added. Context variables
stay in the normal variable columns, store observed user-facing values, and are
fixed to supplied/default values during suggestion generation.

The non-structured cost + review + replicates schema is:

```text
row_id,iteration,status,source,review_status,review_note,replicate_group,replicate_index,<variable columns...>,<objective column>,cost_estimate,cost_actual,predicted_mean,predicted_std,acquisition,utility
```

For v1.1 multi-objective campaigns, the schema scales with the configured objective order:

```text
row_id,iteration,status,source,[stage],[review_status,review_note],[replicate_group,replicate_index],<variable columns...>,<objective_1>,...,<objective_m>,[cost_estimate,cost_actual],predicted_mean_<objective_1>,predicted_std_<objective_1>,...,predicted_mean_<objective_m>,predicted_std_<objective_m>,acquisition,[utility]
```

For v1.4.0 structured campaigns, the bracketed `[stage]` marker is not
compatible with `[cost_estimate,cost_actual]` or `[utility]`; cost-aware
structured campaigns remain deferred.

Multi-objective campaigns in v1.1 assume coupled objective evaluation: every observed row contains all configured objective values. Suggested rows keep every objective column blank until the experiment is complete.

For `configs/01_simple_2d_maximise_logei.yaml`, the concrete columns are:

```text
row_id,iteration,status,source,precursor_ratio,annealing_temperature,activity,predicted_mean,predicted_std,acquisition
```

## 📋 Column Reference

| Column | Required | Meaning |
| --- | --- | --- |
| `row_id` | Yes | Unique row identifier. Suggestions keep the same `row_id` when marked observed. |
| `iteration` | Yes | Non-negative integer campaign iteration. New suggestions use the next iteration. |
| `status` | Yes | Either `suggested` or `observed`. |
| `source` | Yes | One of `manual`, `sobol`, `random`, `log_ei`, `qlog_ei`, `cost_log_ei`, `qmf_kg`, `qlog_ehvi`, or `cost_qlog_ehvi` where supported by the config. |
| `stage` | If `stages:` configured | One configured stage name. Structured logs place this column immediately after `source`. |
| `review_status` | If review enabled | One of `pending`, `accepted`, `rejected`, or `deferred`. |
| `review_note` | If review enabled | Optional one-line human note. Newlines are rejected. |
| `replicate_group` | If replicates enabled | Nonblank replicate-group identifier. Rows in a group share one design. |
| `replicate_index` | If replicates enabled | Zero-based non-negative integer, unique within each replicate group. |
| variable columns | Yes | One column per configured variable, in YAML order, stored in original user units. |
| objective column | Yes | The configured objective name, such as `activity` or `defect_rate`. |
| objective columns | Multi-objective only | The configured objective names, in YAML order. Observed rows must fill every objective value. |
| `cost_estimate` | If cost configured | Optional finite non-negative estimated cost. If filled, it must match the deterministic cost expression. Generated suggestions fill this column. |
| `cost_actual` | If cost configured | Optional finite non-negative realised cost entered when marking observed. |
| `predicted_mean` | Yes | Optional model prediction for suggested model-based rows. Blank is allowed. |
| `predicted_std` | Yes | Optional posterior standard deviation. Blank is allowed. |
| `acquisition` | Yes | Optional acquisition value. Blank is allowed. |
| `utility` | If cost configured | Optional cost-aware utility. Model-based cost-aware suggestions fill this column. |

For multi-objective campaigns, prediction columns are objective-specific:

- `predicted_mean_<objective_name>`;
- `predicted_std_<objective_name>`.

## 🚦 Status Rules

Suggested rows:

- `status` must be `suggested`.
- The objective cell must be blank.
- Variable values must be filled and valid for the configured variable type, except inactive structured-campaign variables, which must be blank.
- `source` is usually `sobol`, `random`, `log_ei`, `qlog_ei`, `qlog_ehvi`, or `cost_qlog_ehvi`.
- For multi-fidelity qMFKG model suggestions, `source=qmf_kg`.
- For review-enabled campaigns, `review_status` can be `pending`, `accepted`, `rejected`, or `deferred`.
- For single-objective cost-aware model suggestions, `source=cost_log_ei` and `utility = acquisition - cost.weight * cost_estimate`.
- For multi-objective cost-aware model suggestions, `source=cost_qlog_ehvi`, `acquisition` stores the qLogEHVI batch acquisition value, and `utility = acquisition - cost.weight * total_batch_cost` is repeated on every row in the selected batch.

Observed rows:

- `status` must be `observed`.
- The objective cell must contain a numeric value.
- Manually entered historical rows should use `source=manual`.
- Rows that started as suggestions keep their original `row_id`, `iteration`, `source`, and variable values.
- For review-enabled campaigns, observed rows must have `review_status=accepted`.
- For cost-aware campaigns, `cost_actual` may be filled when the experiment is marked observed.

If `review` is not enabled, review columns are unsupported extras. If `cost` is not configured, cost and utility columns are unsupported extras. If `replicates` is not enabled, replicate columns are unsupported extras.

## 🧩 Structured Campaign Logs

v1.3.0 adds a minimal structured-campaign foundation through an optional
top-level `stages:` list:

```yaml
stages:
  - name: screen
    variables: [precursor_ratio, solvent]
  - name: refine
    variables: [precursor_ratio, annealing_temperature]
```

Rules:

- stage names must be unique non-empty strings;
- every stage variable must refer to a configured variable name;
- structured CSV logs must include `stage` immediately after `source`;
- every row's `stage` value must match one configured stage name;
- variables active for the row's stage must be filled and valid;
- inactive variables must be blank.
- constraints are evaluated for a row only when every variable referenced by the
  constraint is active in that row's stage;
- `stages:` cannot be combined with `cost:` in v1.5.0.

The blank-only inactive-variable rule is intentional. It keeps public CSV values
editable and prevents ignored inactive values from being confused with active
design settings.

v1.3.1 adds explicit stage-aware suggestions through
`CampaignSession.suggest_next(stage="...")` and `bo-forge suggest --stage ...`.
Generated structured suggestions populate the selected `stage`, fill only that
stage's active variables, and leave inactive variables blank.

v1.3.2 adds read-only structured inspection through `stage_summary()`,
`CampaignSession.stage_summary()`, `CampaignSession.plot_stage_diagnostics()`,
`bo-forge stage-summary`, and `bo-forge plot --kind stage-diagnostics`.
Structured campaign reports include configured stage state, active/inactive
variables, row counts, pending rows, best single-objective rows or
multi-objective Pareto counts where meaningful, and warnings for stages without
observations. Automatic stage transitions, combinations with multi-fidelity or
contextual workflows, cost-aware structured campaigns, and Streamlit structured
campaign creation remain deferred.

## 🧪 Multi-Fidelity Rules

v1.4.0 adds a conservative single-objective multi-fidelity workflow through an
optional top-level `fidelity:` section:

```yaml
fidelity:
  variable: fidelity
  target: 1.0
  fixed_cost: 0.01
  fidelity_cost_weight: 1.0
  num_fantasies: 64

bo:
  acquisition: qmf_kg
```

Rules:

- `fidelity.variable` must name one configured continuous variable;
- all variables in v1.4.0 multi-fidelity configs must be continuous;
- `fidelity.target` must lie within that variable's bounds;
- lower-fidelity observations are real measured objective values at that row's
  fidelity, not approximations stored in a separate column;
- the target fidelity is the value BO Forge ultimately optimizes for;
- qMFKG trades information gain against the configured affine fidelity cost
  model, evaluated on the normalized model-space fidelity coordinate;
- fidelity cost is separate from BO Forge's `cost:` budget/ranking feature;
- model-based qMFKG suggestions use `source=qmf_kg`;
- once model-based qMFKG begins, `batch_size` must be `1`.

Unsupported v1.4.0 combinations are intentional: `fidelity:` cannot be combined
with `objectives:`, `stages:`, `cost:`, or `replicates.enabled: true`.

## 🌐 Contextual BO Rules

v1.5.x adds a conservative single-objective contextual LogEI/qLogEI workflow:

```yaml
context:
  variables: [feedstock_acidity]
  default_values:
    feedstock_acidity: 0.5
```

Rules:

- context variables must be declared under `variables:`;
- at least one non-context decision variable must remain;
- context variables may be continuous, integer, discrete, or categorical;
- context variables are stored as normal CSV variable columns;
- suggestions require every context variable value, supplied through
  `context.default_values`, `context_values={...}`, app inputs, API payloads, or
  CLI `--context NAME=VALUE`;
- generated suggestions fill context variable columns with the fixed values;
- duplicate checks use the full variable row, so the same decision setting at a
  different context is a distinct design;
- constraints are evaluated against the full candidate, including fixed context
  values.
- `context_summary()`, `bo-forge context-summary`, and
  `bo-forge plot --kind context-diagnostics` inspect the same CSV rows and do
  not add schema columns.

Unsupported v1.5.x combinations are intentional: `context:` cannot be combined
with `objectives:`, `stages:`, `fidelity:`, `cost:`, or `replicates:`.

## 🔁 Suggested To Observed Transition

Use `mark_observed()`:

```python
from bo_forge import mark_observed

mark_observed(
    "../examples/01_simple_2d_maximise_logei_working_log.csv",
    row_id="suggested_row_id_here",
    objective_value=1.95,
)
```

`mark_observed()` updates the same row in place:

- fills the objective value;
- changes `status` from `suggested` to `observed`;
- preserves `row_id`, `iteration`, `source`, and variable values;
- validates the CSV structure before and after writing.

For structured campaigns, prefer `CampaignSession.mark_observed()` or pass
`config=config` to the low-level helper so BO Forge can validate the configured
stage and active-variable rules before writing.

For cost-aware campaigns, `mark_observed(..., actual_cost=...)` records a finite non-negative realised cost in `cost_actual`. For review-enabled campaigns, only `review_status=accepted` rows can be marked observed.

For multi-objective campaigns, use objective values keyed by every configured objective name:

```python
campaign.mark_observed(
    row_id="suggested_row_id_here",
    objective_values={"yield_score": 71.2, "waste_score": 13.4},
)
```

The keys must exactly match the configured objective names. Passing a single `objective_value` to a multi-objective campaign fails clearly.

Review decisions do not change `status`; they update `review_status` in place:

```python
from bo_forge import review_suggestion

review_suggestion(
    "../examples/07_cost_aware_human_review_working_log.csv",
    row_id="suggested_row_id_here",
    decision="accept",
    note="run next",
)
```

Allowed decisions are `accept`, `reject`, and `defer`.

## ⬜ Blank Values

Blank objective values are valid only for `status=suggested`.

Blank `predicted_mean`, `predicted_std`, and `acquisition` values are allowed because Sobol/manual rows may not have model predictions.

Blank `utility` is expected for initial Sobol/random suggestions, because no model acquisition exists yet. Blank `cost_actual` is allowed until the experiment has been run. In cost-aware replicate campaigns, a policy-driven `uncertain_best` repeat suggestion fills `cost_estimate` but may leave `utility` blank and keep `source=log_ei` or `qlog_ei`, because it is a repeat decision rather than a cost-utility-ranked new exploration candidate.

## 🧬 Duplicate Rules

- `row_id` values must be unique.
- Exact duplicate variable rows are avoided when BO Forge generates suggestions.
- Mixed-variable duplicate checks use typed user-space values, such as `(0.5, 3, 0.1, "MeCN")`.
- Categorical variables stay as exact labels in CSV logs; v0.4.1 one-hot encoding is internal model-space behavior only.
- When `bo.min_normalized_distance > 0`, near-duplicate checks use encoded model-space distance, not raw user units.
- Without `replicates.enabled: true`, repeated design rows fail validation.
- With `replicates.enabled: true`, repeated design rows are allowed only when they share one `replicate_group`.

## ✅ Constraint Rules

If a non-structured config defines `constraints`, every CSV row must satisfy every constraint regardless of `status` or `source`. In structured campaigns, a constraint is evaluated for a row only when every variable referenced by that constraint is active in the row's stage.

This means manual historical rows, Sobol/random initial suggestions, and LogEI/qLogEI model suggestions all follow the same feasibility rules. A row that violates a constraint fails CSV validation with the row ID, constraint name, and expression.

For multi-objective campaigns, constraints apply to every row in the same way. qLogEHVI suggestions are repaired and checked against configured variable domains, constraints, exact duplicates, and encoded-space near-duplicate thresholds.

## 🎯 Multi-Objective Rules

BO Forge supports `m >= 2` objectives with coupled evaluation. The primary tested range for v1.4.0 is `2 <= m <= 4`; larger objective counts are advanced usage because qLogEHVI, non-dominated partitioning, hypervolume, and visualization become more expensive.

- A config uses `objectives:` instead of `objective:`.
- Each objective requires `name`, `direction`, and a finite numeric `reference_point`.
- Reference points are written in user-facing objective units and should represent meaningfully worse outcomes than the region of interest.
- Observed rows must contain every objective value.
- Suggested rows must leave every objective value blank.
- Hypervolume is computed in internal maximisation space after applying objective directions.
- If no observed point dominates the reference point, hypervolume is reported as `0.0`.

Review, replicate, and deterministic cost metadata are supported for multi-objective campaigns in v1.4.0. Multi-objective cost-aware ranking uses qLogEHVI batch utility; cost is not modeled as another objective.

## 🧑‍⚖️ Review And Budget Rules

For review-enabled campaigns, blocking behavior is review-aware:

- non-review campaigns: any `status=suggested` row blocks new suggestions;
- review-enabled campaigns: `review_status=pending` and `review_status=accepted` suggestions block new suggestions;
- `review_status=rejected` and `review_status=deferred` suggestions stay auditable and duplicate-protected, but do not block new suggestions.

For cost-aware campaigns, budget accounting uses:

- observed rows: `cost_actual` if present, otherwise `cost_estimate`;
- accepted pending suggestions: reserve `cost_estimate`;
- pending, rejected, and deferred suggestions: no budget reservation.

## 🔬 Replicate Rules

Replicates are explicit CSV metadata, not silently inferred.

- `replicate_group` must be a nonblank string with no surrounding whitespace or newlines.
- `replicate_index` is zero-based, non-negative, integer-valued, and unique within each group.
- Rows in the same group must have identical typed user-space design values.
- Manual replicate rows are allowed when explicitly grouped.
- Generated exploration suggestions avoid existing designs, set `replicate_group=row_id`, and set `replicate_index=0`.
- For single-objective replicate campaigns with `suggestion_policy: uncertain_best`, BO Forge may intentionally suggest another observation in the current best replicate group. Those repeat suggestions reuse the existing `replicate_group` and use the next zero-based `replicate_index`.
- If an active repeat fills only part of the requested batch, remaining rows are normal exploration suggestions when budget and design-space constraints allow.
- Multi-objective replicate campaigns use group means plus replicate-derived `train_Yvar` for qLogEHVI fitting. Active repeat selection remains single-objective only in v1.4.0, so MO replicate configs default to `suggestion_policy: new_only` and explicit `uncertain_best` fails clearly.

Replicate summaries are group-level. Cost and review summaries remain row-level when those features are also enabled.

For model fitting, replicate-enabled campaigns use one training row per `replicate_group`. The training objective is the group mean. When at least one group has 2+ observations, BO Forge passes replicate-derived observation variance to BoTorch as `train_Yvar`: repeated groups use `std^2 / n_replicates`, singleton groups use the weighted pooled replicate variance, and every value is clamped by `replicates.noise_floor`. If no group has repeated observations yet, BO Forge keeps the current learned-noise GP behavior.

Aggregate replicate summaries use these columns:

```text
replicate_group,<variable columns...>,n_replicates,objective_mean,objective_std,objective_sem,objective_min,objective_max
```

For single-replicate groups, `objective_std` and `objective_sem` are `NaN`.

For multi-objective replicate summaries, each objective gets its own statistic columns:

```text
replicate_group,<variable columns...>,n_replicates,<objective>_mean,<objective>_std,<objective>_sem,<objective>_min,<objective>_max,...
```

Pareto fronts and hypervolume use one group-mean objective vector per `replicate_group` when multi-objective replicates are enabled. Multi-objective qLogEHVI also consumes replicate-derived per-objective `train_Yvar`, but active repeat selection for multi-objective campaigns remains deferred.

## 🧪 Variable Value Rules

BO Forge validates variables according to their YAML type:

| Type | CSV rule |
| --- | --- |
| `continuous` | Finite numeric value inside `lower`/`upper`. |
| `integer` | Integer-valued numeric value inside inclusive `lower`/`upper`; `3.0` is accepted. |
| `discrete` | Numeric value matching one configured choice after parsing; `0.10` can match `0.1`. |
| `categorical` | Exact configured string label; matching is case-sensitive and whitespace-padded values fail. |
