# 🛠️ Common Errors And Fixes

BO Forge tries to fail early with specific messages. Most errors come from hand-edited YAML or CSV files.

## 🚦 Quick Triage

- `Variable 'temperature' has lower >= upper`: check the YAML `lower` and `upper` values.
- `Campaign log must start with canonical columns`: make sure the CSV begins with `row_id,iteration,status,source`.
- `status='observed' but objective ... is blank`: fill the objective value or change the row back to `suggested`.
- `status='suggested' but objective ... is filled`: suggested rows must leave the objective blank until `mark_observed()` is called.
- `Cannot generate new suggestions while unresolved status='suggested' rows exist`: run the experiment and call `mark_observed()` before requesting another suggestion; in review-enabled campaigns, resolve `pending` or `accepted` review rows first.
- `Row ... has invalid source`: use only `manual`, `sobol`, `random`, `log_ei`, `qlog_ei`, `cost_log_ei`, `qlog_ehvi`, or `cost_qlog_ehvi` for cost-aware multi-objective campaigns.
- `Duplicate row_id`: every row needs a unique `row_id`.
- `Variable ... is outside bounds`: check the variable value against the YAML bounds.
- `violates constraint`: check the row values against the YAML `constraints` block.
- `invalid replicate_group` or `invalid replicate_index`: check explicit replicate metadata.
- `unknown stage` or `inactive variable`: check structured-campaign `stages:` and blank inactive variable cells.

## ⚙️ Config Errors

### `Config file must contain a YAML mapping at the top level.`

The YAML file is empty or starts with a list.

Fix: make the top level a mapping with `campaign_name`, `objective`, `variables`, and `bo`.

### `Variable 'temperature' has lower >= upper`

The variable bounds are reversed or equal.

Fix: set `lower` below `upper`.

```yaml
lower: 300
upper: 800
```

### `Objective 'activity' has invalid direction`

The objective direction is not supported.

Fix: use exactly one of:

```yaml
direction: maximize
```

or:

```yaml
direction: minimize
```

### `Variable '...' has unsupported type`

The variable type is not recognised.

Fix: use one of:

```text
continuous
integer
discrete
categorical
```

### `unsupported keys for type='categorical'`

The YAML contains keys that do not belong to that variable type.

Fix: use `lower`/`upper` for `continuous` and `integer`, and `values` for `discrete` and `categorical`.

### `duplicate discrete value after numeric parsing`

Discrete values are compared numerically, so `1` and `1.0` are duplicates.

Fix: remove duplicate numeric choices.

### `whitespace-padded categorical value`

Categorical labels are exact strings.

Fix: remove leading/trailing spaces from the YAML or CSV value.

### `Constraint '...' references unknown variable`

The constraint expression uses a name that is not one of the configured variables.

Fix: use exact variable names from the YAML `variables` list.

### `Constraint '...' uses unsupported syntax`

The constraint expression contains something outside the safe expression subset, such as a function call or attribute access.

Fix: use only variable names, constants, arithmetic, unary `+`/`-`, boolean logic, comparisons, and parentheses.

### `Cost expression references unknown variable`

The cost expression uses a name that is not a configured variable.

Fix: use exact variable names from the YAML `variables` list.

### `Cost expression must evaluate to a numeric value`

The cost expression returns a final boolean or string instead of a finite non-negative number.

Fix: make the final expression numeric. Boolean arithmetic inside a numeric expression is allowed, for example:

```yaml
expression: "1.0 + 2.0 * (solvent == 'Water')"
```

### `Config must define either 'objective' or 'objectives', not both`

Single-objective campaigns use `objective:`. Multi-objective campaigns use `objectives:`.

Fix: keep exactly one objective section.

### `objectives must contain at least two objectives`

BO Forge supports coupled multi-objective campaigns with at least two objectives. The primary tested range in v1.3.0 is two to four objectives.

Fix: define at least two objective mappings, each with `name`, `direction`, and `reference_point`.

### `reference_point`

Multi-objective reference points must be numeric and finite.

Fix: set a user-facing value that is meaningfully worse than the region of interest for that objective.

### `Duplicate stage name`

Structured campaign stage names must be unique.

Fix: give every item in `stages:` a distinct non-empty `name`.

### `Stage '...' references unknown variable`

A structured stage lists a variable that is not present in the top-level
`variables:` list.

Fix: use exact configured variable names in every stage's `variables:` list.

### `Structured campaigns with cost are not supported`

v1.3.0 validates structured logs, but cost-aware structured campaigns are
deferred because inactive variables are intentionally blank.

Fix: remove either `stages:` or `cost:` from the config.

## 🧾 CSV Schema Errors

### `Campaign log is missing required columns`

The CSV header does not match the config.

Fix: check the canonical column order in `CSV_SCHEMA.md`, including the exact objective and variable names from YAML.

### `Campaign log columns are not in canonical order`

The right columns are present but ordered incorrectly.

Fix: reorder the CSV header to:

```text
row_id,iteration,status,source,<variables...>,<objective>,predicted_mean,predicted_std,acquisition
```

For structured campaigns, `stage` belongs immediately after `source`.

### `Row '...' has unknown stage`

The CSV row has a `stage` value that does not match the configured `stages:`.

Fix: use one of the configured stage names.

### `inactive variable`

A structured-campaign row has a filled value for a variable that is not active
in that row's stage.

Fix: leave inactive variable cells blank. BO Forge intentionally rejects
ignored inactive values so hand-edited CSV logs remain unambiguous.

### `Duplicate row_id`

Two rows use the same identifier.

Fix: give every row a unique `row_id`.

### `Row '...' has invalid status`

The status is not recognised.

Fix: use only:

```text
suggested
observed
```

### `Row '...' has invalid source`

The source is not recognised.

Fix: use only:

```text
manual
sobol
random
log_ei
qlog_ei
cost_log_ei
qlog_ehvi
cost_qlog_ehvi
```

### `review_status is not 'accepted'`

In review-enabled campaigns, a suggested row must be accepted before it can be marked observed.

Fix: call `review_suggestion(..., decision="accept")` or use the CLI `bo-forge review --decision accept` before `mark_observed()`.

### `review_note containing a newline`

Review notes must stay on one CSV row.

Fix: use a one-line note. `review_suggestion()` strips leading and trailing whitespace.

### `invalid replicate_group`

The replicate group is blank, whitespace-padded, or contains a newline.

Fix: use a simple one-line identifier such as `group_0`.

### `invalid replicate_index`

The replicate index is negative, non-integer, blank, or non-numeric.

Fix: use zero-based integer values within each group, such as `0`, `1`, and `2`.

### `Duplicate replicate row`

Two rows have the same `(replicate_group, replicate_index)` pair.

Fix: make each `replicate_index` unique within its group.

### `Rows with the same design must share one replicate_group`

Replicate-aware duplicate rows are allowed only when they are explicitly grouped.

Fix: put repeated measurements of the same design in one shared `replicate_group`, or remove the duplicate design row.

## 🎯 Objective-State Errors

### `status='observed' but objective ... is blank`

The row says the experiment has been observed, but no result is present.

Fix: enter the objective value, or change the row back to `suggested`.

For multi-objective campaigns, every observed row must contain all configured objective values. Partial objective rows are not supported in v1.1.

### `status='suggested' but objective ... is filled`

A suggested row already has a result value.

Fix: use `mark_observed()` to perform the transition, or manually set `status=observed` only if the rest of the row is valid.

For multi-objective campaigns, suggested rows must leave every objective column blank until all coupled objective values are available.

### `objective_value is not valid for multi-objective campaign logs`

The single-value transition path is only for single-objective campaigns.

Fix: pass objective values keyed by name:

```python
campaign.mark_observed(
    row_id="...",
    objective_values={"yield_score": 71.2, "waste_score": 13.4},
)
```

With the CLI, repeat `--objective` once per objective:

```bash
bo-forge mark-observed ... --objective yield_score=71.2 --objective waste_score=13.4
```

### `Cannot generate new suggestions while unresolved status='suggested' rows exist`

BO Forge refuses to suggest more experiments while there is an outstanding suggestion.

Fix: run that experiment and call `mark_observed()`, or use review decisions when review is enabled. `review_status=rejected` and `review_status=deferred` rows do not block new suggestions.

### `Row '...' violates constraint`

The row is structurally valid, but the configured feasibility rules reject its variable values.

Fix: change the row values or update the constraint if the campaign definition is wrong. Constraints apply to all rows, including `manual`, `sobol`, `random`, `log_ei`, `qlog_ei`, and `cost_log_ei`.

### `Could not generate enough feasible, non-duplicate suggestions`

BO Forge exhausted its bounded retry loop while filtering infeasible, exact-duplicate, or near-duplicate candidates.

Fix: check whether constraints are too restrictive, whether the feasible design space is exhausted, or whether `bo.min_normalized_distance` is too large.

### `remaining budget may be too small`

The cost-aware suggestion loop could not find enough feasible candidates within the remaining budget.

Fix: check `campaign.cost_summary()`, the configured `cost.budget`, and the cost expression. Observed rows consume actual cost when present; accepted pending suggestions reserve estimated cost.

## 🔢 Numeric And Bounds Errors

### `non-numeric value for variable`

A variable cell contains text or an empty value.

Fix: enter a numeric value in original user units.

For `categorical` variables, use one of the exact configured labels instead.

### `variable ... outside bounds`

A CSV variable value falls outside the YAML bounds.

Fix: correct the CSV value or widen the YAML bounds if the campaign definition was wrong.

### `non-numeric objective`

An observed objective cell contains text.

Fix: replace it with a numeric result.

### `negative value for column 'cost_estimate'`

Cost columns must be finite and non-negative.

Fix: correct the cost value, or leave `cost_actual` blank until the experiment has been run.

### `cost_estimate inconsistent with cost expression`

A filled `cost_estimate` does not match the deterministic YAML `cost.expression`.

Fix: update `cost_estimate` to the expression result for that row, or leave realised deviations in `cost_actual`.

### `objective_std` or `objective_sem` is blank/NaN

For single-replicate groups, BO Forge reports `objective_std = NaN` and `objective_sem = NaN` by definition.

Fix: add more observed rows to the same `replicate_group` if you need a group-level standard deviation or SEM.

## ✅ Fast Validation Check

Run this from the repository root:

```python
from bo_forge import CampaignConfig, load_campaign_log, validate_campaign_data

config = CampaignConfig.from_yaml("configs/01_simple_2d_maximise_logei.yaml")
df = load_campaign_log("examples/01_simple_2d_maximise_logei_campaign_log.csv", config)
validate_campaign_data(config, df)
```

No output means the config and log passed validation.
