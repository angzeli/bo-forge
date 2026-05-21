# 🛠️ Common Errors And Fixes

BO Forge tries to fail early with specific messages. Most errors come from hand-edited YAML or CSV files.

## 🚦 Quick Triage

- `Variable 'temperature' has lower >= upper`: check the YAML `lower` and `upper` values.
- `Campaign log must start with canonical columns`: make sure the CSV begins with `row_id,iteration,status,source`.
- `status='observed' but objective ... is blank`: fill the objective value or change the row back to `suggested`.
- `status='suggested' but objective ... is filled`: suggested rows must leave the objective blank until `mark_observed()` is called.
- `Cannot generate new suggestions while unresolved status='suggested' rows exist`: run the experiment and call `mark_observed()` before requesting another suggestion.
- `Row ... has invalid source`: use only `manual`, `sobol`, `random`, `log_ei`, or `qlog_ei`.
- `Duplicate row_id`: every row needs a unique `row_id`.
- `Variable ... is outside bounds`: check the variable value against the YAML bounds.

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
```

## 🎯 Objective-State Errors

### `status='observed' but objective ... is blank`

The row says the experiment has been observed, but no result is present.

Fix: enter the objective value, or change the row back to `suggested`.

### `status='suggested' but objective ... is filled`

A suggested row already has a result value.

Fix: use `mark_observed()` to perform the transition, or manually set `status=observed` only if the rest of the row is valid.

### `Cannot generate new suggestions while unresolved status='suggested' rows exist`

BO Forge refuses to suggest more experiments while there is an outstanding suggestion.

Fix: run that experiment and call `mark_observed()`, or remove the suggested row if it should be abandoned.

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

## ✅ Fast Validation Check

Run this from the repository root:

```python
from bo_forge import CampaignConfig, load_campaign_log, validate_campaign_data

config = CampaignConfig.from_yaml("configs/simple_2d_maximise_logei.yaml")
df = load_campaign_log("examples/simple_2d_maximise_logei_campaign_log.csv", config)
validate_campaign_data(config, df)
```

No output means the config and log passed validation.
