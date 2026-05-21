# 🧾 CSV Schema Reference

BO Forge campaign logs are plain CSV files. The schema is deliberately strict so a campaign can be resumed safely after manual edits.

## 📐 Canonical Column Order

```text
row_id,iteration,status,source,<variable columns...>,<objective column>,predicted_mean,predicted_std,acquisition
```

For `configs/simple_2d_maximise_logei.yaml`, the concrete columns are:

```text
row_id,iteration,status,source,precursor_ratio,annealing_temperature,activity,predicted_mean,predicted_std,acquisition
```

## 📋 Column Reference

| Column | Required | Meaning |
| --- | --- | --- |
| `row_id` | Yes | Unique row identifier. Suggestions keep the same `row_id` when marked observed. |
| `iteration` | Yes | Non-negative integer campaign iteration. New suggestions use the next iteration. |
| `status` | Yes | Either `suggested` or `observed`. |
| `source` | Yes | One of `manual`, `sobol`, `random`, `log_ei`, or `qlog_ei`. |
| variable columns | Yes | One column per configured variable, in YAML order, stored in original user units. |
| objective column | Yes | The configured objective name, such as `activity` or `defect_rate`. |
| `predicted_mean` | Yes | Optional model prediction for suggested model-based rows. Blank is allowed. |
| `predicted_std` | Yes | Optional posterior standard deviation. Blank is allowed. |
| `acquisition` | Yes | Optional acquisition value. Blank is allowed. |

## 🚦 Status Rules

Suggested rows:

- `status` must be `suggested`.
- The objective cell must be blank.
- Variable values must be filled and valid for the configured variable type.
- `source` is usually `sobol`, `random`, `log_ei`, or `qlog_ei`.

Observed rows:

- `status` must be `observed`.
- The objective cell must contain a numeric value.
- Manually entered historical rows should use `source=manual`.
- Rows that started as suggestions keep their original `row_id`, `iteration`, `source`, and variable values.

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
- validates before and after writing.

## ⬜ Blank Values

Blank objective values are valid only for `status=suggested`.

Blank `predicted_mean`, `predicted_std`, and `acquisition` values are allowed because Sobol/manual rows may not have model predictions.

## 🧬 Duplicate Rules

- `row_id` values must be unique.
- Exact duplicate variable rows are avoided when BO Forge generates suggestions.
- Mixed-variable duplicate checks use typed user-space values, such as `(0.5, 3, 0.1, "MeCN")`.
- Categorical variables stay as exact labels in CSV logs; v0.4.1 one-hot encoding is internal model-space behavior only.
- Historical manual duplicates should be cleaned before relying on model-based suggestions.

## 🧪 Variable Value Rules

BO Forge validates variables according to their YAML type:

| Type | CSV rule |
| --- | --- |
| `continuous` | Finite numeric value inside `lower`/`upper`. |
| `integer` | Integer-valued numeric value inside inclusive `lower`/`upper`; `3.0` is accepted. |
| `discrete` | Numeric value matching one configured choice after parsing; `0.10` can match `0.1`. |
| `categorical` | Exact configured string label; matching is case-sensitive and whitespace-padded values fail. |
