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
| `source` | Yes | One of `manual`, `sobol`, `log_ei`, or `qlog_ei`. |
| variable columns | Yes | One column per configured variable, in YAML order, stored in original user units. |
| objective column | Yes | The configured objective name, such as `activity` or `defect_rate`. |
| `predicted_mean` | Yes | Optional model prediction for suggested model-based rows. Blank is allowed. |
| `predicted_std` | Yes | Optional posterior standard deviation. Blank is allowed. |
| `acquisition` | Yes | Optional acquisition value. Blank is allowed. |

## 🚦 Status Rules

Suggested rows:

- `status` must be `suggested`.
- The objective cell must be blank.
- Variable values must be filled and inside bounds.
- `source` is usually `sobol`, `log_ei`, or `qlog_ei`.

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
    "../examples/simple_2d_maximise_logei_working_log.csv",
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
- Historical manual duplicates should be cleaned before relying on model-based suggestions.
