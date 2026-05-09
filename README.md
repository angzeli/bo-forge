# 🧪 BO Forge MVP v0.1 

BO Forge is a notebook-first Bayesian optimisation campaign tool. The notebook is the user workflow, while the reusable BO logic lives in the `bo_forge` Python package.

MVP v0.1 is a sequential campaign demo: define a problem, load a CSV log, suggest one experiment, enter one result, reload the log, and repeat.

MVP v0.1 deliberately supports only:

- continuous variables
- one objective
- maximize or minimize direction
- Sobol initial suggestions
- BoTorch `SingleTaskGP`
- LogEI for single suggestions and qLogEI for batches
- CSV campaign logs
- resume from existing logs
- basic diagnostics

It intentionally does not yet cover categorical variables, constraints, noisy BO, multi-objective optimisation, a CLI, or an app UI.

## 🚀 Setup 

Create a dedicated environment at the project root:

```bash
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev]"
```

Run the test suite:

```bash
./.venv/bin/pytest
```

Run lint checks:

```bash
./.venv/bin/ruff check .
```

## Minimal Use ✅

```python
from pathlib import Path

from bo_forge import (
    CampaignConfig,
    append_suggestions,
    load_campaign_log,
    mark_observed,
    suggest_next,
)

config = CampaignConfig.from_yaml("configs/simple_2d.yaml")
log_path = Path("examples/simple_2d_campaign_log.csv")

df = load_campaign_log(log_path, config)
suggestions = suggest_next(config, df)
append_suggestions(log_path, suggestions)

# After running the suggested experiment:
mark_observed(log_path, row_id=suggestions.loc[0, "row_id"], objective_value=1.95)
```

## Canonical CSV Schema

Campaign logs use this column order:

```text
row_id,iteration,status,source,<variable columns...>,<objective column>,predicted_mean,predicted_std,acquisition
```

Rules:

- `status` is `suggested` or `observed`.
- `source` is `manual`, `sobol`, `log_ei`, or `qlog_ei`.
- Suggested rows have blank objective values.
- Observed rows require objective values.
- A suggested experiment becomes observed by updating the same row with `mark_observed()`.
- `row_id`, `iteration`, `source`, and variable values are preserved when a result is entered.

## 📓 Example Notebook 

Open `notebooks/01_simulated_campaign.ipynb` for a simulated end-to-end campaign using `configs/simple_2d.yaml` and `examples/simple_2d_campaign_log.csv`.

The notebook demonstrates the real sequential workflow:

1. load the current log
2. request one candidate
3. append it as `status=suggested`
4. run one experiment
5. enter one result with `mark_observed()`
6. reload the log and repeat

The diagnostics use `bo_forge/plot_style.py`, which captures the bold axes, thicker spines, compact legends, and figure sizing used throughout the local PyTorch & BoTorch tutorial notebooks.

## 📊 Diagnostics 

The plotting helpers produce report-ready black-on-white figures, even when the notebook or IDE uses a dark theme. `plot_progress()` shows observed and best-so-far objective values, while `plot_diagnostics()` shows the observed design space for one- or two-variable campaigns.

Both functions return `(fig, ax)` and can optionally save figures:

```python
from bo_forge.diagnostics import plot_progress

plot_progress(config, df, filename="progress.png")
```

## 👤 Author 

Angze Li
