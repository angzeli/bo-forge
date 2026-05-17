# 🚀 Quickstart And Usage

This page covers setup, the quickstart script, the notebook session API, example notebooks, and diagnostics.

## 🧰 Setup

Create a dedicated environment at the project root:

```bash
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev]"
```

The `dev` extra includes pytest, Ruff, and enough Jupyter tooling to open and execute the example notebooks from a fresh clone.

Run checks:

```bash
./.venv/bin/pytest
./.venv/bin/ruff check .
```

## ✅ Script Quickstart

Run the clean script example:

```bash
./.venv/bin/python examples/quickstart.py
```

It copies the seed CSV log to an ignored working file, requests one suggestion, simulates one result, records that result with `mark_observed()`, and reloads the campaign log.

## 🔁 Session API

The v0.2 notebook workflow should usually start with `CampaignSession`:

```python
from pathlib import Path
import shutil

from bo_forge.session import CampaignSession

seed_log_path = Path("examples/simple_2d_campaign_log.csv")
log_path = Path("examples/simple_2d_working_log.csv")
shutil.copyfile(seed_log_path, log_path)

campaign = CampaignSession.from_files(
    config_path="configs/simple_2d.yaml",
    log_path=log_path,
)

campaign.validate()
campaign.summary()

suggestions = campaign.suggest_next(batch_size=1)
campaign.append_suggestions(suggestions)

# After running the suggested experiment:
campaign.mark_observed(
    row_id=suggestions.loc[0, "row_id"],
    objective_value=1.95,
)

campaign.plot_progress()
```

`suggest_next()` does not mutate the session or write to disk. `append_suggestions()`, `mark_observed()`, and `reload()` refresh `campaign.df`.

| API | Description |
| --- | --- |
| `CampaignSession.from_files(config_path, log_path)` | Load YAML config and CSV log into one session object. |
| `campaign.validate()` | Validate the current in-memory campaign DataFrame. |
| `campaign.summary()` | Return a two-column DataFrame with campaign counts, status, and best observation. |
| `campaign.observed_data()` | Return observed rows used for model fitting. |
| `campaign.pending_suggestions()` | Return unresolved `status='suggested'` rows. |
| `campaign.campaign_status()` | Return the current campaign status without mutating state or writing to disk. |
| `campaign.best_observation()` | Return a canonical-column-order copy of the best observed row, or an empty canonical DataFrame. |
| `campaign.suggest_next(batch_size=None)` | Generate suggestions without mutating `campaign.df` or writing to disk. |
| `campaign.append_suggestions(suggestions)` | Append suggested rows to the CSV log and refresh `campaign.df`. |
| `campaign.mark_observed(row_id, objective_value)` | Mark one pending suggestion observed, write the result, and refresh `campaign.df`. |
| `campaign.reload()` | Reload the CSV log from disk into `campaign.df`. |
| `campaign.plot_progress()` | Plot objective and best-so-far progress; returns figure/axes objects. |
| `campaign.plot_diagnostics()` | Plot basic observed design-space diagnostics; returns figure/axes objects. |

## 🧱 Lower-Level API

The explicit backend functions remain available:

```python
from pathlib import Path
import shutil

from bo_forge import (
    CampaignConfig,
    append_suggestions,
    load_campaign_log,
    mark_observed,
    suggest_next,
)

config = CampaignConfig.from_yaml("configs/simple_2d.yaml")
seed_log_path = Path("examples/simple_2d_campaign_log.csv")
log_path = Path("examples/simple_2d_working_log.csv")
shutil.copyfile(seed_log_path, log_path)

df = load_campaign_log(log_path, config)
suggestions = suggest_next(config, df)
append_suggestions(log_path, suggestions)

# After running the suggested experiment:
mark_observed(log_path, row_id=suggestions.loc[0, "row_id"], objective_value=1.95)
```

## ⚙️ Example Configs

- `configs/simple_2d.yaml`: maximises photocatalyst-style `activity`.
- `configs/simple_2d_minimise.yaml`: minimises process `defect_rate`.

## 📓 Example Notebooks

Open `notebooks/01_maximisation_logei_campaign.ipynb` for a simulated end-to-end maximisation campaign using `configs/simple_2d.yaml` and `examples/simple_2d_campaign_log.csv`.

Open `notebooks/02_minimisation_qlogei_campaign.ipynb` for a shorter minimisation campaign using `configs/simple_2d_minimise.yaml` and `examples/simple_2d_minimise_campaign_log.csv`. It fills the Sobol initial design, then demonstrates one qLogEI batch BO round.

From a fresh clone:

```bash
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev]"
./.venv/bin/jupyter notebook notebooks/01_maximisation_logei_campaign.ipynb
```

The main notebook demonstrates the real sequential workflow:

1. load the current log
2. request one candidate
3. append it as `status=suggested`
4. run one experiment
5. enter one result with `mark_observed()`
6. reload the log and repeat

The notebooks write only ignored working files:

- `examples/simple_2d_working_log.csv`
- `examples/simple_2d_minimise_working_log.csv`
- `examples/latest_suggestions.csv`

## 📊 Diagnostics

The plotting helpers produce report-ready black-on-white figures, even when the notebook or IDE uses a dark theme. `plot_progress()` shows observed and best-so-far objective values, while `plot_diagnostics()` shows the observed design space for one- or two-variable campaigns.

Both functions return figure/axes objects and can optionally save figures:

```python
from bo_forge.diagnostics import plot_progress

plot_progress(config, df, filename="progress.png")
```
