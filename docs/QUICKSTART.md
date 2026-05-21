# 🚀 Quickstart And Usage

This page covers setup, the quickstart script, CLI workflow, notebook session API, example notebooks, and diagnostics.

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

## 💻 CLI Quickstart

The CLI wraps the same `CampaignSession` workflow used in notebooks.

```bash
bo-forge validate \
  --config configs/simple_2d_maximise_logei.yaml \
  --log examples/simple_2d_maximise_logei_campaign_log.csv

bo-forge summary \
  --config configs/simple_2d_maximise_logei.yaml \
  --log examples/simple_2d_maximise_logei_campaign_log.csv

bo-forge suggest \
  --config configs/simple_2d_maximise_logei.yaml \
  --log examples/simple_2d_maximise_logei_working_log.csv \
  --output examples/simple_2d_maximise_logei_latest_suggestions.csv \
  --append
```

`bo-forge suggest` is non-mutating unless `--append` is passed. If both `--output` and `--append` are passed, BO Forge writes the suggestions CSV and appends the same suggestions to the canonical campaign log.

See [CLI.md](CLI.md) for the full command reference.

## 🔁 Session API

The v0.2 notebook workflow should usually start with `CampaignSession`:

```python
from pathlib import Path
import shutil

from bo_forge.session import CampaignSession

seed_log_path = Path("examples/simple_2d_maximise_logei_campaign_log.csv")
log_path = Path("examples/simple_2d_maximise_logei_working_log.csv")
shutil.copyfile(seed_log_path, log_path)

campaign = CampaignSession.from_files(
    config_path="configs/simple_2d_maximise_logei.yaml",
    log_path=log_path,
)

campaign.validate()
campaign.summary()
campaign.next_action()
campaign.report()
campaign.export_report("reports/latest_campaign_report.txt")

suggestions = campaign.suggest_next(batch_size=1)
campaign.append_suggestions(suggestions)

# After running the suggested experiment:
campaign.mark_observed(
    row_id=suggestions.loc[0, "row_id"],
    objective_value=1.95,
)

campaign.plot_progress(save_path="reports/progress.png")
campaign.plot_diagnostics(save_path="reports/diagnostics.png")
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
| `campaign.next_action()` | Return a one-row advisory DataFrame with campaign status, recommended action, reason, and suggested calls. |
| `campaign.report()` | Return read-only report tables for summary, next action, best observation, and pending suggestions. |
| `campaign.export_report(path)` | Write a deterministic plain-text campaign report and return the written path. |
| `campaign.best_observation()` | Return a canonical-column-order copy of the best observed row, or an empty canonical DataFrame. |
| `campaign.suggest_next(batch_size=None)` | Generate suggestions without mutating `campaign.df` or writing to disk. |
| `campaign.append_suggestions(suggestions)` | Append suggested rows to the CSV log and refresh `campaign.df`. |
| `campaign.mark_observed(row_id, objective_value)` | Mark one pending suggestion observed, write the result, and refresh `campaign.df`. |
| `campaign.reload()` | Reload the CSV log from disk into `campaign.df`. |
| `campaign.plot_progress(save_path=None)` | Plot objective and best-so-far progress; returns figure/axes objects. |
| `campaign.plot_diagnostics(save_path=None)` | Plot dimension-aware diagnostics; returns figure/axes objects. |

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

config = CampaignConfig.from_yaml("configs/simple_2d_maximise_logei.yaml")
seed_log_path = Path("examples/simple_2d_maximise_logei_campaign_log.csv")
log_path = Path("examples/simple_2d_maximise_logei_working_log.csv")
shutil.copyfile(seed_log_path, log_path)

df = load_campaign_log(log_path, config)
suggestions = suggest_next(config, df)
append_suggestions(log_path, suggestions)

# After running the suggested experiment:
mark_observed(log_path, row_id=suggestions.loc[0, "row_id"], objective_value=1.95)
```

## ⚙️ Example Configs

- `configs/simple_2d_maximise_logei.yaml`: maximises photocatalyst-style `activity`.
- `configs/simple_2d_minimise_qlogei.yaml`: minimises process `defect_rate`.
- `configs/simple_3d_maximise_logei.yaml`: maximises a three-variable synthetic activity.
- `configs/simple_4d_maximise_logei.yaml`: maximises a four-variable synthetic activity for CLI demos.
- `configs/simple_mixed_logei.yaml`: maximises a mixed continuous/integer/discrete/categorical synthetic yield.

## 🧪 Mixed-Variable Campaigns

v0.4 supports mixed-variable single-objective campaigns:

```yaml
variables:
  - name: catalyst_loading
    type: continuous
    lower: 0.02
    upper: 0.20

  - name: reaction_time
    type: integer
    lower: 10
    upper: 60

  - name: base_equivalents
    type: discrete
    values: [0.1, 0.2, 0.5, 1.0]

  - name: solvent
    type: categorical
    values: [MeCN, EtOH, Water]

bo:
  initial_design_method: sobol
```

`initial_design_method` can be `sobol` or `random`. Model-based mixed suggestions use a latent unit-cube representation internally, then decode and repair suggestions back to valid user-facing values before returning them.

## 📓 Example Notebooks

Open `notebooks/01_maximisation_logei_campaign.ipynb` for a simulated end-to-end maximisation campaign using `configs/simple_2d_maximise_logei.yaml` and `examples/simple_2d_maximise_logei_campaign_log.csv`.

Open `notebooks/02_minimisation_qlogei_campaign.ipynb` for a shorter minimisation campaign using `configs/simple_2d_minimise_qlogei.yaml` and `examples/simple_2d_minimise_qlogei_campaign_log.csv`. It fills the Sobol initial design, then demonstrates one qLogEI batch BO round.

Open `notebooks/03_three_variable_campaign.ipynb` for a compact 3D continuous campaign and the higher-dimensional diagnostic view.

Open `notebooks/04_cli_four_variable_campaign.ipynb` for a 4D campaign driven through the `bo-forge` CLI command surface.

Open `notebooks/05_mixed_variable_campaign.ipynb` for a mixed-variable v0.4 campaign using `configs/simple_mixed_logei.yaml`.

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

- `examples/simple_2d_maximise_logei_working_log.csv`
- `examples/simple_2d_minimise_qlogei_working_log.csv`
- `examples/simple_3d_maximise_logei_working_log.csv`
- `examples/simple_4d_maximise_logei_working_log.csv`
- `examples/simple_mixed_logei_working_log.csv`
- `examples/*_latest_suggestions.csv`

Generated reports and figure exports should go under ignored paths such as `reports/`.

## 📊 Diagnostics

The plotting helpers produce report-ready black-on-white figures, even when the notebook or IDE uses a dark theme. `plot_progress()` shows observed and best-so-far objective values.

For one- and two-variable campaigns, `plot_diagnostics()` can show direct design-space plots. For campaigns with three or more variables, BO Forge switches to dimension-agnostic diagnostics: objective history, direction-aware best-so-far progress, and a normalised variable-coverage heatmap. These 3D+ diagnostics show coverage and progress, not full interaction structure.

Both functions return figure/axes objects and can optionally save figures:

```python
from bo_forge.diagnostics import plot_diagnostics, plot_progress

plot_progress(config, df, save_path="reports/progress.png")
plot_diagnostics(config, df, save_path="reports/diagnostics.png")
```

`save_path` writes exactly to the requested path. The older `filename`/`fig_folder` arguments construct a path from `fig_folder` and `filename`. Passing both `filename` and `save_path` is invalid.
