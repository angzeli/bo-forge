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
  --config configs/01_simple_2d_maximise_logei.yaml \
  --log examples/01_simple_2d_maximise_logei_campaign_log.csv

bo-forge summary \
  --config configs/01_simple_2d_maximise_logei.yaml \
  --log examples/01_simple_2d_maximise_logei_campaign_log.csv

bo-forge suggest \
  --config configs/01_simple_2d_maximise_logei.yaml \
  --log examples/01_simple_2d_maximise_logei_working_log.csv \
  --output examples/01_simple_2d_maximise_logei_latest_suggestions.csv \
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

seed_log_path = Path("examples/01_simple_2d_maximise_logei_campaign_log.csv")
log_path = Path("examples/01_simple_2d_maximise_logei_working_log.csv")
shutil.copyfile(seed_log_path, log_path)

campaign = CampaignSession.from_files(
    config_path="configs/01_simple_2d_maximise_logei.yaml",
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
| `campaign.report()` | Return read-only report tables for summary, next action, best observation, suggestions, review queue, and cost summary. |
| `campaign.export_report(path)` | Write a deterministic plain-text campaign report and return the written path. |
| `campaign.review_queue()` | Return suggested rows still waiting for a review decision when review is enabled. |
| `campaign.review_suggestion(row_id, decision, note="")` | Record `accept`, `reject`, or `defer` for one suggested row and refresh `campaign.df`. |
| `campaign.cost_summary()` | Return cost, reserved-cost, budget, and best-objective fields when cost is configured. |
| `campaign.best_observation()` | Return a canonical-column-order copy of the best observed row, or an empty canonical DataFrame. |
| `campaign.suggest_next(batch_size=None)` | Generate suggestions without mutating `campaign.df` or writing to disk. |
| `campaign.suggestion_quality(suggestions)` | Return read-only feasibility, duplicate, and distance diagnostics for suggestion review. |
| `campaign.append_suggestions(suggestions)` | Append suggested rows to the CSV log and refresh `campaign.df`. |
| `campaign.mark_observed(row_id, objective_value, actual_cost=None)` | Mark one pending suggestion observed, optionally record actual cost, and refresh `campaign.df`. |
| `campaign.reload()` | Reload the CSV log from disk into `campaign.df`. |
| `campaign.plot_progress(save_path=None)` | Plot objective and best-so-far progress; returns figure/axes objects. |
| `campaign.plot_cost_progress(save_path=None)` | Plot best observed objective against cumulative effective cost. |
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

config = CampaignConfig.from_yaml("configs/01_simple_2d_maximise_logei.yaml")
seed_log_path = Path("examples/01_simple_2d_maximise_logei_campaign_log.csv")
log_path = Path("examples/01_simple_2d_maximise_logei_working_log.csv")
shutil.copyfile(seed_log_path, log_path)

df = load_campaign_log(log_path, config)
suggestions = suggest_next(config, df)
append_suggestions(log_path, suggestions)

# After running the suggested experiment:
mark_observed(log_path, row_id=suggestions.loc[0, "row_id"], objective_value=1.95)
```

## ⚙️ Example Configs

- `configs/01_simple_2d_maximise_logei.yaml`: maximises photocatalyst-style `activity`.
- `configs/02_simple_2d_minimise_qlogei.yaml`: minimises process `defect_rate`.
- `configs/03_simple_3d_maximise_logei.yaml`: maximises a three-variable synthetic activity.
- `configs/04_simple_4d_maximise_logei.yaml`: maximises a four-variable synthetic activity for CLI demos.
- `configs/05_simple_mixed_logei.yaml`: maximises a mixed continuous/integer/discrete/categorical synthetic yield.
- `configs/06_mixed_constrained_logei.yaml`: maximises a constrained mixed-variable synthetic yield.
- `configs/07_cost_aware_human_review_logei.yaml`: adds deterministic cost estimates, budget tracking, and human-review decisions.

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
  min_normalized_distance: 0.03
```

`initial_design_method` can be `sobol` or `random`. Model-based mixed suggestions use a latent representation internally, then decode and repair suggestions back to valid user-facing values before returning them.

In v0.4.0, categorical variables used scalar-bin encoding. In v0.4.1, categorical variables use one-hot model features internally while public logs still store exact category labels. This removes artificial ordinal structure between labels, but categorical modelling still uses a standard continuous GP over one-hot features. Dedicated categorical kernels or learned embeddings are future work.

For qLogEI batches with categorical variables, BO Forge uses BoTorch's mixed optimizer over enumerated one-hot category assignments. Batch rows may therefore contain different category labels. The `acquisition` value stored on each returned row is the shared batch-level qLogEI value, not a separate independent score for that row.

v0.4.2 adds optional feasibility constraints:

```yaml
constraints:
  - name: no_water_high_base
    expression: "not (solvent == 'Water' and base_equivalents >= 0.5)"

  - name: water_needs_longer_time
    expression: "solvent != 'Water' or reaction_time >= 35"
```

Constraint expressions are checked twice. Config-time validation checks safe syntax and known variable names when the YAML is loaded. Row-time validation evaluates each expression on normalized user-space row values during CSV validation. Constraints apply to all CSV rows regardless of `status` or `source`.

Allowed constraint syntax is deliberately small: campaign variable names, numeric and string constants, arithmetic, unary `+`/`-`, boolean `and`/`or`/`not`, comparisons, and parentheses. Function calls, attributes, subscripts, comprehensions, imports, and unknown names fail clearly.

`bo.min_normalized_distance` is computed in encoded model space, not raw user units. That encoded space includes continuous dimensions, relaxed integer/discrete dimensions, and one-hot categorical dimensions. Exact duplicates are always rejected; near-duplicates are rejected only when this threshold is greater than `0`.

Use `campaign.suggestion_quality(suggestions)` to inspect feasibility, constraint violations, exact duplicates, nearest distances, and threshold pass/fail before appending suggestions. Constraint handling in v0.4.2 is repair/filter/retry based; it is not a probabilistic constrained acquisition with a learned feasibility model.

The v0.4.2 constraint and diversity design is inspired by `/Users/liangze/Desktop/bo_forge/PyTorch & BoTorch/Part 5/tutorial_03_mixed_variable_and_constrained_bo_worked.ipynb` for repair-aware constrained mixed-variable BO, and `/Users/liangze/Desktop/bo_forge/PyTorch & BoTorch/Part 5/tutorial_02_batch_bo_for_parallel_experimentation_worked.ipynb` for within-batch distance summaries.

v0.4.3 adds optional deterministic cost and review sections:

```yaml
cost:
  expression: "1.0 + 0.04 * reaction_time + 2.0 * (solvent == 'Water')"
  weight: 0.5
  budget: 30.0
  candidate_pool_size: 128
  top_k: 24

review:
  enabled: true
```

Initial Sobol/random suggestions fill `cost_estimate` and leave `utility` blank. Model-based cost-aware suggestions use `source=cost_log_ei` and fill `utility = acquisition - cost.weight * cost_estimate`. For `batch_size > 1`, v0.4.3 uses greedy single-candidate utility rather than joint cost-aware qLogEI.

When `cost_estimate` is filled, validation checks it against the deterministic cost expression. Use `cost_actual` for realised experiment costs that differ from the estimate.

Review-enabled campaigns keep `status` as only `suggested` or `observed`; review decisions live in `review_status`. Pending and accepted suggestions block new suggestions. Rejected and deferred suggestions remain in the CSV for auditability and duplicate avoidance, but do not reserve budget and do not block new suggestions.

`campaign.next_action()` is review-aware: pending review rows point to `campaign.review_queue()` and `campaign.review_suggestion(...)`, while accepted rows point to `campaign.mark_observed(..., actual_cost=...)`.

For review-enabled campaigns, `campaign.summary()` and exported reports include separate `pending_review`, `accepted_pending`, `rejected`, and `deferred` counts.

Budget semantics are:

- observed rows consume `cost_actual` when present, otherwise `cost_estimate`;
- accepted pending suggestions reserve `cost_estimate`;
- pending, rejected, and deferred suggestions do not reserve budget.

The v0.4.3 cost and review design is inspired by `/Users/liangze/Desktop/bo_forge/PyTorch & BoTorch/Part 5/tutorial_04_budget_aware_and_human_in_the_loop_bo_workflows_worked.ipynb`, especially acquisition-minus-cost utility, cumulative-cost comparison, and accepted/rejected workflow history.

## 📓 Example Notebooks

Open `notebooks/01_maximisation_logei_campaign.ipynb` for a simulated end-to-end maximisation campaign using `configs/01_simple_2d_maximise_logei.yaml` and `examples/01_simple_2d_maximise_logei_campaign_log.csv`.

Open `notebooks/02_minimisation_qlogei_campaign.ipynb` for a shorter minimisation campaign using `configs/02_simple_2d_minimise_qlogei.yaml` and `examples/02_simple_2d_minimise_qlogei_campaign_log.csv`. It fills the Sobol initial design, then demonstrates one qLogEI batch BO round.

Open `notebooks/03_three_variable_campaign.ipynb` for a compact 3D continuous campaign and the higher-dimensional diagnostic view.

Open `notebooks/04_cli_four_variable_campaign.ipynb` for a 4D campaign driven through the `bo-forge` CLI command surface.

Open `notebooks/05_mixed_variable_campaign.ipynb` for a mixed-variable v0.4 campaign using `configs/05_simple_mixed_logei.yaml`.

Open `notebooks/06_constrained_mixed_campaign.ipynb` for a constrained mixed-variable campaign using `configs/06_mixed_constrained_logei.yaml`.

Open `notebooks/07_cost_aware_human_review_campaign.ipynb` for a cost-aware and human-review campaign using `configs/07_cost_aware_human_review_logei.yaml`.

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

- `examples/01_simple_2d_maximise_logei_working_log.csv`
- `examples/02_simple_2d_minimise_qlogei_working_log.csv`
- `examples/03_simple_3d_maximise_logei_working_log.csv`
- `examples/04_simple_4d_maximise_logei_working_log.csv`
- `examples/05_simple_mixed_logei_working_log.csv`
- `examples/06_mixed_constrained_logei_working_log.csv`
- `examples/07_cost_aware_human_review_working_log.csv`
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
