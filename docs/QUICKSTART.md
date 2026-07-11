# 🚀 Quickstart And Usage

This page covers setup, the quickstart script, CLI workflow, notebook session API, example notebooks, and diagnostics.

For a compact overview of supported, read-only, rejected, and deferred feature
combinations, see [CAPABILITY_MATRIX.md](CAPABILITY_MATRIX.md).

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

## 🖥️ Streamlit App Quickstart

The local Streamlit app wraps the same `CampaignSession` workflow:

```bash
pip install "bo-forge[app]"
bo-forge-app
```

Module launch is equivalent:

```bash
python -m bo_forge_app
```

For trusted LAN access:

```bash
bo-forge-app --host 0.0.0.0 --port 8501
```

BO Forge has no built-in authentication. For local-only, trusted-LAN, SSH/VPN,
and authenticated reverse-proxy guidance, see
[STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md).

On macOS, you can create a double-click launcher:

```bash
bo-forge-app --make-launcher ~/Desktop/BO-Forge.command
```

For development from a clone, use `./.venv/bin/pip install -e ".[app]"` and `./.venv/bin/bo-forge-app`. The raw Streamlit command remains a development fallback: `./.venv/bin/python -m streamlit run bo_forge_app/streamlit_app.py`.

Suggestions are generated as a dry run and staged in app session state. They are appended to the selected CSV log only after the explicit append button is clicked.

The app can also export staged suggestions to a separate CSV without changing the staged suggestions or the campaign log.

The app uses a Forge Suite-inspired workbench layout with a compact campaign source bar, a `Create Campaign` flow, and stateful `Overview`, `Suggest`, `Resolve`, `Reports`, and `Data` panels. v1.3.4 adds structured-campaign stage display, stage-aware dry-run suggestions, and stage diagnostics in the app while keeping BO logic in `CampaignSession`. Staged suggestions are blocked from append if the selected stage, config, or log changes after staging.

Environment checks remain CLI workflows. Empty-log creation is also available through the CLI when you already have a config:

```bash
./.venv/bin/python -m bo_forge doctor
./.venv/bin/python -m bo_forge init-log --config configs/01_simple_2d_maximise_logei.yaml --log examples/new_campaign_log.csv
```

See [STREAMLIT_APP.md](STREAMLIT_APP.md) for setup details and write-action warnings. See [09_APP_CREATED_CAMPAIGN_TUTORIAL.md](09_APP_CREATED_CAMPAIGN_TUTORIAL.md) for a step-by-step app-created campaign tutorial.

## 🧪 Experimental API Probe

BO Forge includes an optional FastAPI probe around the internal app service
layer:

```bash
pip install "bo-forge[api]"
bo-forge-api --root . --host 127.0.0.1 --port 8765
```

The probe is experimental, root-bound, unauthenticated, and intended only for
local or trusted-network exploration. Streamlit remains the recommended local
UI. See [API_PROBE.md](API_PROBE.md) before using it beyond localhost.

## 🧩 Structured Campaign Configs

v1.3.0 added backend validation for structured campaign logs with named stages.
Each stage lists the variables active in that stage:

```yaml
stages:
  - name: screen
    variables: [precursor_ratio, solvent]
  - name: refine
    variables: [precursor_ratio, annealing_temperature]
```

Structured CSV logs include `stage` immediately after `source`. For each row,
variables active in that row's stage must be filled and valid; inactive variable
cells must be blank. Constraints are evaluated only when every referenced
variable is active for that row's stage.

v1.3.1 added explicit stage-aware suggestions through the session API and CLI:

```python
suggestions = campaign.suggest_next(batch_size=1, stage="screen")
```

```bash
bo-forge init-log \
  --config configs/13_structured_campaign_core.yaml \
  --log examples/13_structured_campaign_core_working_log.csv

bo-forge suggest \
  --config configs/13_structured_campaign_core.yaml \
  --log examples/13_structured_campaign_core_working_log.csv \
  --stage screen
```

Generated structured suggestions populate `stage`, fill only the selected
stage's active variables, and leave inactive variables blank. If a structured
campaign has multiple stages, pass the stage explicitly.

v1.3.2 adds read-only stage inspection:

```python
campaign.stage_summary()
campaign.plot_stage_diagnostics(save_path="reports/stage_diagnostics.png")
```

```bash
bo-forge stage-summary \
  --config configs/13_structured_campaign_core.yaml \
  --log examples/13_structured_campaign_core_campaign_log.csv

bo-forge plot \
  --config configs/13_structured_campaign_core.yaml \
  --log examples/13_structured_campaign_core_campaign_log.csv \
  --kind stage-diagnostics \
  --output reports/stage_diagnostics.png
```

Automatic stage transitions, cost-aware structured campaigns, and Streamlit
structured campaign creation remain deferred.

The structured tutorial files are:

- `configs/14_structured_campaign_tutorial.yaml`;
- `examples/14_structured_campaign_tutorial_campaign_log.csv`;
- `notebooks/14_structured_campaign_tutorial.ipynb`.

## 🧪 Multi-Fidelity qMFKG

v1.4.0 adds a conservative single-objective multi-fidelity workflow with one
continuous fidelity variable. v1.4.1 adds read-only summaries and diagnostics,
and v1.4.2 adds a notebook walkthrough for that workflow:

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

The fidelity variable remains a normal CSV variable column. Lower-fidelity rows
are real objective measurements at that fidelity; the target fidelity is the
value BO Forge ultimately optimizes for. qMFKG trades target-fidelity
information gain against the configured affine fidelity cost evaluated on the
normalized model-space fidelity coordinate. This fidelity cost is not the same
as BO Forge's existing `cost:` budget/ranking feature.

Try the bundled seed log:

```bash
bo-forge validate \
  --config configs/15_multi_fidelity_qmfkg.yaml \
  --log examples/15_multi_fidelity_qmfkg_campaign_log.csv

bo-forge suggest \
  --config configs/15_multi_fidelity_qmfkg.yaml \
  --log examples/15_multi_fidelity_qmfkg_campaign_log.csv \
  --batch-size 1

bo-forge fidelity-summary \
  --config configs/15_multi_fidelity_qmfkg.yaml \
  --log examples/15_multi_fidelity_qmfkg_campaign_log.csv

bo-forge plot \
  --config configs/15_multi_fidelity_qmfkg.yaml \
  --log examples/15_multi_fidelity_qmfkg_campaign_log.csv \
  --kind fidelity-diagnostics \
  --output reports/15_multi_fidelity_diagnostics.png
```

From Python:

```python
campaign = CampaignSession.from_files(
    "configs/15_multi_fidelity_qmfkg.yaml",
    "examples/15_multi_fidelity_qmfkg_campaign_log.csv",
)
campaign.fidelity_summary()
```

The tutorial notebook is:

- `notebooks/15_multi_fidelity_qmfkg_campaign.ipynb`.

In v1.4.x, multi-fidelity is single-objective only and cannot be combined with
`objectives:`, `stages:`, `cost:`, `replicates.enabled: true`, categorical,
integer, or discrete variables, or model-based `batch_size > 1`.

## 🌐 Contextual LogEI/qLogEI

BO Forge includes a conservative contextual BO workflow for single-objective
LogEI/qLogEI campaigns plus read-only context summaries, diagnostics, and a
notebook. Context variables are declared as normal variables and remain normal
CSV columns, but they are fixed at suggestion time rather than optimized:

```yaml
context:
  variables: [feedstock_acidity]
  default_values:
    feedstock_acidity: 0.5
```

Try the bundled contextual example:

```bash
bo-forge validate \
  --config configs/16_contextual_logei.yaml \
  --log examples/16_contextual_logei_campaign_log.csv

bo-forge suggest \
  --config configs/16_contextual_logei.yaml \
  --log examples/16_contextual_logei_campaign_log.csv \
  --context feedstock_acidity=0.25 \
  --batch-size 1

bo-forge context-summary \
  --config configs/16_contextual_logei.yaml \
  --log examples/16_contextual_logei_campaign_log.csv

bo-forge plot \
  --config configs/16_contextual_logei.yaml \
  --log examples/16_contextual_logei_campaign_log.csv \
  --kind context-diagnostics \
  --output /tmp/bo_forge_context_diagnostics.png
```

From Python:

```python
campaign = CampaignSession.from_files(
    "configs/16_contextual_logei.yaml",
    "examples/16_contextual_logei_campaign_log.csv",
)
suggestions = campaign.suggest_next(
    batch_size=1,
    context_values={"feedstock_acidity": 0.25},
)
campaign.context_summary()
```

Contextual campaigns add no CSV columns. The configured context variables use
their existing variable columns, and suggested rows fill those columns with the
fixed context values. The tutorial notebook is
`notebooks/16_contextual_logei_campaign.ipynb`. The Streamlit app can also
create `Campaign kind = Contextual LogEI` configs with selected context
variables and optional defaults. In v1.5.x, `context:` cannot be combined with
`objectives:`, `stages:`, `fidelity:`, `cost:`, or `replicates.enabled: true`.

## 🧠 Model Profiles

The v2.2 line includes optional model profiles for supported single-objective
campaigns configured with `bo.acquisition: log_ei` or `qlog_nei`.
Profiles select curated GP covariance behavior without exposing raw BoTorch
kernel passthrough:

```yaml
model:
  profile: smooth
```

Supported values are `default`, `smooth`, `rough`, and `robust`. Non-default
profiles are intentionally limited to supported single-objective workflows configured
with `bo.acquisition: log_ei` or `qlog_nei` in v2.2.1; multi-objective, multi-fidelity, and
structured campaigns should use the default profile.

Try the bundled model-profile example:

```bash
bo-forge validate \
  --config configs/17_model_profile_logei.yaml \
  --log examples/17_model_profile_campaign_log.csv

bo-forge model-summary \
  --config configs/17_model_profile_logei.yaml \
  --log examples/17_model_profile_campaign_log.csv

bo-forge model-compare \
  --config configs/17_model_profile_logei.yaml \
  --log examples/17_model_profile_campaign_log.csv

bo-forge plot \
  --config configs/17_model_profile_logei.yaml \
  --log examples/17_model_profile_campaign_log.csv \
  --kind model-diagnostics \
  --output /tmp/bo_forge_model_diagnostics.png

bo-forge plot \
  --config configs/17_model_profile_logei.yaml \
  --log examples/17_model_profile_campaign_log.csv \
  --kind model-comparison \
  --output /tmp/bo_forge_model_comparison.png
```

Model comparison is diagnostic only. It does not update the YAML profile,
rewrite the CSV log, or automatically select a model.

The tutorial notebook is:

- `notebooks/17_model_profile_logei_campaign.ipynb`.

For noisy or pending-aware single-objective BO, inspect the qLogNEI seed
campaign:

```bash
python -m bo_forge validate \
  --config configs/18_noisy_pending_qlognei.yaml \
  --log examples/18_noisy_pending_qlognei_campaign_log.csv

python -m bo_forge qlog-nei-summary \
  --config configs/18_noisy_pending_qlognei.yaml \
  --log examples/18_noisy_pending_qlognei_campaign_log.csv

python -m bo_forge suggest \
  --config configs/18_noisy_pending_qlognei.yaml \
  --log examples/18_noisy_pending_qlognei_campaign_log.csv \
  --batch-size 1

python -m bo_forge plot \
  --config configs/18_noisy_pending_qlognei.yaml \
  --log examples/18_noisy_pending_qlognei_campaign_log.csv \
  --kind qlog-nei-diagnostics \
  --output /tmp/bo_forge_qlog_nei_diagnostics.png
```

Accepted review suggestions are treated as active pending experiments for
qLogNEI; review rows still marked `pending` must be resolved first.
The notebook walkthrough is
`notebooks/18_noisy_pending_qlognei_campaign.ipynb`.

## 🔁 Session API

Notebook campaign workflows should usually start with `CampaignSession`:

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
| `campaign.observed_data()` | Return raw observed CSV rows. Replicate-aware model fitting aggregates separately. |
| `campaign.pending_suggestions()` | Return unresolved `status='suggested'` rows. |
| `campaign.campaign_status()` | Return the current campaign status without mutating state or writing to disk. |
| `campaign.next_action()` | Return a one-row advisory DataFrame with campaign status, recommended action, reason, and suggested calls. |
| `campaign.report()` | Return read-only report tables for summary, next action, best raw observation, replicate summaries, suggestions, review queue, and cost summary. |
| `campaign.export_report(path)` | Write a deterministic plain-text campaign report and return the written path. |
| `campaign.review_queue()` | Return suggested rows still waiting for a review decision when review is enabled. |
| `campaign.review_suggestion(row_id, decision, note="")` | Record `accept`, `reject`, or `defer` for one suggested row and refresh `campaign.df`. |
| `campaign.cost_summary()` | Return cost, reserved-cost, budget, and best-objective fields when cost is configured. |
| `campaign.best_observation()` | Return a canonical-column-order copy of the best observed row, or an empty canonical DataFrame. |
| `campaign.replicate_summary()` | Return group-level replicate counts, mean, std, SEM, min, and max when replicates are enabled. |
| `campaign.best_replicate_group()` | Return the best single-objective replicate group by mean objective. Multi-objective replicate campaigns should use `campaign.replicate_summary()` and `campaign.pareto_front()`. |
| `campaign.pareto_front()` | Return nondominated observed rows for a multi-objective campaign. |
| `campaign.pareto_summary()` | Return Pareto-count, reference-point, direction, and hypervolume fields. |
| `campaign.stage_summary()` | Return read-only structured stage counts, active/inactive variables, warnings, and transition-readiness guidance. |
| `campaign.suggest_next(batch_size=None, stage=None, context_values=None)` | Generate suggestions without mutating `campaign.df` or writing to disk. Structured campaigns with multiple stages require `stage`; contextual campaigns use `context_values` when YAML defaults are incomplete or should be overridden. |
| `campaign.suggestion_quality(suggestions)` | Return read-only feasibility, duplicate, and distance diagnostics for suggestion review. |
| `campaign.append_suggestions(suggestions)` | Append suggested rows to the CSV log and refresh `campaign.df`. |
| `campaign.mark_observed(row_id, objective_value, actual_cost=None)` | Mark one pending suggestion observed, optionally record actual cost, and refresh `campaign.df`. |
| `campaign.reload()` | Reload the CSV log from disk into `campaign.df`. |
| `campaign.plot_progress(save_path=None)` | Plot objective and best-so-far progress; returns figure/axes objects. |
| `campaign.plot_cost_progress(save_path=None)` | Plot best observed objective against cumulative effective cost. |
| `campaign.plot_replicates(save_path=None)` | Plot raw replicate observations and replicate-group mean summaries. |
| `campaign.plot_pareto(save_path=None)` | Plot a 2D Pareto scatter for two objectives, or pairwise Pareto projections for 3+ objectives. |
| `campaign.plot_pareto_parallel(save_path=None)` | Plot normalized Pareto-front parallel coordinates for 3+ objective campaigns. |
| `campaign.plot_hypervolume(save_path=None)` | Plot hypervolume progress for a multi-objective campaign. |
| `campaign.plot_diagnostics(save_path=None)` | Plot dimension-aware diagnostics; returns figure/axes objects. |
| `campaign.plot_stage_diagnostics(save_path=None)` | Plot structured stage row counts and active/inactive variable maps. |

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
append_suggestions(log_path, suggestions, config=config)

# After running the suggested experiment:
mark_observed(log_path, row_id=suggestions.loc[0, "row_id"], objective_value=1.95)
```

Prefer `CampaignSession.append_suggestions()` or `append_suggestions(..., config=config)` for production workflows. Passing the config lets BO Forge prevalidate the combined CSV log before writing. The backward-compatible `append_suggestions(log_path, suggestions)` form remains available for non-replicate logs, but replicate and structured logs require config-aware append validation. For structured logs, use `CampaignSession.mark_observed()` or pass `config=config` to low-level `mark_observed()`.

## ⚙️ Example Configs

- `configs/01_simple_2d_maximise_logei.yaml`: maximises photocatalyst-style `activity`.
- `configs/02_simple_2d_minimise_qlogei.yaml`: minimises process `defect_rate`.
- `configs/03_simple_3d_maximise_logei.yaml`: maximises a three-variable synthetic activity.
- `configs/04_simple_4d_maximise_logei.yaml`: maximises a four-variable synthetic activity for CLI demos.
- `configs/05_simple_mixed_logei.yaml`: maximises a mixed continuous/integer/discrete/categorical synthetic yield.
- `configs/06_mixed_constrained_logei.yaml`: maximises a constrained mixed-variable synthetic yield.
- `configs/07_cost_aware_human_review_logei.yaml`: adds deterministic cost estimates, budget tracking, and human-review decisions.
- `configs/08_replicate_aware_logei.yaml`: adds explicit replicate rows, replicate-derived observation variance, and noisy GP fitting.
- `configs/10_multi_objective_mixed_constrained_qlogehvi.yaml`: adds coupled two-objective qLogEHVI with mixed variables and constraints.
- `configs/11_four_objective_mixed_constrained_qlogehvi.yaml`: generalizes qLogEHVI to a four-objective mixed constrained campaign.
- `configs/12_cost_aware_multi_objective_qlogehvi.yaml`: adds deterministic cost-aware batch ranking to a three-objective qLogEHVI campaign.
- `configs/13_structured_campaign_core.yaml`: demonstrates structured stage validation and explicit stage-aware suggestions.
- `configs/14_structured_campaign_tutorial.yaml`: demonstrates a staged screening/refinement workflow with mixed variables, stage-aware constraints, and a notebook walkthrough.
- `configs/15_multi_fidelity_qmfkg.yaml`: demonstrates single-objective continuous-fidelity qMFKG.
- `configs/16_contextual_logei.yaml`: demonstrates single-objective contextual LogEI with a fixed feedstock context.
- `configs/17_model_profile_logei.yaml`: demonstrates single-objective model-profile diagnostics.
- `configs/18_noisy_pending_qlognei.yaml`: demonstrates single-objective qLogNEI with accepted pending review suggestions.

## 🎯 Multi-Objective qLogEHVI Campaigns

BO Forge supports coupled multi-objective campaigns with `m >= 2` objectives. The primary tested range for v2.2.1 is `2 <= m <= 4`; larger objective counts are advanced usage because qLogEHVI, non-dominated partitioning, hypervolume, and visualization become more expensive.

```yaml
objectives:
  - name: yield_score
    direction: maximize
    reference_point: 40.0
  - name: waste_score
    direction: minimize
    reference_point: 25.0
bo:
  acquisition: qlog_ehvi
```

Every observed row must contain all configured objective values. Suggested rows leave every objective column blank until the experiment has been run.

```python
campaign = CampaignSession.from_files(
    "configs/10_multi_objective_mixed_constrained_qlogehvi.yaml",
    "examples/10_multi_objective_mixed_constrained_campaign_log.csv",
)

suggestions = campaign.suggest_next(batch_size=2)
campaign.append_suggestions(suggestions)
campaign.mark_observed(
    row_id=suggestions.loc[0, "row_id"],
    objective_values={"yield_score": 71.2, "waste_score": 13.4},
)
campaign.pareto_front()
campaign.pareto_summary()
campaign.plot_hypervolume(save_path="reports/hypervolume.png")
```

For a 3+ objective campaign, `campaign.plot_pareto()` renders all objective-pair projections using one shared full-space Pareto set, and `campaign.plot_pareto_parallel()` shows normalized Pareto-front trade-off profiles.

The reference point is written in user-facing units and should be meaningfully worse than the region of interest. `hypervolume()` reports current hypervolume for the observed state, using group means when replicates are enabled. `hypervolume_progress()` and `plot_hypervolume()` show cumulative best-so-far progress, so progress plots do not decrease when a later replicate worsens an existing group mean. Hypervolume is reported as `0.0` when no observed point dominates the reference point. v1.4.0 supports review, replicate, and deterministic cost metadata for coupled multi-objective campaigns; decoupled objective evaluation remains deferred.

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

Cost-aware and review-enabled campaigns use optional deterministic `cost` and `review` sections:

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

Initial Sobol/random suggestions fill `cost_estimate` and leave `utility` blank. Single-objective model-based cost-aware exploration suggestions use `source=cost_log_ei` and fill `utility = acquisition - cost.weight * cost_estimate`. Multi-objective model-based cost-aware suggestions use `source=cost_qlog_ehvi`; `acquisition` stores the qLogEHVI batch acquisition value and `utility = acquisition - cost.weight * total_batch_cost` is repeated on every row in the selected batch. If a cost-aware replicate campaign uses the active `uncertain_best` repeat policy, the repeat suggestion fills `cost_estimate` but keeps `source=log_ei` or `qlog_ei` and leaves `utility` blank because it is a repeat decision, not a cost-utility-ranked new exploration candidate.

When `cost_estimate` is filled, validation checks it against the deterministic cost expression. Use `cost_actual` for realised experiment costs that differ from the estimate.

Review-enabled campaigns keep `status` as only `suggested` or `observed`; review decisions live in `review_status`. Pending and accepted suggestions block new suggestions. Rejected and deferred suggestions remain in the CSV for auditability and duplicate avoidance, but do not reserve budget and do not block new suggestions.

`campaign.next_action()` is review-aware: pending review rows point to `campaign.review_queue()` and `campaign.review_suggestion(...)`, while accepted rows point to `campaign.mark_observed(..., actual_cost=...)`.

For review-enabled campaigns, `campaign.summary()` and exported reports include separate `pending_review`, `accepted_pending`, `rejected`, and `deferred` counts.

Budget semantics are:

- observed rows consume `cost_actual` when present, otherwise `cost_estimate`;
- accepted pending suggestions reserve `cost_estimate`;
- pending, rejected, and deferred suggestions do not reserve budget.

The cost and review design is inspired by `/Users/liangze/Desktop/bo_forge/PyTorch & BoTorch/Part 5/tutorial_04_budget_aware_and_human_in_the_loop_bo_workflows_worked.ipynb`, especially acquisition-minus-cost utility, cumulative-cost comparison, and accepted/rejected workflow history.

Replicate-aware campaigns use optional explicit replicate tracking:

```yaml
replicates:
  enabled: true
  suggestion_policy: uncertain_best
  replicate_threshold: 0.10
  min_repeats_at_best: 3
  max_repeats_per_group: 5
  noise_floor: 1.0e-8
```

When replicates are enabled, the CSV log adds `replicate_group` and `replicate_index` after `source` or after review columns. `replicate_index` is zero-based. Rows in the same replicate group must have identical typed user-space design values, and repeated design rows are allowed only when they share one `replicate_group`.

For model fitting, observed replicate rows are aggregated by group mean. When at least one group has 2+ observations, BO Forge passes replicate-derived observation variance into BoTorch as `train_Yvar`: repeated groups use `std^2 / n_replicates`, singleton groups use the weighted pooled replicate variance, and `replicates.noise_floor` prevents zero-noise fixed-noise fits. If no group has repeated observations yet, BO Forge keeps the learned-noise GP behavior.

With the default single-objective `suggestion_policy: uncertain_best`, BO Forge may suggest another observation in the current best replicate group when its posterior standard deviation is above `replicate_threshold` or the group has fewer than `min_repeats_at_best` observations. If a repeat policy produces fewer rows than the requested batch size, BO Forge fills the remaining slots with normal exploration suggestions when budget and design space allow. Set `suggestion_policy: new_only` to disable active repeat suggestions while still using replicate-derived `train_Yvar`. `replicate_threshold` is in objective units and should be tuned to the assay or measurement-noise scale.

Generated exploration suggestions still avoid existing designs, set `replicate_group=row_id`, and set `replicate_index=0`. Policy-driven repeat suggestions reuse the existing `replicate_group` and use the next zero-based `replicate_index`, without adding new CSV columns.

The committed `examples/08_replicate_aware_campaign_log.csv` still has one initial-design slot left, so a first CLI suggestion may be Sobol. To exercise the active `uncertain_best` repeat path from a throwaway repeat-ready log, use a temporary copy with one extra observed design:

```bash
cp examples/08_replicate_aware_campaign_log.csv /tmp/bo_forge_08_repeat_ready.csv
./.venv/bin/python - <<'PY'
import pandas as pd

path = "/tmp/bo_forge_08_repeat_ready.csv"
df = pd.read_csv(path, keep_default_na=False)
df.loc[len(df)] = [
    "rep_seed_3a", 3, "observed", "manual", "rep_3", 0,
    0.85, 430, 1.10, "", "", "",
]
df.to_csv(path, index=False)
PY
./.venv/bin/python -m bo_forge suggest \
  --config configs/08_replicate_aware_logei.yaml \
  --log /tmp/bo_forge_08_repeat_ready.csv \
  --batch-size 3
```

Public CSV logs still store every raw replicate row. `campaign.best_observation()` remains the best raw observed row, while `campaign.best_replicate_group()` returns the best single-objective group by mean objective. For multi-objective replicate campaigns, qLogEHVI uses group means plus per-objective `train_Yvar`, but active repeat selection is deferred; MO replicate configs default to `suggestion_policy: new_only`, and explicit `uncertain_best` fails clearly. Use `campaign.replicate_summary()` and `campaign.pareto_front()` to inspect group-level statistics and group-mean Pareto rows. For single-replicate groups, `objective_std` and `objective_sem` are `NaN`.

Cost and review summaries remain row-level when combined with replicates. Replicate summaries are group-level, so a replicate group may contain multiple rows with their own costs and review states.

The replicate-aware noisy BO design is inspired by `/Users/liangze/Desktop/from-pytorch-to-bayesian-optimisation/part_6/tutorial_01_noisy_and_replication_aware_bo.ipynb`, especially empirical replicate variance, noisy GP fitting, and repeat-vs-explore decisions.

## 📓 Example Notebooks

The example notebooks now run deeper simulated campaigns. Each notebook finishes with 15 completed observed campaign units, including seed data; the replicate-aware notebook finishes with 15 observed replicate groups.

Open `notebooks/01_maximisation_logei_campaign.ipynb` for a simulated end-to-end maximisation campaign using `configs/01_simple_2d_maximise_logei.yaml` and `examples/01_simple_2d_maximise_logei_campaign_log.csv`.

Open `notebooks/02_minimisation_qlogei_campaign.ipynb` for a minimisation campaign using `configs/02_simple_2d_minimise_qlogei.yaml` and `examples/02_simple_2d_minimise_qlogei_campaign_log.csv`. It fills the Sobol initial design, then demonstrates qLogEI batch BO rounds.

Open `notebooks/03_three_variable_campaign.ipynb` for a compact 3D continuous campaign and the higher-dimensional diagnostic view.

Open `notebooks/04_cli_four_variable_campaign.ipynb` for a 4D campaign driven through the `bo-forge` CLI command surface.

Open `notebooks/05_mixed_variable_campaign.ipynb` for a mixed-variable v0.4 campaign using `configs/05_simple_mixed_logei.yaml`.

Open `notebooks/06_constrained_mixed_campaign.ipynb` for a constrained mixed-variable campaign using `configs/06_mixed_constrained_logei.yaml`.

Open `notebooks/07_cost_aware_human_review_campaign.ipynb` for a cost-aware and human-review campaign using `configs/07_cost_aware_human_review_logei.yaml`.

Open `notebooks/08_replicate_aware_campaign.ipynb` for a replicate-aware campaign using `configs/08_replicate_aware_logei.yaml`.

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
- `examples/08_replicate_aware_working_log.csv`
- `examples/17_model_profile_logei_working_log.csv`
- `examples/*_latest_suggestions.csv`

Generated reports and figure exports should go under ignored paths such as `reports/`.

## 📊 Diagnostics

The plotting helpers produce report-ready black-on-white figures, even when the notebook or IDE uses a dark theme. `plot_progress()` shows observed and best-so-far objective values.

For one- and two-variable campaigns, `plot_diagnostics()` can show direct design-space plots. For campaigns with three or more variables, BO Forge switches to dimension-agnostic diagnostics: objective history, direction-aware best-so-far progress, and a normalised variable-coverage heatmap. These 3D+ diagnostics show coverage and progress, not full interaction structure.

Both functions return figure/axes objects and can optionally save figures:

```python
from bo_forge.diagnostics import plot_diagnostics, plot_progress, plot_replicates

plot_progress(config, df, save_path="reports/progress.png")
plot_diagnostics(config, df, save_path="reports/diagnostics.png")
plot_replicates(config, df, save_path="reports/replicates.png")
```

`save_path` writes exactly to the requested path. The older `filename`/`fig_folder` arguments construct a path from `fig_folder` and `filename`. Passing both `filename` and `save_path` is invalid.
