# 🖥️ Streamlit App

BO Forge v1.5.2 provides a local Streamlit workbench around the existing `CampaignSession` workflow.

The app is intentionally thin: it loads a YAML config and CSV log from local paths, then calls an internal non-HTTP service layer that delegates BO behavior to the same `CampaignSession` methods used by notebooks and the CLI.

The v1.1.4 UI baseline uses a Forge Suite-inspired workbench style with a compact source bar, stateful panel selector, in-app campaign creation, compact tables, and lazy report/plot rendering.

v1.3.4 adds Streamlit support for existing structured campaign semantics. The
app detects configured stages, shows active/inactive variables, lets users pick
the stage for dry-run suggestions, and stores that stage in the staged
suggestion bundle before explicit append.

v1.4.3 completes the Streamlit-facing single-objective multi-fidelity qMFKG
workflow. The app can create conservative continuous-fidelity qMFKG configs,
load existing fidelity configs, show fidelity summaries, and route fidelity
diagnostic plots through the existing backend/session workflow.

v1.5.2 adds Streamlit creation support for contextual BO. The app can create
single-objective Contextual LogEI configs with selected context variables and
optional defaults. When a loaded config defines `context:`, the `Suggest` panel
renders one input per context variable, passes those values to
`CampaignSession.suggest_next(context_values=...)`, and records the values in
the staged suggestion bundle before explicit append. Contextual campaigns also
show Context Summary tables and expose Context Diagnostics in `Reports`.

The optional FastAPI probe added in v1.2.3 is documented separately in
[API_PROBE.md](API_PROBE.md). It is experimental and does not replace the
Streamlit workbench.

## 🧩 Structured Campaigns

Structured campaigns remain backend-owned through `CampaignSession`; Streamlit
does not implement automatic stage transitions or a second campaign engine.

When a loaded config defines `stages:`, the app:

- shows configured stages and active/inactive variables in the `Suggest` panel;
- requires a stage selection before stage-aware dry-run suggestions;
- stages suggestions with the selected stage recorded in the app bundle;
- blocks append if the config, log, staged suggestions, or selected stage changed
  after staging;
- shows stage summary tables in `Overview` and `Data`;
- exposes the backend stage-diagnostics plot in `Reports`.

Mutations remain explicit. Generating suggestions does not write to the CSV log;
append, review, and observation actions still run through the backend service
layer and `CampaignSession`.

## 🧪 Multi-Fidelity qMFKG Campaigns

Multi-fidelity campaigns remain backend-owned through `CampaignSession`; the
app only builds validated YAML, stages suggestions, and calls the existing
session methods.

When `Create Campaign` uses `Campaign kind = Multi-fidelity qMFKG`, the app:

- keeps the campaign single-objective;
- restricts generated variables to continuous variables;
- lets one continuous variable act as the fidelity variable;
- defaults a variable named `fidelity` as the fidelity variable when present,
  otherwise the last variable;
- defaults target fidelity to the selected variable's upper bound;
- writes a top-level `fidelity:` block;
- sets `bo.acquisition: qmf_kg` and `bo.batch_size: 1`;
- uses responsive qMFKG defaults matching the tutorial example;
- allows optional `review.enabled: true`;
- leaves cost, replicates, structured stages, multi-objective fields,
  discrete/categorical fidelity variables, and batch qMFKG out of scope.

## 🌐 Contextual Campaigns

Contextual campaigns remain backend-owned through `CampaignSession`; the app
does not implement a separate contextual optimizer.

When `Create Campaign` uses `Campaign kind = Contextual LogEI`, the app:

- keeps the campaign single-objective;
- lets one or more configured variables act as context variables;
- writes a top-level `context:` block with `context.variables`;
- writes `context.default_values` only for defaults enabled in the form;
- sets `bo.acquisition: log_ei`;
- leaves multi-objective, structured, multi-fidelity, cost-aware, and
  replicate-aware contextual workflows out of scope.

When a loaded config defines `context:`, the app:

- shows context inputs in the `Suggest` panel;
- uses YAML `context.default_values` as initial widget values when present;
- passes those values to backend dry-run suggestions;
- stages suggestions with the selected context values recorded in the app
  bundle;
- blocks append if the config, log, staged suggestions, selected stage, or
  context values changed after staging;
- shows Context Summary tables in `Overview` and `Data`;
- exposes the backend Context Diagnostics (`context-diagnostics`) plot in
  `Reports`.

v1.5.2 app support is limited to single-objective contextual LogEI/qLogEI
campaigns. Contextual multi-objective BO, contextual structured campaigns,
contextual multi-fidelity, contextual cost-aware, and contextual
replicate-aware workflows remain deferred.

## 🧰 Install

Install the app extra:

```bash
pip install "bo-forge[app]"
```

For development from a clone:

```bash
./.venv/bin/pip install -e ".[app]"
```

For development, the `dev` extra also includes Streamlit:

```bash
./.venv/bin/pip install -e ".[dev]"
```

## ▶️ Run

Start the local app:

```bash
bo-forge-app
```

Module launch is equivalent:

```bash
python -m bo_forge_app
```

Bind to a specific host or port when needed:

```bash
bo-forge-app --host 127.0.0.1 --port 8501
bo-forge-app --no-browser
bo-forge-app --browser
```

For trusted LAN access:

```bash
bo-forge-app --host 0.0.0.0 --port 8501
```

BO Forge has no built-in authentication. Keep deployment details in one place:
read [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md) before sharing the app
beyond one local machine.

On macOS, create an optional double-click launcher:

```bash
bo-forge-app --make-launcher ~/Desktop/BO-Forge.command
```

The launcher records the current working directory so relative campaign paths
remain predictable when double-clicked.

From a source checkout, the raw Streamlit command is available as a development
fallback:

```bash
./.venv/bin/python -m streamlit run bo_forge_app/streamlit_app.py
```

Then use the compact campaign source bar to enter:

- a YAML config path, such as `configs/01_simple_2d_maximise_logei.yaml`;
- a CSV log path, preferably an ignored working log such as `examples/01_simple_2d_maximise_logei_working_log.csv`.

Use a working log rather than editing seed example logs directly.

You can also use `Create Campaign` from the same source bar to build a config from structured fields, inspect or edit the generated YAML, and write both the config and an empty canonical CSV log. Choose `Campaign kind` for single-objective, multi-objective, multi-fidelity qMFKG, or Contextual LogEI creation. Multi-objective creation supports 2-4 coupled objectives plus optional review, replicates, and deterministic cost sections. Multi-fidelity creation is single-objective continuous-fidelity qMFKG only. Contextual creation is single-objective LogEI only. The app validates the YAML before writing files and refuses to overwrite existing config or log paths.

For a full walkthrough, see [09_APP_CREATED_CAMPAIGN_TUTORIAL.md](09_APP_CREATED_CAMPAIGN_TUTORIAL.md).

## 🔁 Workflow

The app follows the same explicit BO Forge rhythm:

1. create or load campaign files;
2. validate and inspect campaign state;
3. generate suggestions as a dry run;
4. review staged suggestions;
5. optionally export staged suggestions to a standalone CSV;
6. append staged suggestions explicitly;
7. run the experiment outside BO Forge;
8. mark the suggested row observed;
9. reload and repeat.

Generated suggestions are staged in Streamlit session state. They are not appended to the CSV log until the append button is clicked.

Exporting staged suggestions writes a separate CSV file only. It does not modify the staged suggestions, append fingerprint, selected campaign log, or loaded session state.

## 🧭 Panels

The app keeps file selection in the source bar, followed by five practical campaign panels. Only the active panel renders on each Streamlit rerun:

- `Overview`: validation, campaign status, next action, compact metrics, best-observation or Pareto summary, and compact cost/replicate/fidelity summaries.
- `Suggest`: dry-run generation, staged suggestions, staged CSV export, suggestion quality, and explicit append.
- `Resolve`: review queue, observable suggestions, single-objective mark-observed, coupled multi-objective objective entry, and actual-cost input when configured.
- `Reports`: report preview/export and supported plot controls. Report text and figures are generated only after explicit actions.
- `Data`: raw summary and next-action tables, observed rows, pending rows, Pareto tables, cost/replicate/fidelity summaries, and the full raw log.

## ⚠️ Write Actions

Append, review, and mark-observed actions modify the selected CSV log.

Report and plot export actions write files to the selected output path.

The app invalidates staged suggestions if the selected config path, config file, log path, or log file changes after suggestions are generated.

## 🧪 CLI Setup Checks

Environment checks remain CLI workflows. Empty-log creation is also still available through the CLI when you already have a config:

```bash
./.venv/bin/python -m bo_forge doctor
./.venv/bin/python -m bo_forge init-log --config configs/01_simple_2d_maximise_logei.yaml --log examples/new_campaign_log.csv
```

## 📌 Current Limitations

- local file paths only;
- no authentication or multi-user state;
- no database storage;
- no production FastAPI backend or React frontend;
- no Streamlit-owned BO models, acquisitions, or CSV schemas beyond the backend package;
- Streamlit cost support surfaces deterministic cost metadata and ranking, not a
  learned cost model;
- multi-objective replicate active-repeat selection remains deferred; multi-objective
  replicate configs use the backend `new_only` policy in v1.4.0;
- no automatic structured-stage transitions or Streamlit-owned structured
  campaign engine;
- no multi-objective, structured, cost-aware, replicate-aware,
  discrete/categorical, or batch multi-fidelity workflows.
- no contextual combinations beyond single-objective LogEI/qLogEI configs.

The v1.2.3 FastAPI probe is experimental, optional, and separate from the
Streamlit workflow. It is not a production backend.
