# 🖥️ Streamlit App

BO Forge v1.2.0 provides a local Streamlit workbench around the existing `CampaignSession` workflow.

The app is intentionally thin: it loads a YAML config and CSV log from local paths, then calls the same backend/session methods used by notebooks and the CLI.

The v1.1.4 UI baseline uses a Forge Suite-inspired workbench style with a compact source bar, stateful panel selector, in-app campaign creation, compact tables, and lazy report/plot rendering.

v1.2.0 keeps backend behavior and app workflow logic unchanged while improving launcher access, trusted-LAN startup guidance, and optional macOS click-to-start support.

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

For trusted LAN access, use the primary wildcard bind:

```bash
bo-forge-app --host 0.0.0.0 --port 8501
```

Open `http://<host-machine-lan-ip>:8501` from another trusted device. Wildcard or
non-loopback hosts expose the app to the network and trigger the same warning.
Examples include `0.0.0.0`, `::`, a LAN IP, or a LAN hostname. Network mode has
no built-in authentication, should only be used on a trusted LAN, VPN, or SSH
tunnel, and should not be exposed directly to the public internet. The app reads
and writes files on the host machine, so use a known working directory and keep
CSV log backups.

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

You can also use `Create Campaign` from the same source bar to build a config from structured fields, inspect or edit the generated YAML, and write both the config and an empty canonical CSV log. The default builder creates a single-objective campaign; advanced mode supports 2-4 coupled objectives plus optional review, replicates, and deterministic cost sections. The app validates the YAML before writing files and refuses to overwrite existing config or log paths.

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

- `Overview`: validation, campaign status, next action, compact metrics, best-observation or Pareto summary, and compact cost/replicate summaries.
- `Suggest`: dry-run generation, staged suggestions, staged CSV export, suggestion quality, and explicit append.
- `Resolve`: review queue, observable suggestions, single-objective mark-observed, coupled multi-objective objective entry, and actual-cost input when configured.
- `Reports`: report preview/export and supported plot controls. Report text and figures are generated only after explicit actions.
- `Data`: raw summary and next-action tables, observed rows, pending rows, Pareto tables, cost/replicate summaries, and the full raw log.

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
- no FastAPI or React frontend;
- no new BO models, acquisitions, or CSV schemas beyond the backend package;
- Streamlit cost support surfaces deterministic cost metadata and ranking, not a
  learned cost model;
- multi-objective replicate active-repeat selection remains deferred; multi-objective
  replicate configs use the backend `new_only` policy in v1.2.0.
