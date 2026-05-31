# 🖥️ Streamlit App

BO Forge v1.1.1 provides a local Streamlit workbench around the existing `CampaignSession` workflow.

The app is intentionally thin: it loads a YAML config and CSV log from local paths, then calls the same backend/session methods used by notebooks and the CLI.

The v1.1.1 UI uses a Forge Suite-inspired workbench style: warm paper tones, compact status chips, rounded panels, practical campaign tabs, in-app campaign creation, compact tables, and clearer empty states.

v1.1.1 keeps the local app on the v1.0 workflow baseline. Generalized coupled multi-objective qLogEHVI support is available through the backend, session API, CLI, and notebooks first; full app-specific multi-objective workflow polish is deferred.

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

From a source checkout, this also works:

```bash
./.venv/bin/python -m streamlit run bo_forge_app/streamlit_app.py
```

Then use the `Campaign Files` panel on the main workbench page to enter:

- a YAML config path, such as `configs/01_simple_2d_maximise_logei.yaml`;
- a CSV log path, preferably an ignored working log such as `examples/01_simple_2d_maximise_logei_working_log.csv`.

Use a working log rather than editing seed example logs directly.

You can also use `Create Campaign` in the same panel to build a basic config from structured fields, inspect or edit the generated YAML, and write both the config and an empty canonical CSV log. The app validates the YAML before writing files and refuses to overwrite existing config or log paths.

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

The app keeps file selection on the main workbench page, followed by four practical campaign panels:

- `Campaign`: validation, campaign status, next action, summary, best observation, observed rows, pending suggestions, and full log preview.
- `Suggest`: dry-run generation, staged suggestions, staged CSV export, suggestion quality, and explicit append.
- `Resolve`: review queue, rows ready to observe, mark-observed form, and actual-cost input when configured.
- `Reports`: report preview/export and progress, diagnostics, cost-progress, or replicate plots when supported by the config.

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
- no new BO models, acquisitions, or CSV schemas beyond the backend package.
