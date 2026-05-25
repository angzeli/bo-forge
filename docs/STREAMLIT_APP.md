# 🖥️ Streamlit App

BO Forge v0.5.0 adds a local Streamlit prototype around the existing `CampaignSession` workflow.

The app is intentionally thin: it loads a YAML config and CSV log from local paths, then calls the same backend/session methods used by notebooks and the CLI.

## 🧰 Install

Install the app extra from the project root:

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
./.venv/bin/python -m streamlit run bo_forge_app/streamlit_app.py
```

Then enter:

- a YAML config path, such as `configs/01_simple_2d_maximise_logei.yaml`;
- a CSV log path, preferably an ignored working log such as `examples/01_simple_2d_maximise_logei_working_log.csv`.

Use a working log rather than editing seed example logs directly.

## 🔁 Workflow

The app follows the same explicit BO Forge rhythm:

1. load campaign files;
2. validate and inspect campaign state;
3. generate suggestions as a dry run;
4. review staged suggestions;
5. append staged suggestions explicitly;
6. run the experiment outside BO Forge;
7. mark the suggested row observed;
8. reload and repeat.

Generated suggestions are staged in Streamlit session state. They are not appended to the CSV log until the append button is clicked.

## ⚠️ Write Actions

Append, review, and mark-observed actions modify the selected CSV log.

Report and plot export actions write files to the selected output path.

The app invalidates staged suggestions if the selected config path, config file, log path, or log file changes after suggestions are generated.

## 📌 Current Limitations

- local file paths only;
- no authentication or multi-user state;
- no database storage;
- no FastAPI or React frontend;
- no new BO models, acquisitions, or CSV schemas beyond the backend package.
