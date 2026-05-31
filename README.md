# 🧪 BO Forge v1.1.0

BO Forge is a practical Bayesian optimisation campaign tool with notebook, CLI, and local Streamlit workflows. The reusable BO logic lives in the `bo_forge` Python package, while notebooks, the CLI, and the app wrap that package.

v1.1.0 builds on the first stable public release with coupled two-objective qLogEHVI campaigns, Pareto-front reporting, and hypervolume progress while keeping the same YAML/CSV/session/CLI foundation.

BO Forge deliberately supports only:

- continuous, integer, discrete, and categorical variables
- single-objective campaigns, plus coupled two-objective campaigns
- maximize or minimize direction
- Sobol or random initial suggestions
- BoTorch `SingleTaskGP`
- LogEI/qLogEI for single-objective campaigns and qLogEHVI for two-objective campaigns
- CSV campaign logs
- optional feasibility constraints
- optional cost-aware ranking and human review
- optional replicate tracking and replicate-aware aggregation
- resume from existing logs
- basic diagnostics, Pareto-front plots, and hypervolume progress
- a notebook-first `CampaignSession` workflow
- a small `bo-forge` CLI workflow
- a local Streamlit workbench

It intentionally does not yet cover qNEI, learned noise models, decoupled or asynchronous multi-objective evaluation, 3+ objectives, database-backed storage, or a production multi-user web backend.

---

## 🧰 Install

Install the backend package and CLI:

```bash
pip install bo-forge
```

Install the local Streamlit workbench:

```bash
pip install "bo-forge[app]"
```

For local development from a clone:

```bash
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev]"
```

Check the installed version and environment:

```bash
bo-forge --version
bo-forge doctor
```

Launch the packaged local app:

```bash
bo-forge-app
```

The module entrypoint is also supported:

```bash
python -m bo_forge --version
```

---

## 🔁 Workflow

```mermaid
flowchart LR
    A["YAML config"] --> B["Load CSV log"]
    B --> C["Validate campaign data"]
    C --> D{"Enough observations?"}
    D -- "No" --> E["Sobol/random suggestion"]
    D -- "Yes" --> F["Fit SingleTaskGP"]
    F --> G["Score LogEI / qLogEI"]
    G --> H["Suggest candidate(s)"]
    E --> H
    H --> I["Append status=suggested"]
    I --> J["Run experiment"]
    J --> K["mark_observed()"]
    K --> B
```

The Streamlit app is intentionally a thin wrapper.

Future interfaces should keep wrapping this backend package rather than moving BO logic into notebooks, CLI commands, or app code.

---

## 🗂️ Repository Structure

```text
bo-forge/
├── bo_forge/       # reusable backend package
├── bo_forge_app/   # local Streamlit wrapper
├── configs/        # YAML campaign configs
├── examples/       # seed CSV logs and runnable scripts
├── notebooks/      # notebook-first campaign workflows with 15-step demos
├── reports/        # generated local reports and figures
├── docs/           # quickstart, CLI, schema, troubleshooting, repo guide
└── tests/          # pytest coverage
```
---

## 📚 Documentation

- [docs/QUICKSTART.md](https://github.com/angzeli/bo-forge/blob/main/docs/QUICKSTART.md): setup, quickstart commands, session API example, notebooks, and diagnostics.
- [docs/INSTALLATION.md](https://github.com/angzeli/bo-forge/blob/main/docs/INSTALLATION.md): pip install tutorial for core, app, development, wheel, and sdist installs.
- [docs/CLI.md](https://github.com/angzeli/bo-forge/blob/main/docs/CLI.md): terminal workflow and command reference.
- [docs/STREAMLIT_APP.md](https://github.com/angzeli/bo-forge/blob/main/docs/STREAMLIT_APP.md): local Streamlit app setup and workflow.
- [docs/09_APP_CREATED_CAMPAIGN_TUTORIAL.md](https://github.com/angzeli/bo-forge/blob/main/docs/09_APP_CREATED_CAMPAIGN_TUTORIAL.md): step-by-step tutorial for creating a new campaign inside the app.
- [docs/CLI_ERROR_EXAMPLES.md](https://github.com/angzeli/bo-forge/blob/main/docs/CLI_ERROR_EXAMPLES.md): intentional CLI failures with expected error and hint output.
- [docs/CSV_SCHEMA.md](https://github.com/angzeli/bo-forge/blob/main/docs/CSV_SCHEMA.md): canonical CSV columns, allowed values, blanks, and status transitions.
- [docs/COMMON_ERRORS.md](https://github.com/angzeli/bo-forge/blob/main/docs/COMMON_ERRORS.md): troubleshooting guide for common YAML and CSV errors.
- [docs/PUBLIC_API.md](https://github.com/angzeli/bo-forge/blob/main/docs/PUBLIC_API.md): stable public imports supported by the `bo_forge` package.
- [docs/RELEASE_CHECKLIST.md](https://github.com/angzeli/bo-forge/blob/main/docs/RELEASE_CHECKLIST.md): GitHub and PyPI release checklist.
- [docs/REPOSITORY_STRUCTURE.md](https://github.com/angzeli/bo-forge/blob/main/docs/REPOSITORY_STRUCTURE.md): detailed package layout and development workflow.
- [CHANGELOG.md](https://github.com/angzeli/bo-forge/blob/main/CHANGELOG.md): release history.
- [ROADMAP_PRE_V1.md](https://github.com/angzeli/bo-forge/blob/main/ROADMAP_PRE_V1.md): completed milestones through v1.0.0.
- [ROADMAP_AFTER_V1.md](https://github.com/angzeli/bo-forge/blob/main/ROADMAP_AFTER_V1.md): post-1.0 direction.

---

## 📌 Tested Versions

The primary dependency source is `pyproject.toml`.

A direct-dependency snapshot from the v1.1.0 environment is recorded in `requirements-lock.txt`.

---

## 👤 Author

Angze Li
