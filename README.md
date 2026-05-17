# 🧪 BO Forge v0.2.2

BO Forge is a notebook-first Bayesian optimisation campaign tool. The notebook is the user workflow, while the reusable BO logic lives in the `bo_forge` Python package.

v0.2 adds a `CampaignSession` notebook engine: define a problem, load a CSV log, validate and summarise campaign state, suggest experiments, enter results, reload from disk, and plot progress through one reusable object. v0.2.2 adds a read-only `next_action()` helper for notebook guidance.

BO Forge deliberately supports only:

- continuous variables
- one objective
- maximize or minimize direction
- Sobol initial suggestions
- BoTorch `SingleTaskGP`
- LogEI for single suggestions and qLogEI for batches
- CSV campaign logs
- resume from existing logs
- basic diagnostics
- a notebook-first `CampaignSession` workflow

It intentionally does not yet cover categorical variables, constraints, noisy BO, multi-objective optimisation, a CLI, or an app UI.

## 🔁 Workflow

```mermaid
flowchart LR
    A["YAML config"] --> B["Load CSV log"]
    B --> C["Validate campaign data"]
    C --> D{"Enough observations?"}
    D -- "No" --> E["Sobol suggestion"]
    D -- "Yes" --> F["Fit SingleTaskGP"]
    F --> G["Score LogEI / qLogEI"]
    G --> H["Suggest candidate(s)"]
    E --> H
    H --> I["Append status=suggested"]
    I --> J["Run experiment"]
    J --> K["mark_observed()"]
    K --> B
```

The app/UI layer is intentionally absent in this MVP. 

Future interfaces should wrap this backend package rather than moving BO logic into notebooks or app code.

## 🗂️ Repository Structure

```text
bo-forge/
├── bo_forge/       # reusable backend package
├── configs/        # YAML campaign configs
├── examples/       # seed logs and runnable examples
├── notebooks/      # notebook-first campaign workflows
├── docs/           # quickstart, schema, troubleshooting, repo guide
└── tests/          # pytest coverage
```

## 📚 Documentation

- [docs/QUICKSTART.md](docs/QUICKSTART.md): setup, quickstart commands, session API example, notebooks, and diagnostics.
- [docs/CSV_SCHEMA.md](docs/CSV_SCHEMA.md): canonical CSV columns, allowed values, blanks, and status transitions.
- [docs/COMMON_ERRORS.md](docs/COMMON_ERRORS.md): troubleshooting guide for common YAML and CSV errors.
- [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md): detailed package layout and development workflow.
- [ROADMAP.md](ROADMAP.md): completed milestones and planned direction.

## 📌 Tested Versions

The primary dependency source is `pyproject.toml`. A direct-dependency snapshot from the v0.2.2 environment is recorded in `requirements-lock.txt`.

## 👤 Author 

Angze Li
