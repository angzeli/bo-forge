# 🗂️ BO Forge Repository Structure

This repository is organised around one rule: the Bayesian optimisation engine lives in the `bo_forge` Python package, while notebooks, the CLI, and future apps call that package.

## 🏗️ Top-Level Layout

```text
bo-forge/
├── bo_forge/                         # Reusable backend package
├── configs/                          # YAML campaign definitions
├── examples/                         # Seed CSV logs and runnable scripts
├── notebooks/                        # Notebook-first campaign workflows
├── reports/                          # Generated local reports and figures
├── docs/                             # Quickstart, schema, errors, and repository guides
│   ├── QUICKSTART.md
│   ├── CLI.md
│   ├── CLI_ERROR_EXAMPLES.md
│   ├── CSV_SCHEMA.md
│   ├── COMMON_ERRORS.md
│   └── REPOSITORY_STRUCTURE.md
├── tests/                            # Pytest coverage for package behavior
├── README.md                         # Project overview and documentation links
├── ROADMAP.md                        # Completed milestones and planned direction
├── pyproject.toml                    # Package metadata and dependencies
├── requirements-lock.txt             # Tested direct dependency snapshot
├── LICENSE                           # Project license
└── .gitignore                        # Local artifacts excluded from Git
```

The local tutorial directory `PyTorch & BoTorch/` is intentionally ignored. It is reference material, not package source.

`reports/` is for local outputs created by notebooks, such as campaign reports and diagnostic figures. It is not source data.

## 📦 Backend Package

`bo_forge/` contains the reusable campaign engine:

- `config.py`: dataclasses and strict YAML parsing.
- `cli.py`: terminal command wrappers around `CampaignSession`.
- `errors.py`: custom exception types used across the package.
- `logs.py`: CSV loading, `append_suggestions()`, and `mark_observed()`.
- `session.py`: notebook-oriented `CampaignSession` workflow wrapper.
- `validation.py`: schema, bounds, status, source, and objective-state validation.
- `transforms.py`: internal user-space to model-space transforms, including one-hot categorical encoding.
- `models.py`: conversion from campaign logs to tensors and GP fitting.
- `acquisition.py`: LogEI and qLogEI acquisition optimisation.
- `suggestions.py`: Sobol and model-based candidate generation.
- `diagnostics.py`: user-facing diagnostic plots.
- `plot_style.py`: shared matplotlib styling helpers.
- `io.py`: canonical empty-log creation.

Most users should start with the `bo-forge` CLI, `CampaignSession`, or the public functions exported from `bo_forge/__init__.py` rather than importing implementation helpers directly.

## 🚀 How To Use The Repository

Create a local environment and install the package:

```bash
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev]"
```

Run the clean quickstart script:

```bash
./.venv/bin/python examples/quickstart.py
```

Validate a campaign from the terminal:

```bash
bo-forge doctor

bo-forge init-log \
  --config configs/01_simple_2d_maximise_logei.yaml \
  --log examples/my_new_campaign_log.csv

bo-forge validate \
  --config configs/01_simple_2d_maximise_logei.yaml \
  --log examples/01_simple_2d_maximise_logei_campaign_log.csv
```

Open the full simulated workflow:

```bash
./.venv/bin/jupyter notebook notebooks/01_maximisation_logei_campaign.ipynb
```

Run checks before committing:

```bash
./.venv/bin/pytest
./.venv/bin/ruff check .
```

## 📁 Campaign Files

A campaign needs two files:

- A YAML config in `configs/`, such as `configs/simple_2d_maximise_logei.yaml`.
- A canonical CSV log in `examples/` or another working directory.

The repository also includes `configs/simple_2d_minimise_qlogei.yaml` as a small minimisation example, `configs/simple_3d_maximise_logei.yaml` as a three-variable continuous example, `configs/simple_4d_maximise_logei.yaml` as a four-variable CLI workflow example, and `configs/simple_mixed_logei.yaml` as a mixed-variable v0.4 example.

Seed logs in `examples/` should remain small and clean. Example scripts and notebooks copy them to local working logs before making changes, so the committed seed data stays reproducible. Generated reports and diagnostic figures belong in `reports/`.

## 🧭 Development Rules

- Keep variables in original user units in public DataFrames and CSV files.
- Keep transforms internal to `transforms.py`.
- Pass `config` and `df` explicitly; avoid global campaign state.
- Prefer single-purpose functions with clear error messages.
- Add tests when changing validation, log transitions, or suggestion behavior.
- Do not commit `.venv/`, notebook checkpoints, generated working logs, or `PyTorch & BoTorch/`.
- Put generated reports and figure exports under `reports/`.
