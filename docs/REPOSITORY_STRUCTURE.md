# 🗂️ BO Forge Repository Structure

This repository is organised around one rule: the Bayesian optimisation engine lives in the `bo_forge` Python package, while notebooks and future apps call that package.

## 🏗️ Top-Level Layout

```text
bo-forge/
├── bo_forge/                         # Reusable backend package
├── configs/                          # YAML campaign definitions
├── examples/                         # Seed logs and runnable examples
├── notebooks/                        # Notebook-first campaign workflows
├── tests/                            # Pytest coverage for package behavior
├── docs/                             # Quickstart, schema, errors, and repository guides
│   ├── QUICKSTART.md
│   ├── CSV_SCHEMA.md
│   ├── COMMON_ERRORS.md
│   └── REPOSITORY_STRUCTURE.md
├── README.md                         # Project overview and documentation links
├── ROADMAP.md                        # Completed milestones and planned direction
├── pyproject.toml                    # Package metadata and dependencies
└── .gitignore                        # Local artifacts excluded from Git
```

The local tutorial directory `PyTorch & BoTorch/` is intentionally ignored. It is reference material, not package source.

## 📦 Backend Package

`bo_forge/` contains the reusable campaign engine:

- `config.py`: dataclasses and strict YAML parsing.
- `errors.py`: custom exception types used across the package.
- `logs.py`: CSV loading, `append_suggestions()`, and `mark_observed()`.
- `session.py`: notebook-oriented `CampaignSession` workflow wrapper.
- `validation.py`: schema, bounds, status, source, and objective-state validation.
- `transforms.py`: internal user-space to unit-cube transforms.
- `models.py`: conversion from campaign logs to tensors and GP fitting.
- `acquisition.py`: LogEI and qLogEI acquisition optimisation.
- `suggestions.py`: Sobol and model-based candidate generation.
- `diagnostics.py`: user-facing diagnostic plots.
- `plot_style.py`: shared matplotlib styling helpers.
- `io.py`: canonical empty-log creation.

Most users should start with `CampaignSession` or the public functions exported from `bo_forge/__init__.py` rather than importing implementation helpers directly.

## 🚀 How To Use The Repository

Create a local environment and install the package:

```bash
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev]"
```

Run the clean quickstart:

```bash
./.venv/bin/python examples/quickstart.py
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

- A YAML config in `configs/`, such as `configs/simple_2d.yaml`.
- A canonical CSV log in `examples/` or another working directory.

The repository also includes `configs/simple_2d_minimise.yaml` as a small minimisation example.

The seed log in `examples/simple_2d_campaign_log.csv` should remain small and clean. Example scripts and notebooks copy it to ignored working logs before making changes, so the committed seed data stays reproducible.

## 🧭 Development Rules

- Keep variables in original user units in public DataFrames and CSV files.
- Keep transforms internal to `transforms.py`.
- Pass `config` and `df` explicitly; avoid global campaign state.
- Prefer single-purpose functions with clear error messages.
- Add tests when changing validation, log transitions, or suggestion behavior.
- Do not commit `.venv/`, notebook checkpoints, generated working logs, or `PyTorch & BoTorch/`.
