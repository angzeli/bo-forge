# 🗂️ BO Forge Repository Structure

This repository is organised around one rule: the Bayesian optimisation engine lives in the `bo_forge` Python package, while notebooks, the CLI, and the local Streamlit app call that package.

## 🏗️ Top-Level Layout

```text
bo-forge/
├── bo_forge/                         # Reusable backend package
├── bo_forge_app/                     # Local Streamlit wrapper
├── configs/                          # YAML campaign definitions
├── examples/                         # Seed CSV logs and runnable scripts
├── notebooks/                        # Notebook-first campaign workflows
├── reports/                          # Generated local reports and figures
├── docs/                             # Quickstart, schema, errors, and repository guides
│   ├── QUICKSTART.md
│   ├── INSTALLATION.md
│   ├── CLI.md
│   ├── STREAMLIT_APP.md
│   ├── STREAMLIT_DEPLOYMENT.md
│   ├── API_PROBE.md
│   ├── CAPABILITY_MATRIX.md
│   ├── QLOGNEHVI_FEASIBILITY.md
│   ├── 09_APP_CREATED_CAMPAIGN_TUTORIAL.md
│   ├── CLI_ERROR_EXAMPLES.md
│   ├── CSV_SCHEMA.md
│   ├── COMMON_ERRORS.md
│   ├── PUBLIC_API.md
│   ├── RELEASE_CHECKLIST.md
│   └── REPOSITORY_STRUCTURE.md
├── tests/                            # Pytest coverage for package behavior
├── README.md                         # Project overview and documentation links
├── ROADMAP_V0_TO_V1.md               # Milestones through v1.0.0
├── ROADMAP_V1_X.md                   # Completed v1.x roadmap
├── ROADMAP_V2_X.md                   # Active v2.x roadmap
├── CHANGELOG.md                      # Release history
├── MANIFEST.in                       # Source distribution file inclusion rules
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
- `constraints.py`: safe constraint expression validation and row feasibility checks.
- `contextual.py`: context-value resolution, fixed-feature translation, and read-only summaries for contextual suggestions.
- `costs.py`: safe deterministic cost expressions, effective-cost accounting, and budget summaries.
- `cli.py`: terminal command wrappers around `CampaignSession`.
- `errors.py`: custom exception types used across the package.
- `logs.py`: CSV loading, `append_suggestions()`, `review_suggestion()`, and `mark_observed()`.
- `multi_objective.py`: Pareto-front, reference-point, and hypervolume helpers for coupled multi-objective campaigns.
- `multifidelity.py`: BoTorch multi-fidelity helper translations for qMFKG campaigns.
- `replicates.py`: explicit replicate aggregation, replicate-derived observation variance, summaries, and best group selection.
- `structured.py`: read-only structured-campaign stage summaries and transition-readiness guidance.
- `session.py`: notebook-oriented `CampaignSession` workflow wrapper.
- `validation.py`: schema, bounds, status, source, and objective-state validation.
- `transforms.py`: internal user-space to model-space transforms, including one-hot categorical encoding.
- `models.py`: conversion from campaign logs to tensors, model-profile summaries, and GP fitting.
- `acquisition.py`: LogEI, qLogEI, qLogEHVI, and qMFKG acquisition optimisation.
- `suggestions.py`: Sobol, LogEI/qLogEI, qLogEHVI, and qMFKG candidate generation.
- `diagnostics.py`: user-facing diagnostic plots.
- `plot_style.py`: shared matplotlib styling helpers.
- `io.py`: canonical empty-log creation.

Most users should start with the `bo-forge` CLI, `CampaignSession`, or the public functions exported from `bo_forge/__init__.py` rather than importing implementation helpers directly.

`bo_forge_app/` contains the local Streamlit wrapper. `cli.py` resolves the packaged app script for the `bo-forge-app` command, and `__main__.py` supports `python -m bo_forge_app`. The launcher owns host, port, browser, trusted-LAN, and optional macOS `.command` startup concerns. Deployment guidance lives in `docs/STREAMLIT_DEPLOYMENT.md`. The app should call `CampaignSession` and helper functions rather than reimplementing BO logic.

`bo_forge_app/service.py` is an internal, non-HTTP app service layer. It wraps `CampaignSession` for Streamlit-facing workflow operations such as validation, staged suggestions, append, review, mark-observed, reports, and plot routing. It is not a stable public API.

`bo_forge_app/api.py` and `bo_forge_app/api_cli.py` contain an experimental optional FastAPI probe behind the `bo-forge-api` command and `[api]` extra. The probe is root-bound, local/trusted-network only, and not a production backend.

## 🚀 How To Use The Repository

Create a local environment and install the package:

```bash
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev]"
```

For installed use, the stable commands are:

```bash
pip install bo-forge
pip install "bo-forge[app]"
pip install "bo-forge[api]"
bo-forge --version
bo-forge-app
bo-forge-api --help
python -m bo_forge_app --help
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

- A YAML config in `configs/`, such as `configs/01_simple_2d_maximise_logei.yaml`.
- A canonical CSV log in `examples/` or another working directory.

The repository also includes:
- `configs/02_simple_2d_minimise_qlogei.yaml` as a small minimisation example,
- `configs/03_simple_3d_maximise_logei.yaml` as a three-variable continuous example,
- `configs/04_simple_4d_maximise_logei.yaml` as a four-variable CLI workflow example,
- `configs/05_simple_mixed_logei.yaml` as a mixed-variable example,
- `configs/06_mixed_constrained_logei.yaml` as a constrained mixed-variable example,
- `configs/07_cost_aware_human_review_logei.yaml` as a cost-aware human-review example,
- `configs/08_replicate_aware_logei.yaml` as a replicate-aware example,
- `configs/10_multi_objective_mixed_constrained_qlogehvi.yaml` as a coupled two-objective mixed constrained example,
- `configs/11_four_objective_mixed_constrained_qlogehvi.yaml` as a four-objective mixed constrained example,
- `configs/12_cost_aware_multi_objective_qlogehvi.yaml` as a three-objective cost-aware qLogEHVI example,
- `configs/13_structured_campaign_core.yaml` as a structured stage-validation and explicit stage-aware suggestion example,
- `configs/14_structured_campaign_tutorial.yaml` as a staged screening/refinement tutorial paired with `notebooks/14_structured_campaign_tutorial.ipynb`,
- `configs/15_multi_fidelity_qmfkg.yaml` as a single-objective continuous-fidelity qMFKG example paired with `notebooks/15_multi_fidelity_qmfkg_campaign.ipynb`,
- `configs/16_contextual_logei.yaml` as a single-objective contextual LogEI example with `feedstock_acidity` fixed at suggestion time, paired with `notebooks/16_contextual_logei_campaign.ipynb`,
- `configs/17_model_profile_logei.yaml` as a single-objective model-profile example paired with `notebooks/17_model_profile_logei_campaign.ipynb`,
- `configs/18_noisy_pending_qlognei.yaml` as a single-objective qLogNEI example with accepted pending review suggestions, paired with `notebooks/18_noisy_pending_qlognei_campaign.ipynb`,
- `configs/19_multi_objective_qlognehvi.yaml` as a coupled multi-objective qLogNEHVI example with accepted pending review suggestions,
- and `configs/20_contextual_cost_review_logei.yaml` as a contextual cost-review LogEI example paired with `notebooks/20_contextual_cost_review_logei_campaign.ipynb`.

Seed logs in `examples/` should remain small and clean. Example scripts and notebooks copy them to local working logs before making changes, so the committed seed data stays reproducible. Generated reports and diagnostic figures belong in `reports/`.

## 🧭 Development Rules

- Keep variables in original user units in public DataFrames and CSV files.
- Keep transforms internal to `transforms.py`.
- Pass `config` and `df` explicitly; avoid global campaign state.
- Prefer single-purpose functions with clear error messages.
- Add tests when changing validation, log transitions, or suggestion behavior.
- Do not commit `.venv/`, notebook checkpoints, generated working logs, or `PyTorch & BoTorch/`.
- Put generated reports and figure exports under `reports/`.
