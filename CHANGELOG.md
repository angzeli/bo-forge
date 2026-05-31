# 📝 BO Forge Changelog

## v1.1.0 - Two-Objective qLogEHVI Campaigns

This release adds BO Forge's first post-1.0 modelling expansion.

- Adds exactly-two-objective YAML configs with required reference points.
- Adds strict coupled-objective CSV validation and `qlog_ehvi` suggestions.
- Adds Pareto-front extraction, Pareto summaries, and hypervolume progress.
- Extends `CampaignSession`, reports, diagnostics, and CLI `mark-observed` for multi-objective campaigns.
- Adds a mixed constrained two-objective config, seed log, and notebook.
- Defers 3+ objectives, decoupled objective evaluation, `ModelListGP`, cost/review/replicate combinations, and full Streamlit workflow polish.

## v1.0.0 - First Stable Public Release

This release hardens BO Forge for a first stable public release.

- Bumps package and documentation references to `1.0.0`.
- Hardens PyPI metadata, package discovery, and release documentation.
- Adds the `bo-forge-app` console command for launching the packaged Streamlit workbench.
- Adds public API and release checklist documentation.
- Keeps BO models, acquisitions, CSV schemas, campaign semantics, notebooks, and CLI behaviour unchanged from the v0.5.4 baseline.

## v0.5.x - Local Streamlit Workbench

The v0.5 line added a local Forge Suite-inspired Streamlit workbench around `CampaignSession`.

- Staged dry-run suggestions before append.
- App coverage for inspect, suggest, append, review, mark observed, reports, and plots.
- Forge Suite UI styling and practical campaign panels.
- In-app campaign creation with safe config/log writing.
- Final workbench polish and screenshots.

## v0.4.x - Practical Single-Objective Campaigns

The v0.4 line made BO Forge useful for realistic single-objective experimental campaigns.

- Mixed continuous, integer, discrete, and categorical variables.
- One-hot categorical encoding for model features.
- Constraints, duplicate handling, and suggestion quality summaries.
- Cost-aware ranking and human-review state.
- Replicate tracking and mean-aggregated model fitting.
- Deeper 15-step example notebooks.

## v0.3.x - CLI Workflow

The v0.3 line added the `bo-forge` command-line workflow.

- Validation, summary, status, next action, suggest, mark observed, report, and plot commands.
- `python -m bo_forge` module invocation.
- CLI error hints.
- `doctor` and `init-log` commands.

## v0.2.x - CampaignSession Notebook Engine

The v0.2 line introduced the higher-level `CampaignSession` workflow.

- One-line session creation from config and log.
- Summary, next action, reports, safe append, safe mark observed, reload, and plotting helpers.
- Figure export paths and higher-dimensional diagnostics.

## v0.1.x - MVP Backend Foundation

The v0.1 line established the CSV-backed BO campaign engine.

- Typed YAML configs and canonical CSV campaign logs.
- Sobol initial suggestions.
- BoTorch `SingleTaskGP` with LogEI/qLogEI.
- Maximise/minimise objective directions.
- Validation, duplicate avoidance, diagnostics, docs, examples, and tests.
