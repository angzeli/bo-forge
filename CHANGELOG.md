# 📝 BO Forge Changelog

## v1.2.2 - Python Backend Service Layer

This release adds an internal, non-HTTP Python service layer for the local
Streamlit workbench while preserving backend BO behavior and user-facing app
workflow semantics.

- Adds `bo_forge_app/service.py` with `CampaignAppService` as the Streamlit-facing
  workflow coordinator.
- Keeps BO behavior delegated to `CampaignSession`; notebooks, CLI commands,
  YAML/CSV schemas, launcher behavior, and public `bo_forge` imports are
  unchanged.
- Routes app workflow operations through service methods for load validation,
  panel view data, dry-run suggestions, staged append, review, mark-observed,
  report export, and plot dispatch.
- Preserves the existing staged-suggestion bundle dictionary shape and
  fingerprint semantics.
- Keeps display-only helpers in `streamlit_helpers.py` and does not document the
  service as a stable public API.

## v1.2.1 - Safe Streamlit Deployment Docs

This release adds canonical deployment and safety documentation for using the
existing Streamlit workbench beyond a single local terminal.

- Adds `docs/STREAMLIT_DEPLOYMENT.md` as the canonical guide for local-only,
  trusted-LAN, SSH/VPN, and externally authenticated reverse-proxy modes.
- Documents that BO Forge has no built-in auth, no multi-user state
  coordination, no database-backed campaign store, and is not hardened for
  direct public internet exposure.
- Clarifies that users with app access can interact with host-local files
  selected in the app.
- Recommends dedicated campaign working directories, copied CSV logs, CSV
  backups, and avoiding simultaneous writes.
- Keeps README, Quickstart, Installation, and Streamlit workflow docs concise
  with links back to the deployment guide.
- Updates release-readiness checks to assert the deployment guide exists,
  contains the required safety language, and ships in the source distribution.
- Does not change BO behavior, YAML/CSV schemas, launcher behavior,
  authentication, storage, or Streamlit workflow logic.

## v1.2.0 - App Launcher And LAN Access

This release improves local app startup and trusted-LAN access while preserving v1.1.4 backend, session, CLI, YAML/CSV, and Streamlit workflow behavior.

- Keeps `bo-forge-app` as the primary packaged Streamlit workbench launcher.
- Adds `python -m bo_forge_app` as an equivalent module launch path.
- Adds launcher-owned `--host`, `--port`, `--browser`, and `--no-browser` options.
- Rejects conflicting Streamlit server passthrough options so BO Forge launcher flags remain authoritative.
- Prints local and trusted-LAN access URLs before Streamlit starts.
- Adds explicit network safety warnings for wildcard or non-loopback hosts such as `0.0.0.0`, `::`, LAN IPs, and LAN hostnames: no built-in authentication, trusted LAN/VPN/SSH tunnel only, no public internet exposure, and host-file read/write behavior.
- Adds optional macOS `.command` creation through `bo-forge-app --make-launcher ~/Desktop/BO-Forge.command`.
- Keeps `--help` and launcher creation paths independent of Streamlit imports.
- Updates app, installation, quickstart, release, and repository docs for the new launcher surface.

## v1.1.4 - Streamlit Performance And Coherent UI

This release closes the v1.1.x line with Streamlit workflow completion and app performance polish while preserving v1.1.3 backend/session/CLI behavior.

- Replaces eager workflow tabs with a stateful active-panel selector so inactive panels do not render expensive backend views.
- Reorganizes the app into `Overview`, `Suggest`, `Resolve`, `Reports`, and `Data` panels.
- Replaces the large campaign file area with a compact campaign source bar and form-backed load/create controls.
- Adds Streamlit multi-objective coupled observation entry with one required value per objective and optional actual cost for cost-aware campaigns.
- Adds advanced app campaign creation for 2-4 objective configs with optional review, replicates, and deterministic cost sections.
- Moves raw summaries, next-action tables, full logs, and full data tables into the `Data` panel.
- Makes report text and Matplotlib plots render only after explicit user actions.
- Exposes backend multi-objective Pareto, Pareto-parallel, hypervolume, cost-progress, and replicate plot controls where supported.
- Simplifies Forge Suite CSS to reduce expensive backdrop, mask, and large decorative paint effects.

## v1.1.3 - Cost-Aware Multi-Objective qLogEHVI

This release adds deterministic cost-aware ranking to coupled multi-objective qLogEHVI campaigns while preserving v1.1.2 review and replicate behavior.

- Allows multi-objective configs to combine `objectives:` with `cost:`, `review:`, `replicates:`, or any combination of those sections.
- Adds multi-objective cost CSV columns: `cost_estimate`, `cost_actual`, and `utility`.
- Adds `cost_qlog_ehvi` model-based suggestions with batch-level utility `acquisition - cost.weight * total_batch_cost`.
- Keeps initial Sobol/random multi-objective suggestions source-compatible while filling `cost_estimate` and leaving `utility` blank.
- Extends multi-objective `mark_observed(..., objective_values={...}, actual_cost=...)` and CLI `--actual-cost` support for cost-enabled logs.
- Adds multi-objective cost summaries with current hypervolume, Pareto count, budget, reserved cost, and budget remaining.
- Makes cost-progress plots for multi-objective cost campaigns show cumulative effective cost against best-so-far hypervolume.
- Adds a three-objective cost-aware multi-objective config, seed log, and notebook demonstration.
- Keeps cost deterministic and variable-only; cost is not modeled as another objective and no learned cost surrogate is added.

## v1.1.2 - Multi-Objective Review And Replicate Support

This release adds noisy replicate-aware GP fitting plus review and replicate metadata support to coupled multi-objective qLogEHVI campaigns while keeping multi-objective cost-aware ranking deferred.

- Allows multi-objective configs to use `review.enabled: true`, `replicates.enabled: true`, or both.
- Keeps multi-objective configs with `cost:` rejected.
- Extends `replicates:` with optional noisy-BO controls for repeat policy, uncertainty threshold, repeat bounds, and noise floor.
- Fits replicate-enabled campaigns with group means and replicate-derived `train_Yvar` when empirical replicate variance is available.
- Adds a single-objective `uncertain_best` repeat policy that can intentionally suggest another row in the current best replicate group.
- Extends multi-objective CSV schemas with optional review and replicate metadata columns.
- Makes qLogEHVI suggestions fill review and replicate metadata when configured.
- Uses replicate group-mean objective vectors and per-objective replicate variance for multi-objective model fitting, while Pareto fronts, hypervolume, and hypervolume progress remain group-mean diagnostics.
- Adds per-objective multi-objective replicate summary columns such as `<objective>_mean`, `<objective>_std`, and `<objective>_sem`.
- Extends reports, CLI review/mark-observed/replicate-summary flows, and replicate plots for multi-objective review+replicate campaigns.
- Adds a minimal Streamlit guard so unsupported multi-objective observation entry is not shown as a single-objective form.

## v1.1.1 - Generalized 3+ Objective qLogEHVI Campaigns

This release generalizes the v1.1 multi-objective path from the initial two-objective qLogEHVI workflow to `m >= 2` coupled objectives.

- Accepts two or more configured objectives with finite user-facing reference points.
- Centralizes dynamic multi-objective CSV schema construction for objective and prediction metadata columns.
- Generalizes Pareto-front extraction, deterministic Pareto ordering, hypervolume, and hypervolume progress.
- Adds pairwise Pareto projections for `m >= 3` using one shared full-space Pareto set.
- Adds Pareto parallel-coordinate plotting for campaigns with three or more objectives.
- Adds read-only CLI `pareto-front` and `pareto-summary` commands plus `plot --kind pareto-parallel`.
- Adds a four-objective mixed constrained config, seed log, and notebook demonstration.
- Keeps coupled objective evaluation, shared-input multi-output `SingleTaskGP`, qLogEHVI, and existing mixed-variable repair semantics.
- Defers decoupled objective evaluation, `ModelListGP`, noisy MOBO acquisitions, scalarization, cost/review/replicate combinations for multi-objective campaigns, and Streamlit multi-objective workflow support.

## v1.1.0 - Two-Objective qLogEHVI Campaigns

This release adds BO Forge's first post-1.0 modelling expansion.

- Adds the first coupled two-objective YAML configs with required reference points.
- Adds strict coupled-objective CSV validation and `qlog_ehvi` suggestions.
- Adds Pareto-front extraction, Pareto summaries, and hypervolume progress.
- Extends `CampaignSession`, reports, diagnostics, and CLI `mark-observed` for multi-objective campaigns.
- Adds a mixed constrained two-objective config, seed log, and notebook.
- Defers higher-objective counts, decoupled objective evaluation, `ModelListGP`, cost/review/replicate combinations, and full Streamlit workflow polish.

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
