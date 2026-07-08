# 📝 BO Forge Changelog

## v2.1.2 - Model Profile Comparison Diagnostics

This read-only closeout patch adds model-profile comparison diagnostics without
changing model configuration, suggestion behavior, config keys, or CSV schemas.

- Adds `model_profile_comparison(config, df)` and
  `CampaignSession.model_profile_comparison()`.
- Adds `bo-forge model-compare` with repeatable `--profile` filters.
- Adds `plot --kind model-comparison` and Streamlit Reports routing for
  supported single-objective campaigns.
- Compares `default`, `smooth`, `rough`, and `robust` on current observed
  fitting rows and reports model-space RMSE, MAE, predicted uncertainty, fit
  status, fitting-row count, encoded dimension, and train-Y variance use.
- Preserves process-local `last_fit_*` metadata used by `model_summary()`.
- Keeps comparison diagnostic only; BO Forge does not automatically choose or
  change the configured profile.

## v2.1.1 - Model Profile Hardening And Tutorial

This focused follow-up hardens model-profile diagnostics and adds a lightweight
notebook walkthrough without changing model profiles, covariance behavior,
config keys, or CSV schemas.

- Ensures `model_summary()` only shows process-local `last_fit_*` metadata when
  the current fitting inputs match the most recent in-process fit.
- Reports `last_fit_status: not_recorded` and `fallback_status: not_recorded`
  when no matching fit metadata is available.
- Clarifies CLI and public API docs for process-local `last_fit_*` metadata.
- Adds `notebooks/17_model_profile_logei_campaign.ipynb` using the existing
  model-profile config and seed log.
- Keeps non-default model profiles limited to supported single-objective
  LogEI/qLogEI campaigns.

## v2.1.0 - Model Profiles And Diagnostics

This release adds curated single-objective model profiles and read-only model
diagnostics without changing CSV schemas or exposing raw BoTorch kernel
configuration.

- Adds optional `model.profile` config support with `default`, `smooth`,
  `rough`, and `robust` profiles.
- Keeps non-default profiles limited to single-objective LogEI/qLogEI campaigns;
  multi-objective, multi-fidelity, and structured campaigns use the default
  profile in v2.1.0.
- Adds RBF/ARD (`smooth`) and Matern-1.5/ARD (`rough`) covariance profiles for
  supported single-objective campaigns.
- Adds `model_summary(config, df)`, `CampaignSession.model_summary()`, and
  report Model Summary sections.
- Adds `bo-forge model-summary` and
  `bo-forge plot --kind model-diagnostics`.
- Adds Streamlit model-profile creation controls for supported
  single-objective campaigns, plus Model Summary and Model Diagnostics routing
  for loaded campaigns.
- Adds `configs/17_model_profile_logei.yaml` and
  `examples/17_model_profile_campaign_log.csv`.

## v2.0.0 - Stable Interface Hardening Baseline

This release starts the v2.x line as a hardening-only major baseline after
v1.5.3. It does not add BO algorithms, config keys, CSV columns, or new workflow
combinations.

- Preserves v1 YAML/CSV/session/CLI/notebook/Streamlit/service/API probe
  behavior as the compatibility baseline.
- Adds `docs/CAPABILITY_MATRIX.md` to document supported, read-only, rejected,
  and deferred feature combinations.
- Adds `ROADMAP_V2_X.md` as the active roadmap while keeping `ROADMAP_V1_X.md`
  as completed history.
- Fixes the source-distribution manifest license directive and strengthens
  release-readiness checks for manifest directives, wheel/sdist contents, and
  optional app/API extras.
- Keeps contextual multi-objective, structured-contextual,
  multi-fidelity-contextual, cost-aware contextual, replicate-aware contextual,
  database, auth, production API, and SaaS workflows deferred.

## v1.5.3 - Contextual BO Release Closeout

This patch closes the v1.5.x contextual BO line with Streamlit state-safety,
labeling, docs, and release-readiness polish. It does not add contextual
modeling scope, new config keys, new CSV columns, new public APIs, or new CLI
commands.

- Distinguishes app-created YAML defaults (`Default context: ...`) from
  per-batch suggestion values (`Suggestion context: ...`).
- Scopes contextual Streamlit widget state by campaign/config/log identity and
  context schema so campaign switches do not reuse stale values.
- Keeps staged contextual suggestions protected against context, config, log,
  and staged-payload changes before append.
- Adds AppTest coverage for context changes after staging, campaign switches,
  contextual-to-non-contextual loads, and app-created contextual append/observe
  round trips.
- Keeps contextual multi-objective, structured, multi-fidelity, cost-aware, and
  replicate-aware workflows deferred to a later minor release line.

## v1.5.2 - Streamlit Contextual Workflow

This patch adds Streamlit creation support for the existing conservative
contextual BO workflow. It does not change contextual model fitting, YAML/CSV
schema semantics, acquisition behavior, or supported contextual combinations.

- Adds `Campaign kind = Contextual LogEI` to the Streamlit create flow.
- Lets app users choose one or more configured variables as context variables.
- Adds optional typed default-value inputs for selected context variables.
- Generates YAML with `context.variables`, optional `context.default_values`,
  and `bo.acquisition: log_ei`.
- Keeps editable YAML preview, no-overwrite creation, canonical empty CSV logs,
  immediate load-after-create, and staged suggestion fingerprint safety.
- Keeps contextual multi-objective, structured, multi-fidelity, cost-aware, and
  replicate-aware workflows out of scope.

## v1.5.1 - Contextual Reports, Diagnostics, and Notebook

This patch adds the read-only inspection and tutorial layer for BO Forge's
conservative contextual BO workflow. It does not change contextual model fitting,
YAML/CSV schema, acquisition behavior, or supported contextual combinations.

- Adds `context_summary(config, df)` and `CampaignSession.context_summary()`.
- Adds `bo-forge context-summary`.
- Adds `CampaignSession.plot_context_diagnostics()` and
  `bo-forge plot --kind context-diagnostics`.
- Includes contextual summaries in campaign reports and the Streamlit app
  service/view data for loaded contextual campaigns.
- Adds `notebooks/16_contextual_logei_campaign.ipynb`, an output-free tutorial
  using the existing `configs/16_contextual_logei.yaml` and seed CSV log.
- Keeps contextual Streamlit campaign creation, contextual multi-objective,
  structured, multi-fidelity, cost-aware, and replicate-aware workflows out of
  scope.

## v1.5.0 - Contextual BO Core

This release adds BO Forge's conservative contextual BO foundation for
single-objective LogEI/qLogEI campaigns.

- Adds a top-level `context:` config section with context variable names and
  optional `default_values`.
- Keeps context variables as ordinary user-facing CSV variable columns; no new
  CSV columns are introduced.
- Fixes context variables at suggestion time while optimizing over the remaining
  decision variables.
- Adds `CampaignSession.suggest_next(..., context_values={...})` and
  `bo-forge suggest --context NAME=VALUE`.
- Adds contextual fixed-feature handling for continuous, integer, discrete, and
  categorical context variables.
- Adds minimal Streamlit loaded-campaign support for entering context values in
  the `Suggest` panel.
- Adds API/service dry-run support for contextual suggestions and staged-bundle
  context metadata.
- Adds `configs/16_contextual_logei.yaml` and
  `examples/16_contextual_logei_campaign_log.csv`.
- Keeps contextual multi-objective, structured, multi-fidelity, cost-aware, and
  replicate-aware workflows out of scope for v1.5.0.

## v1.4.3 - Streamlit Multi-Fidelity Workflow

This patch completes the Streamlit-facing v1.4 multi-fidelity workflow without
changing backend BO behavior, YAML/CSV semantics, public APIs, or CLI commands.

- Replaces the app creation checkbox model with a clearer `Campaign kind`
  selector for single-objective, multi-objective, and multi-fidelity qMFKG
  campaigns.
- Adds Streamlit-created single-objective continuous-fidelity qMFKG configs with
  a top-level `fidelity:` block, `bo.acquisition: qmf_kg`, forced
  `bo.batch_size: 1`, and responsive tutorial-style qMFKG defaults.
- Defaults the fidelity variable to a variable named `fidelity` when present,
  otherwise the last continuous variable, and defaults target fidelity to that
  variable's upper bound.
- Keeps review optional for app-created multi-fidelity campaigns while leaving
  cost, replicates, structured stages, multi-objective fields, and
  discrete/categorical fidelity workflows out of scope.
- Adds a qMFKG note and batch-size cap in the `Suggest` panel for loaded
  fidelity configs.

## v1.4.2 - Multi-Fidelity Tutorial Workflow

This patch adds a lightweight tutorial notebook for the existing v1.4
single-objective qMFKG workflow.

- Adds `notebooks/15_multi_fidelity_qmfkg_campaign.ipynb`.
- Demonstrates copying the seed log to an ignored working log, validating,
  inspecting `summary()`, `fidelity_summary()`, and `next_action()`, and running
  a sequential qMFKG loop to 15 observed rows.
- Shows explicit append, deterministic local observation simulation,
  `mark_observed()`, report export, and progress, diagnostics, and fidelity
  diagnostics plots.
- Keeps BO modelling, YAML/CSV schemas, Streamlit workflow semantics, CLI
  commands, and public APIs unchanged from v1.4.1.

## v1.4.1 - Multi-Fidelity Reporting and Diagnostics

This patch adds read-only reporting and diagnostics for the v1.4.0
single-objective qMFKG workflow.

- Adds `fidelity_summary(config, df)` and `CampaignSession.fidelity_summary()`.
- Adds `bo-forge fidelity-summary`.
- Adds `plot_fidelity_diagnostics()` and
  `bo-forge plot --kind fidelity-diagnostics`.
- Includes fidelity summaries in campaign reports and Streamlit/app-service
  view data for loaded fidelity campaigns.
- Keeps qMFKG modelling, YAML/CSV schemas, acquisition scope, and unsupported
  multi-fidelity combinations unchanged from v1.4.0.

## v1.4.0 - Single-Objective Multi-Fidelity qMFKG

This release adds BO Forge's first conservative multi-fidelity Bayesian
optimisation workflow.

- Adds a top-level `fidelity:` config section for one continuous fidelity
  variable, a target fidelity, affine fidelity cost settings, and qMFKG fantasy
  count.
- Adds `bo.acquisition: qmf_kg` for single-objective multi-fidelity campaigns.
- Fits BoTorch `SingleTaskMultiFidelityGP` models and optimizes
  `qMultiFidelityKnowledgeGradient` through BoTorch's one-shot KG workflow.
- Uses BoTorch's `AffineFidelityCostModel`,
  `InverseCostWeightedUtility`, `project_to_target_fidelity`,
  target-fidelity `PosteriorMean` current values, and
  `gen_one_shot_kg_initial_conditions`.
- Keeps the CSV schema unchanged: the fidelity variable remains an ordinary
  user-facing variable column and model-based suggestions use `source=qmf_kg`.
- Adds `configs/15_multi_fidelity_qmfkg.yaml` and
  `examples/15_multi_fidelity_qmfkg_campaign_log.csv`.
- Keeps multi-objective, structured, cost-aware, replicate-aware, batch,
  discrete/categorical, and Streamlit-created multi-fidelity workflows out of
  scope for v1.4.0.

## v1.3.4 - Streamlit Structured Campaign Workflow

This release wraps the existing structured-campaign backend/session workflow in
the local Streamlit workbench.

- Detects structured campaigns in Streamlit and displays configured stages with
  active/inactive variables.
- Adds a `Suggest` panel stage selector for stage-aware dry-run suggestions.
- Records the selected stage in the staged suggestion bundle.
- Blocks append if the staged suggestions, selected stage, config, or log changed
  after staging.
- Shows stage summaries in `Overview` and `Data`.
- Exposes the backend stage-diagnostics plot in `Reports`.
- Keeps suggestion generation non-mutating and append/review/observation actions
  explicit.
- Keeps backend BO behavior, YAML/CSV schemas, CLI behavior, notebooks,
  automatic stage transitions, Streamlit campaign creation semantics,
  multi-fidelity campaigns, contextual BO, database storage, cloud deployment,
  and authentication out of scope.

## v1.3.3 - Structured Campaign Example And Notebook

This release adds a lightweight structured-campaign tutorial on top of the
v1.3.2 stage reports and diagnostics.

- Adds `configs/14_structured_campaign_tutorial.yaml` with screening and
  refinement stages, mixed variables, and stage-aware constraints.
- Adds `examples/14_structured_campaign_tutorial_campaign_log.csv` as a small
  validated seed log.
- Adds `notebooks/14_structured_campaign_tutorial.ipynb` as an output-free
  teaching notebook covering load, validate, `stage_summary()`, stage-aware
  dry-run suggestions, explicit append, simulated observations, reports, stage
  diagnostics, and a compact CLI demo.
- Documents the tutorial in the quickstart and repository structure docs.
- Keeps backend modelling, automatic stage transitions, Streamlit structured
  workflow, multi-fidelity campaigns, contextual BO, and broad structured
  workflow changes out of scope.

## v1.3.2 - Stage Reports And Diagnostics

This release adds read-only structured-campaign inspection on top of the
v1.3.1 stage-aware suggestion path.

- Adds `stage_summary(config, df)` and `CampaignSession.stage_summary()` for
  deterministic stage counts, active/inactive variable lists, warnings, and
  transition-readiness guidance.
- Adds structured stage sections to campaign reports without changing
  non-structured report output.
- Adds `CampaignSession.plot_stage_diagnostics()` and
  `bo-forge plot --kind stage-diagnostics` for stage row counts and
  active-variable maps.
- Adds read-only `bo-forge stage-summary` CLI inspection.
- Reports best single-objective rows by stage and Pareto counts for
  multi-objective structured stages where meaningful.
- Keeps automatic stage transitions, new modelling features, notebooks,
  Streamlit structured workflow, multi-fidelity campaigns, contextual BO,
  database storage, and async execution out of scope.

## v1.3.1 - Stage-aware Suggestions And CLI

This release adds explicit stage-aware suggestion generation on top of the
v1.3.0 structured campaign core.

- Adds `stage=` support to `CampaignSession.suggest_next()` and the backend
  `suggest_next()` helper for structured campaigns.
- Adds `bo-forge suggest --stage STAGE_NAME` for dry-run and `--append`
  structured suggestions.
- Populates the canonical `stage` column on generated structured rows.
- Generates only active variables for the selected stage and leaves inactive
  variables blank.
- Requires an explicit stage when a structured campaign has multiple stages,
  while allowing safe inference for a single-stage structured campaign.
- Keeps duplicate checks stage-aware by evaluating candidate quality in the
  selected stage's active-variable space.
- Evaluates only constraints whose referenced variables are active for the
  selected stage.
- Keeps automatic stage transitions, stage reports, notebooks, Streamlit
  structured workflow, multi-fidelity campaigns, contextual BO, database
  storage, and async execution out of scope.

## v1.3.0 - Structured Campaign Core

This release adds the backend foundation for structured campaign logs with
named stages and stage-specific active variables.

- Adds optional `stages:` config support with unique stage names and validated
  variable references.
- Adds a canonical `stage` CSV column for structured campaign logs.
- Validates that every structured row uses a configured stage.
- Requires variables active for a row's stage to be filled and valid.
- Requires inactive variable cells to stay blank so public CSV logs remain
  unambiguous and editable.
- Evaluates constraints for a structured row only when every referenced
  variable is active in that row's stage.
- Rejects `stages:` + `cost:` configs in v1.3.0 because stage-aware cost
  evaluation remains deferred.
- Adds structured-campaign helper exports for stage detection and active
  variable lookup.
- Adds minimal session-summary fields for structured campaign stage metadata.
- Adds a minimal structured config and seed log example.
- Keeps stage-aware suggestion generation, automatic stage transitions,
  cost-aware structured campaigns, multi-fidelity campaigns, contextual BO,
  Streamlit structured workflows, database storage, and async execution out of
  scope.

## v1.2.3 - FastAPI Probe

This release adds an experimental optional FastAPI probe around the internal
`CampaignAppService` for local or trusted-network API exploration.

- Adds `bo_forge_app/api.py` with a `create_app(root=...)` FastAPI factory,
  root-bound path checks, JSON-safe DataFrame payloads, stateless staged-bundle
  conversion, and structured error responses.
- Adds `bo_forge_app/api_cli.py` and the `bo-forge-api` console command.
- Adds the optional `api` extra with FastAPI and Uvicorn dependencies while
  keeping core installs free of API imports.
- Exposes health, validation, summary, dry-run suggestion, staged append,
  review, and mark-observed endpoints.
- Keeps mutations delegated to `CampaignAppService` and existing CSV write
  paths, including staged-bundle fingerprint validation.
- Adds `docs/API_PROBE.md` as the canonical experimental API guide.
- Completes the v1.2 app/access track before the planned v1.3 structured
  campaign line.
- Does not add auth, CORS broadening, a database, report/plot endpoints,
  production deployment architecture, or new BO behavior.

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
