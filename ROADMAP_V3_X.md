# BO Forge v3.x Roadmap

This roadmap is directional, not a release promise. The v3.x train focuses on
assurance, reproducibility, scientific validation, and maintainability around
the local YAML/CSV campaign model rather than primarily expanding features.

Current prepared baseline: `v3.0.1`. It adds release-assurance infrastructure
without publishing a tag, GitHub Release, or package artifact.

## Roadmap So Far

```mermaid
flowchart LR
    v30["v3.0<br/>Architecture + release assurance"] --> v31["v3.1<br/>Durable campaign provenance"] --> v32["v3.2<br/>Predictive diagnostics correctness"] --> v33["v3.3<br/>Scientific workflow validation"] --> v34["v3.4<br/>Interface + maintenance hardening"]

    v300["v3.0.0<br/>Architecture + scientific UX reset"]
    v301["v3.0.1<br/>CI-backed release foundation"]
    v302["v3.0.2<br/>Conditional stabilization"]
    v310["v3.1.0<br/>Provenance schema + identity"]
    v311["v3.1.1<br/>Fail-closed resume"]
    v312["v3.1.2<br/>Migration + lineage"]
    v313["v3.1.3<br/>Provenance acceptance"]
    v320["v3.2.0<br/>Honest fit metrics"]
    v321["v3.2.1<br/>Explicit fit metadata"]
    v322["v3.2.2<br/>Small-data prediction checks"]
    v323["v3.2.3<br/>Synthetic acceptance"]
    v330["v3.3.0<br/>Benchmark harness"]
    v331["v3.3.1<br/>Mixed + noisy routes"]
    v332["v3.3.2<br/>MO + MF routes"]
    v333["v3.3.3<br/>Notebook execution"]
    v334["v3.3.4<br/>Performance evidence"]
    v340["v3.4.0<br/>Read-only CLI JSON"]
    v341["v3.4.1<br/>Dry-run automation contracts"]
    v342["v3.4.2<br/>Opportunistic decomposition"]
    v343["v3.4.3<br/>Typed-package decision"]
    v344["v3.4.4<br/>API boundary disposition"]

    v30 -.-> v300
    v30 -.-> v301
    v30 -.-> v302
    v31 -.-> v310
    v31 -.-> v311
    v31 -.-> v312
    v31 -.-> v313
    v32 -.-> v320
    v32 -.-> v321
    v32 -.-> v322
    v32 -.-> v323
    v33 -.-> v330
    v33 -.-> v331
    v33 -.-> v332
    v33 -.-> v333
    v33 -.-> v334
    v34 -.-> v340
    v34 -.-> v341
    v34 -.-> v342
    v34 -.-> v343
    v34 -.-> v344

    class v30 majorActive
    class v31,v32,v33,v34 majorFuture
    class v300,v301 patchDone
    class v302,v310,v311,v312,v313,v320,v321,v322,v323,v330,v331,v332,v333,v334,v340,v341,v342,v343,v344 patchFuture

    classDef majorDone fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px,color:#111827;
    classDef majorActive fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#111827;
    classDef majorFuture fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#111827;
    classDef patchDone fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#111827;
    classDef patchActive fill:#dcfce7,stroke:#15803d,stroke-width:1.5px,color:#111827;
    classDef patchFuture fill:#f3f4f6,stroke:#6b7280,stroke-width:1.5px,color:#111827;
```

### Patch Plan

| Version | Status | Summary |
| --- | --- | --- |
| `v3.0.0` | implemented | Architecture and scientific-UX reset with compatibility facades |
| `v3.0.1` | prepared | Reproducible environments, required CI, tag gate, and release-process hardening |
| `v3.0.2` | conditional | Evidence-driven post-release packaging or CI corrections only |
| `v3.1.x` | planned | Durable campaign provenance and lineage |
| `v3.2.x` | planned | Predictive diagnostics correctness and explicit fit metadata |
| `v3.3.x` | planned | Closed-loop scientific and executable-workflow validation |
| `v3.4.x` | planned | Structured automation interfaces and maintenance decisions |

## v3.0.x - Architecture And Release Assurance

Status: active

### v3.0.0 - Architecture And Scientific UX Reset

Status: implemented

- Preserve documented top-level APIs, YAML/CSV formats, numerical routing, and
  local workflow semantics.
- Separate configuration, campaign, optimization, diagnostics, application,
  Streamlit, and optional API ownership behind compatibility entrypoints.
- Replace the five-panel workbench with `Campaign`, `Run`, and `Analyze`.
- Standardize scoped scientific figure styling and semantic colors.
- Add architecture, complexity, import-boundary, public-signature, and UI
  behavior-freeze tests.
- Document compatibility facades, navigation mapping, API isolation, and visual
  changes in `docs/MIGRATION_V3.md`.

### v3.0.1 - CI-Backed Release Foundation

Status: prepared; publication requires separate authorization and exact-commit CI

Audit mapping: `REL-001`, `REP-002`, `DOC-001`, `DOC-002`, `DX-001`.

- Generate hashed, fully resolved Python 3.11/3.12 constraints with a pinned
  resolver and explicit Linux/macOS platform targets.
- Require Linux Python 3.11 and 3.12 full-suite CI.
- Add economical macOS 3.12 coverage for canonical/symlink paths, file locks,
  stale fingerprints, atomic replacement, rollback, mode preservation, and
  thread/process mutation paths.
- Run Ruff, syntax/metadata checks, and the complete existing pytest suite.
- Build wheel and sdist through PEP 517 in runner-temporary storage and verify
  both with Twine and package-boundary contracts.
- Install wheel and sdist outside the checkout under the selected constraints,
  run `pip check`, and prove imports resolve from installed artifacts.
- Smoke packaged `bo-forge`, `bo-forge-app`, and `bo-forge-api` entrypoints
  without starting unbounded servers or exposing non-loopback listeners.
- Run representative real BoTorch qMFKG paths in a separate CPU-only bounded
  job rather than in every fast validation job.
- Add a future `v*` tag gate that verifies tag/version identity, builds from the
  exact tagged commit, and retains private workflow artifacts without creating
  a release or publishing to a registry.
- Remove author-home paths from release-facing docs and notebooks and enforce a
  precise regression scan.
- Add `CONTRIBUTING.md` and `SECURITY.md`.
- Make production API/database limitation wording version-neutral.
- Change package maturity metadata to Beta and align authoritative version
  sources at `3.0.1` only after the foundation passes local validation.
- Keep branch/tag protection as a separately configured GitHub repository
  setting; repository workflow files do not prove server-side protection.
- Do not tag, publish, upload public release assets, or write the final release
  announcement as part of this milestone implementation.

Acceptance criteria:

- [ ] All generated constraints pass structural/freshness checks.
- [ ] Required Linux, macOS, package, entrypoint, and numerical CI jobs are
  represented in repository workflows with least-privilege permissions and
  bounded timeouts.
- [ ] Clean Python 3.12 local preflight passes; Python 3.11 remains a required
  CI result rather than an unverified local claim when unavailable.
- [ ] Wheel/sdist metadata, contents, and external installs pass.
- [ ] Release-facing files contain no private author-home paths.
- [ ] Version, maturity classifier, changelog, roadmap, docs, and artifact names
  agree on the prepared release identity.
- [ ] No tag, GitHub Release, package publication, or push occurs.

### v3.0.2 - Conditional Post-Release Stabilization

Status: conditional/reserved

- Exists only for evidence-driven CI platform incompatibilities, packaging
  omissions, constraint corrections, or release-process documentation fixes
  discovered after v3.0.1.
- Must not contain new scientific features, campaign combinations, schemas, or
  interface expansion.
- May be skipped entirely when v3.0.1 evidence remains sound.

## v3.1.x - Durable Campaign Provenance

Status: planned

Audit mapping: `REP-001`.

### v3.1.0 - Provenance Schema And Campaign Identity

- Add a versioned sidecar manifest without replacing YAML/CSV source data.
- Record exact config bytes/hash and normalized semantic config identity.
- Record BO Forge version and commit when available, Python/platform/dependency
  identity, seeds, optimization settings, and creation/mutation timestamps.
- Define ownership and atomic-write behavior before enabling mutation.

### v3.1.1 - Fail-Closed Resume Semantics

- Verify manifests on campaign load and resume.
- Classify byte, semantic, environment, and lineage mismatches clearly.
- Reject unexplained semantic mismatch by default while preserving established
  exception and mutation contracts.

### v3.1.2 - Explicit Migration And Lineage

- Add explicit legacy campaign adoption without fabricating unknown history.
- Preserve parent/child identity and old/new manifests.
- Record migrations and authorized overrides append-only.

### v3.1.3 - Provenance Acceptance And Documentation

- Complete compatibility, mutation, concurrency, and migration tests.
- Document adoption, mismatch recovery, lineage, and limitations.
- Require clean artifact and legacy-campaign acceptance before release.

## v3.2.x - Predictive Diagnostics Correctness

Status: planned

### v3.2.0 - Honest In-Sample Metric Labelling

- Rename or explicitly label training RMSE/MAE as in-sample diagnostics.
- Prevent CLI, app, plot, and documentation wording from implying predictive
  model ranking.
- Preserve compatible facades where practical.

### v3.2.1 - Explicit Fit Result And Metadata Ownership

Audit mapping: `ARC-001`.

- Replace process-global mutable `last_fit_*` metadata with an explicit fit
  result/diagnostics object or equivalent owned state.
- Add interleaved and concurrent-fit tests.
- Preserve suggestion numerics and documented summaries during migration.

### v3.2.2 - Small-Data Predictive Diagnostics

- Add carefully bounded LOO and/or cross-validation where statistically useful.
- Report standardized residuals, predictive log-density where appropriate,
  interval coverage, and calibration behavior.
- State small-sample limits and avoid automatic profile selection claims.

### v3.2.3 - Synthetic Acceptance And Interpretation Contract

Audit mapping: diagnostics portion of `SCI-001`.

- Add calibrated and intentionally miscalibrated synthetic fixtures.
- Document valid and invalid metric interpretations.
- Require evidence before any automatic model-selection recommendation.

## v3.3.x - Closed-Loop Scientific And Workflow Validation

Status: planned

Audit mapping: closed-loop portion of `SCI-001`, plus `TST-001` and `PERF-001`.

### v3.3.0 - Benchmark Harness Foundation

- Add versioned benchmark configuration and fixed repeated seed sets.
- Cover continuous single-objective campaigns against random and Sobol
  baselines.
- Report regret, failures, runtime, and distributions rather than exact
  stochastic trajectories.

### v3.3.1 - Mixed, Constrained, And Noisy Routes

- Extend repeated-seed evidence to mixed variables, constraints, and
  noisy/pending behavior.
- Report feasibility, failures, and runtime without weakening assertions.

### v3.3.2 - Multi-Objective And Multi-Fidelity Routes

- Add hypervolume or hypervolume-regret evidence for multi-objective workflows.
- Add bounded multi-fidelity quality/cost evidence with failure disclosure.
- Keep unsupported combinations unchanged unless separately reviewed.

### v3.3.3 - Full Notebook Execution

- Execute every notebook from a clean source archive in temporary workspaces.
- Enforce bounded timeouts and no tracked-file writes.
- Keep committed notebooks output-free; use a representative PR subset and a
  complete release job if total runtime requires separation.

### v3.3.4 - Stable Performance Evidence

- Add fixed-runner import/startup checks and representative q=1/2/4 workflows.
- Record environment manifests, runtime variance, and memory evidence where
  practical.
- Keep performance regressions distinct from scientific quality claims.

## v3.4.x - Interface And Maintenance Hardening

Status: planned

### v3.4.0 - Versioned CLI JSON For Read-Only Commands

Audit mapping: first half of `UX-001`.

- Add versioned structured output for validate, summary, model
  summary/comparison, and context/fidelity/replicate/stage summaries.
- Separate stdout data from stderr diagnostics.
- Add schema and compatibility contract tests.

### v3.4.1 - Structured Dry-Run And Automation Contracts

Audit mapping: second half of `UX-001`.

- Add structured suggestion dry-run output and stable automation errors where
  appropriate.
- Add golden/schema tests without changing explicit append semantics.

### v3.4.2 - Opportunistic Module Decomposition

Audit mapping: `MNT-001`.

- Split modules only when touched by real work.
- Prioritize CLI command groups, API stage lifecycle/storage, session read/write
  responsibilities, and validation domains.
- Preserve public compatibility and explicitly reject a wholesale rewrite.

### v3.4.3 - Typed-Package Product Decision

Audit mapping: `DX-002`.

- Decide whether typing is a public compatibility promise.
- If yes, add focused type checking and `py.typed` only after the supported
  surface passes; if no, document the boundary honestly.

### v3.4.4 - API Product-Boundary Disposition

Audit mapping: `OPS-001`.

- Keep the FastAPI adapter experimental by default.
- Preserve loopback defaults, trusted-network warnings, process-local stages,
  and non-multi-worker limitations.
- Add negative tests preventing accidental production-security claims.
- Treat a true production backend as a separate architecture milestone,
  probably outside this local-first v3 train, requiring durable transactional
  storage, authenticated identity, authorization, idempotency, rate limiting,
  persistent audit, multi-worker coordination, backup/restore, and deployment
  observability.

## Roadmap-Wide Definition Of Done

The v3.x train is complete only when evidence supports all applicable items:

- [ ] Exact release commit is validated before an exactly matching tag is
  created.
- [ ] Python 3.11 and 3.12 are required CI, with Linux full-suite and macOS core
  filesystem coverage.
- [ ] Wheel and sdist install from outside the source checkout.
- [ ] Durable campaign provenance and fail-closed semantic mismatch handling
  exist with explicit legacy migration.
- [ ] Training fit and predictive validation are clearly separated.
- [ ] Repeated-seed closed-loop baselines cover representative scientific paths.
- [ ] Notebooks execute in temporary workspaces while Git copies stay clean.
- [ ] CLI structured output is versioned and tested.
- [ ] Global mutable fit metadata is removed.
- [ ] API security and deployment limits remain explicit and evidence-aligned.
- [ ] Release-facing files contain no author-home paths.
- [ ] Contributor and security reporting guidance stays current.
- [ ] Maturity and capability claims match the available evidence.

Status vocabulary in this roadmap is deliberate: **implemented** means present
and tested in the repository; **planned** is directional; **conditional** may
be skipped; **deferred** belongs to a later decision; and **out of scope** is
explicitly excluded from the named milestone.
