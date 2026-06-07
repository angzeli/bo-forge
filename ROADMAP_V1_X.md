# 🧭 BO Forge Roadmap After v1.0

This roadmap starts after the first stable public release. It is directional, not a release promise. BO Forge should keep the stable YAML/CSV/session/CLI/app foundation while exploring larger workflow and modelling shifts in separate release lines.

## 🧭 Roadmap So Far

```mermaid
flowchart LR
    v10["v1.0<br/>Stable public release"] --> v11["v1.1<br/>Coupled multi-objective qLogEHVI"] --> v12["v1.2<br/>App Launcher + LAN Access"] --> v13["v1.3<br/>Structured campaigns"] --> later["Later<br/>Multi-fidelity + contextual BO"]

    v111["v1.1.1<br/>3+ objective generalisation"]
    v112["v1.1.2<br/>MO review + replicates"]
    v113["v1.1.3<br/>Cost-aware MO qLogEHVI"]
    v114["v1.1.4<br/>Streamlit performance + coherent UI"]
    
    v11 -.-> v111
    v11 -.-> v112
    v11 -.-> v113
    v11 -.-> v114

    class v10,v11,v12 majorDone
    class v13,later majorFuture
    class v110,v111,v112,v113,v114,v120 patchDone

    classDef majorDone fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px,color:#111827;
    classDef majorFuture fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#111827;
    classDef patchDone fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#111827;
```

Current baseline: `v1.2.0`. The v1.1.x line is complete; v1.2 starts with local app launcher, module launch, and trusted-LAN access polish.

### Patch Notes So Far

| Version | Type | Summary |
| --- | --- | --- |
| `v1.0.0` | Stable | First stable public release, packaging, public API, and release docs |
| `v1.1.0` | Major | Coupled two-objective qLogEHVI campaigns, Pareto fronts, and hypervolume progress |
| `v1.1.1` | Minor | Generalized coupled `m >= 2` objective qLogEHVI campaigns and 3+ objective Pareto diagnostics |
| `v1.1.2` | Minor | Review/replicate support for multi-objective qLogEHVI plus noisy replicate-aware GP fitting and single-objective active repeats |
| `v1.1.3` | Minor | Cost-aware multi-objective qLogEHVI with deterministic batch utility, budget filtering, and cost-progress diagnostics |
| `v1.1.4` | Minor | Final v1.1.x Streamlit performance and coherent UI patch covering all v1.1 backend workflows |
| `v1.2.0` | Minor | Testable `bo-forge-app` launcher, `python -m bo_forge_app`, host/port/browser controls, trusted-LAN warnings, and optional macOS `.command` launcher |

## 🧬 v1.1 - Coupled Multi-Objective qLogEHVI Campaigns

Status: completed

- Coupled multi-objective campaigns with `m >= 2` objectives.
- Primary tested range is `2 <= m <= 4`; larger objective counts are advanced usage.
- User-facing objective directions and reference points.
- qLogEHVI suggestions with mixed variables and feasibility constraints.
- Strict dynamic multi-objective CSV schema.
- Pareto-front reporting in user-facing units.
- Pairwise Pareto projections for 3+ objective campaigns using one full-space Pareto set.
- Pareto parallel-coordinate plots for 3+ objective campaigns.
- Hypervolume progress with `0.0` when no point dominates the reference point.
- Session, CLI, report, notebook, and diagnostic plot support.
- Review and replicate metadata for coupled multi-objective campaigns.
- Noisy replicate-aware GP fitting with replicate-derived observation variance.
- Single-objective active repeat suggestions through the `uncertain_best` replicate policy.
- Cost-aware multi-objective qLogEHVI using deterministic batch utility.
- Streamlit workflow completion for v1.1 backend capabilities, including coupled multi-objective observation entry and lazy report/plot rendering.

## 🏗️ v1.2 - App Launcher And Access Path

Status: active

- Testable `bo-forge-app` launcher with explicit host, port, and browser flags.
- `python -m bo_forge_app` module launch.
- Trusted-LAN startup guidance without adding authentication or deployment infrastructure.
- Optional macOS double-click `.command` launcher.
- Safe Streamlit Deployment Docs
- Python Backend Service Layer
- Clearer separation between local app prototype and deployable service.
- FastAPI backend exploration.

## 🧩 v1.3 - Structured Campaigns

Status: planned

- Staged or hierarchical campaign workflows.
- Variables that appear only in specific campaign stages.
- Stage-aware validation, reporting, and diagnostics.

## 🔮 Later

Status: directional

- Multi-fidelity BO.
- Contextual BO.
- More specialised surrogate models or kernels.
- Deeper app collaboration workflows.
