# 🧭 BO Forge Roadmap After v1.0

This roadmap starts after the first stable public release. It is directional, not a release promise. BO Forge should keep the stable YAML/CSV/session/CLI/app foundation while exploring larger workflow and modelling shifts in separate release lines.

## 🧭 Roadmap So Far

```mermaid
flowchart LR
    v10["v1.0<br/>Stable public release"] --> v11["v1.1<br/>Coupled multi-objective qLogEHVI"] --> v12["v1.2<br/>Production app path"] --> v13["v1.3<br/>Structured campaigns"] --> later["Later<br/>Multi-fidelity + contextual BO"]

    v110["v1.1.0<br/>Two-objective qLogEHVI"]
    v111["v1.1.1<br/>3+ objective generalization"]

    v11 -.-> v110
    v11 -.-> v111

    class v10,v11 majorDone
    class v12 majorNext
    class v13,later majorFuture
    class v110,v111 patchDone

    classDef majorDone fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px,color:#111827;
    classDef majorNext fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#111827;
    classDef majorFuture fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#111827;
    classDef patchDone fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#111827;
```

Current baseline: `v1.1.1`. The next planned milestone is `v1.2`, focused on separating the local Streamlit prototype from a more production-ready app path.

### Patch Notes So Far

| Version | Type | Summary |
| --- | --- | --- |
| `v1.0.0` | Stable | First stable public release, packaging, public API, and release docs |
| `v1.1.0` | Major | Coupled two-objective qLogEHVI campaigns, Pareto fronts, and hypervolume progress |
| `v1.1.1` | Minor | Generalized coupled `m >= 2` objective qLogEHVI campaigns and 3+ objective Pareto diagnostics |

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

Deferred beyond `v1.1.1`:

- Missing/asynchronous objective values.
- `ModelListGP`.
- Cost, review, and replicate combinations for multi-objective campaigns.
- Full Streamlit multi-objective workflow polish.

## 🏗️ v1.2 - Production App Path

Status: planned

- Clearer separation between local app prototype and deployable service.
- FastAPI or equivalent backend exploration.
- Persistent campaign storage beyond local CSV files.
- Auth and multi-user design only if the deployment path requires it.

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
