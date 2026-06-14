# 🧭 BO Forge Roadmap After v1.0

This roadmap starts after the first stable public release. It is directional, not a release promise. BO Forge should keep the stable YAML/CSV/session/CLI/app foundation while exploring larger workflow and modelling shifts in separate release lines.

## 🧭 Roadmap So Far

```mermaid
flowchart LR
    v10["v1.0<br/>Stable public release"] --> v11["v1.1<br/>Coupled multi-objective qLogEHVI"] --> v12["v1.2<br/>App Launcher + LAN Access"] --> v13["v1.3<br/>Structured campaigns"] --> v14["v1.4<br/>Single-objective multi-fidelity qMFKG"] --> later["Later<br/>Contextual BO + deeper workflow lines"]

    v110["v1.1.0<br/>Two-objective qLogEHVI"]
    v111["v1.1.1<br/>3+ objective generalisation"]
    v112["v1.1.2<br/>MO review + replicates"]
    v113["v1.1.3<br/>Cost-aware MO qLogEHVI"]
    v114["v1.1.4<br/>Streamlit performance + coherent UI"]
    v120["v1.2.0<br/>App launcher + LAN access"]
    v121["v1.2.1<br/>Safe Streamlit deployment docs"]
    v122["v1.2.2<br/>Python backend service layer"]
    v123["v1.2.3<br/>FastAPI probe"]
    v130["v1.3.0<br/>Structured campaign core"]
    v131["v1.3.1<br/>Stage-aware suggestions + CLI"]
    v132["v1.3.2<br/>Stage reports + diagnostics"]
    v133["v1.3.3<br/>Structured tutorial example"]
    v134["v1.3.4<br/>Streamlit structured workflow"]
    v140["v1.4.0<br/>Single-objective qMFKG"]
    v141["v1.4.1<br/>Fidelity reporting + diagnostics"]
    
    v11 -.-> v110
    v11 -.-> v111
    v11 -.-> v112
    v11 -.-> v113
    v11 -.-> v114
    v12 -.-> v120
    v12 -.-> v121
    v12 -.-> v122
    v12 -.-> v123
    v13 -.-> v130
    v13 -.-> v131
    v13 -.-> v132
    v13 -.-> v133
    v13 -.-> v134
    v14 -.-> v140
    v14 -.-> v141

    class v10,v11,v12,v13 majorDone
    class v14 majorActive
    class later majorFuture
    class v110,v111,v112,v113,v114,v120,v121,v122,v123,v130,v131,v132,v133,v134,v140,v141 patchDone

    classDef majorDone fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px,color:#111827;
    classDef majorActive fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#111827;
    classDef majorFuture fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#111827;
    classDef patchDone fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#111827;
```

Current baseline: `v1.4.1`. The v1.3.x line is complete; v1.4 begins with a conservative single-objective continuous-fidelity qMFKG workflow using BoTorch's current multi-fidelity KG stack and now includes read-only fidelity summaries and diagnostics while keeping multi-objective, structured, cost-aware, replicate-aware, and batch multi-fidelity workflows deferred.

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
| `v1.2.1` | Patch | Safe Streamlit deployment docs covering local-only, trusted-LAN, SSH/VPN, and externally authenticated reverse-proxy modes |
| `v1.2.2` | Patch | Internal non-HTTP Python service layer for Streamlit-facing campaign workflows |
| `v1.2.3` | Patch | Experimental optional FastAPI probe around `CampaignAppService` for local or trusted-network API exploration |
| `v1.3.0` | Minor | Structured campaign core with stage config, canonical `stage` CSV column, and stage-aware log validation |
| `v1.3.1` | Minor | Explicit stage-aware backend/session/CLI suggestions for structured campaigns |
| `v1.3.2` | Minor | Read-only stage summaries, report sections, CLI inspection, and stage diagnostics for structured campaigns |
| `v1.3.3` | Patch | Structured campaign tutorial config, seed log, and notebook for staged screening/refinement workflows |
| `v1.3.4` | Patch | Streamlit structured campaign workflow wrapper with stage selector, stage summaries, and stage diagnostics |
| `v1.4.0` | Minor | Single-objective continuous-fidelity qMFKG with unchanged CSV schema |
| `v1.4.1` | Patch | Read-only fidelity summaries, CLI inspection, reports, Streamlit routing, and fidelity diagnostics |

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

Status: completed

- Testable `bo-forge-app` launcher with explicit host, port, and browser flags.
- `python -m bo_forge_app` module launch.
- Trusted-LAN startup guidance without adding authentication or deployment infrastructure.
- Optional macOS double-click `.command` launcher.
- Safe Streamlit deployment docs.
- Python backend service layer for local app workflows.
- Experimental optional FastAPI probe around the app service layer.
- Clearer separation between local app prototype and deployable service.
- Production auth, database, multi-user app state, and deeper collaboration workflows remain outside v1.2.

## 🧩 v1.3 - Structured Campaigns

Status: completed

- Optional `stages:` config block with named stages.
- Variables that appear only in specific campaign stages.
- Canonical `stage` CSV column for structured campaign logs.
- Stage-aware row validation with inactive variables required blank.
- Minimal session-summary reporting for configured stages.
- Explicit stage-aware backend/session/CLI suggestions for a selected stage.
- Generated structured suggestions populate `stage`, fill only active variables, and keep inactive variables blank.
- Read-only stage summaries, structured report sections, CLI stage inspection, and stage diagnostics.
- Structured screening/refinement tutorial config, seed log, and output-free notebook.
- Streamlit stage display, stage-aware dry-run suggestions, stage summaries, and stage diagnostics.
- Automatic stage transitions, cost-aware structured campaigns, Streamlit structured campaign creation, and multi-fidelity semantics remain deferred.

## 🧪 v1.4 - Single-Objective Multi-Fidelity qMFKG

Status: active

- Optional `fidelity:` config section for one continuous fidelity variable.
- `bo.acquisition: qmf_kg` for single-objective multi-fidelity campaigns.
- BoTorch `SingleTaskMultiFidelityGP` fitting.
- BoTorch `qMultiFidelityKnowledgeGradient` with one-shot KG initial
  conditions, affine fidelity cost, inverse-cost utility, target-fidelity
  projection, and target-fidelity posterior-mean current value.
- No new CSV columns; the fidelity variable remains a normal user-facing
  variable column.
- Initial Sobol/random suggestions remain unchanged; model-based suggestions
  use `source=qmf_kg`.
- Read-only `fidelity_summary`, `bo-forge fidelity-summary`, campaign report
  sections, Streamlit routing, and fidelity diagnostics are available.
- Multi-objective, structured, cost-aware, replicate-aware, batch, and
  discrete/categorical multi-fidelity workflows remain deferred.

## 🔮 Later

Status: directional

- Contextual BO.
- More specialised surrogate models or kernels.
- Deeper app collaboration workflows.
