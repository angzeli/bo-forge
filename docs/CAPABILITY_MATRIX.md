# BO Forge Capability Matrix

BO Forge v2.2.1 keeps the v1 YAML, CSV, session, CLI, Streamlit, service, and
experimental API workflows stable while making supported and intentionally
deferred combinations explicit. v2.2 starts the noisy and pending-aware BO line
with conservative single-objective qLogNEI support.

Legend:

- `supported`: implemented for normal use.
- `read-only/reporting only`: inspection, summaries, reports, or plots exist,
  but suggestion generation for the combination is not implemented.
- `rejected`: config or workflow validation fails clearly.
- `deferred`: intentionally not part of the current supported surface.

## Core Campaign Capabilities

| Capability | Status | Notes |
| --- | --- | --- |
| Single-objective LogEI/qLogEI | supported | Standard BO Forge campaign path. |
| Single-objective qLogNEI | supported | Noisy expected improvement for single-objective, non-structured, non-fidelity, non-contextual, non-cost campaigns; accepted review rows become `X_pending`. |
| Single-objective model profiles | supported | `default`, `smooth`, `rough`, and `robust` profiles for configs with `bo.acquisition: log_ei` or `qlog_nei`. |
| Coupled multi-objective qLogEHVI | supported | Primary tested range is `2 <= m <= 4`. |
| Review metadata | supported | Works for single-objective and coupled multi-objective campaigns. |
| Replicate metadata | supported | Includes group-mean summaries and replicate-derived `train_Yvar` where available. |
| Single-objective active replicate repeats | supported | `uncertain_best` is single-objective only. |
| Multi-objective active replicate repeats | rejected | Multi-objective replicate configs default to `new_only`; active repeats are deferred. |
| Deterministic `cost:` ranking | supported | Single-objective and coupled multi-objective deterministic cost workflows are supported. |
| Structured stages | supported | Explicit stage selection, validation, summaries, diagnostics, CLI, and Streamlit workflow. |
| Single-objective multi-fidelity qMFKG | supported | One continuous fidelity variable, `batch_size=1`, no new CSV columns. |
| Single-objective contextual LogEI/qLogEI | supported | Context variables are fixed at suggestion time and remain normal CSV variables. |

## Combination Matrix

| Combination | Status | Notes |
| --- | --- | --- |
| Multi-objective + review | supported | Coupled objectives are observed together. |
| Multi-objective + replicates | supported | Pareto/hypervolume use replicate group means. |
| Multi-objective + deterministic cost | supported | Uses deterministic cost-aware qLogEHVI batch utility. |
| Multi-objective + review + replicates + cost | supported | Backend/session/CLI support through v1.1.x semantics. |
| Structured + review | supported | Stage-aware rows and review metadata can coexist. |
| Structured + replicates | supported | Stage summaries use replicate group means where needed. |
| Structured + cost | deferred | Cost-aware structured workflows are not implemented in v2.2.1. |
| Structured + contextual | deferred | No contextual structured-stage suggestion path yet. |
| Structured + multi-fidelity | deferred | No staged qMFKG or fidelity-by-stage workflow yet. |
| Contextual + review | deferred | Planned as a controlled v2.x combination. |
| Contextual + deterministic cost | deferred | Contextual cost-aware ranking is not implemented. |
| Contextual + replicates | deferred | Contextual replicate-aware BO is not implemented. |
| Contextual + multi-objective | deferred | No contextual qLogEHVI path yet. |
| Contextual + multi-fidelity | deferred | No contextual qMFKG path yet. |
| qLogNEI + deterministic cost | rejected | Cost-aware qLogNEI ranking is deferred. |
| qLogNEI + contextual | rejected | Contextual qLogNEI is deferred. |
| qLogNEI + structured stages | rejected | Stage-aware qLogNEI is deferred. |
| qLogNEI + multi-fidelity | rejected | No noisy qMFKG path in v2.2.1. |
| qLogNEI + multi-objective | rejected | qLogNEHVI is deferred. |
| qLogNEI + replicate active repeats | rejected | Use `replicates.suggestion_policy: new_only` for replicate-derived variance. |
| Multi-fidelity + review | supported | Conservative single-objective qMFKG can use review metadata. |
| Multi-fidelity + cost | rejected | qMFKG fidelity cost is separate from BO Forge `cost:`. |
| Multi-fidelity + replicates | rejected | Replicate-aware qMFKG is not implemented. |
| Multi-fidelity + multi-objective | rejected | qMFKG support is single-objective only. |
| Model-profile comparison diagnostics | supported | Read-only comparison of `default`, `smooth`, `rough`, and `robust` on current single-objective fitting rows; not automatic model selection. |
| Non-default model profile + multi-objective | rejected | Non-default profiles require supported single-objective configs with `bo.acquisition: log_ei` or `qlog_nei` in v2.2.1. |
| Non-default model profile + multi-fidelity | rejected | qMFKG keeps its existing multi-fidelity GP path. |
| Non-default model profile + structured stages | rejected | Stage-specific model-profile support is deferred. |

## Interface Coverage

| Interface | Status | Notes |
| --- | --- | --- |
| `CampaignSession` | supported | Primary Python workflow wrapper. |
| `bo-forge` CLI | supported | Wraps backend/session behavior; mutating actions remain explicit. |
| Example notebooks | supported | Teaching wrappers; committed notebooks stay output-free. |
| Streamlit workbench | supported | Local wrapper around `CampaignSession` and app service. |
| Internal app service layer | supported internally | Not documented as stable public API. |
| Experimental FastAPI probe | supported experimentally | Optional, local/trusted-network only, no built-in auth. |
| Production API/auth/database | deferred | No production SaaS backend, auth system, or database store in v2.x. |

## Safety Boundaries

- CSV logs remain the source of truth.
- Suggestion generation remains non-mutating unless append is explicit.
- Mutating session, CLI, app, and API actions validate before writing.
- Staged suggestions retain fingerprint checks.
- The API probe remains unauthenticated and should be used only with trusted
  clients on localhost, trusted LAN, VPN, or SSH tunnel.
