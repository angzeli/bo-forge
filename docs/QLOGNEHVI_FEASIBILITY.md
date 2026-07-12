# qLogNEHVI Feasibility Review

BO Forge v2.2.2 keeps noisy multi-objective BO under feasibility review.
`bo.acquisition: qlog_nehvi` is not a public acquisition yet, and
`source=qlog_nehvi` is not a valid CSV source.

The review result is: conservative qLogNEHVI looks feasible for a narrow
future scope, but it should not be exposed until pending-state semantics,
duplicate protection, and unsupported combinations are locked down in tests.

## Acquisition Shape

The existing coupled multi-objective qLogEHVI path already fits one
multi-output `SingleTaskGP` with shared `train_X` and `train_Y`. That shape is
compatible with the expected qLogNEHVI model shape if BO Forge keeps coupled
objectives only.

The safe future shape is:

- use the same encoded decision space and objective direction transforms as
  qLogEHVI;
- use encoded observed design points as `X_baseline`; objective values remain model outputs, not baseline inputs;
- keep the configured multi-objective reference point transformed in model
  space;
- pass accepted pending candidate designs as `X_pending` when the campaign
  review state allows it;
- keep `m >= 2`, with `2 <= m <= 4` as the tested range.

Open implementation risk: qLogNEHVI must use the same objective-order contract
as `pareto_front`, `hypervolume`, and qLogEHVI so reported objective order,
reference points, and model-space transforms cannot drift.

## Pending Semantics

The recommended future semantics are:

- without review, all `status=suggested` rows are active pending designs and
  should become `X_pending`;
- with review enabled, `review_status=accepted` rows become `X_pending`;
- with review enabled, `review_status=pending` rows block new qLogNEHVI
  suggestions until accepted, rejected, or deferred;
- `review_status=rejected` and `review_status=deferred` rows remain auditable
  and duplicate-protected, but are not passed as `X_pending`.

This mirrors the single-objective qLogNEI safety model and avoids silently
treating unapproved review rows as experiments in flight.

## Recommended v2.2.3 Scope

If implemented in v2.2.3, qLogNEHVI should start with this narrow scope:

- coupled multi-objective campaigns only;
- `2 <= m <= 4`;
- no new CSV columns;
- generated model-based rows use `source=qlog_nehvi`;
- no decoupled or asynchronous objective rows;
- review metadata supported with the pending semantics above;
- no deterministic cost integration;
- no replicate-aware noisy MOBO;
- no structured-stage, contextual, or multi-fidelity qLogNEHVI;
- no Streamlit creation controls beyond loading valid backend configs.

If this scope proves unstable, v2.2.3 should close the v2.2 line with
qLogNEHVI explicitly deferred.

## Rejected Combinations For First Implementation

| Combination | v2.2.2 decision | Reason |
| --- | --- | --- |
| Multi-objective + review | Feasible candidate | Needs explicit pending and accepted semantics. |
| Multi-objective without review | Feasible candidate | Suggested rows can map directly to `X_pending`. |
| Multi-objective + cost | Defer | Cost-aware qLogNEHVI needs utility and budget semantics. |
| Multi-objective + replicates | Defer | Group means plus noisy pending baselines need separate validation. |
| Structured + qLogNEHVI | Defer | Stage-specific noisy MOBO is a separate design. |
| Contextual + qLogNEHVI | Defer | Fixed-context noisy MOBO needs a separate context contract. |
| Multi-fidelity + qLogNEHVI | Defer | No noisy multi-fidelity multi-objective design exists yet. |
| Decoupled objectives | Defer | BO Forge requires coupled objective rows for current MOBO. |

## v2.2.2 Guardrails

v2.2.2 intentionally enforces:

- `bo.acquisition: qlog_nehvi` raises a clear `ConfigError`;
- `source=qlog_nehvi` remains invalid in CSV logs;
- qLogEHVI examples continue to validate unchanged;
- qLogNEI examples continue to validate unchanged;
- no public helper, CLI command, notebook, Streamlit control, or API endpoint
  exposes qLogNEHVI.
