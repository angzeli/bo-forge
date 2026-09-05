# Migrating To BO Forge v3

BO Forge v3 changes internal ownership and the local workbench layout. It
does not require YAML or CSV migration.

## Durable Campaign Files

Existing YAML configuration keys, defaults, canonical CSV columns, status
transitions, fingerprints, write locking, and explicit append behavior remain
compatible. YAML and CSV continue to be campaign source data. Starting in
v3.1.0 and later, `CampaignSession.initialize()` and `bo-forge init-log` create an
additional versioned provenance manifest beside a new CSV log. Existing
campaigns without this sidecar remain legacy-compatible and are not adopted
silently. Pickled internal Python objects are not a compatibility target.

Starting in v3.1.1, compatible loading remains the default, while
`CampaignSession.from_files(..., provenance_policy="required")` can enforce
managed-only resume. Interrupted schema-v1 transactions now require explicit
`recover_provenance()` or `bo-forge provenance-recover`; ordinary load and mutation
paths do not repair them automatically. No manifest, YAML, or CSV migration is needed.

## Python Imports

The documented names in `bo_forge.__all__` retain their v2 names and
signatures. The familiar implementation modules remain importable as
compatibility modules while focused packages own the extracted concerns:

- `bo_forge.config` delegates parsing and combination checks to `_config`;
- `bo_forge.logs` and `session` retain the public mutation and session
  entrypoints while shared schema, validation, persistence, and report helpers
  live in `_campaign`;
- `bo_forge.suggestions` delegates routing and candidate generation to
  `_optimization`;
- `bo_forge.diagnostics` delegates figure implementations to `_diagnostics`.

Only the documented top-level imports are a stable v3 API. Private helpers in
facade or underscore-prefixed modules may continue to evolve.

## Application And API Ownership

The non-HTTP `CampaignAppService` now lives in `bo_forge.application`.
`bo_forge_app.service` remains an import-compatible shim.

The experimental FastAPI app, launcher, contracts, and process-local stage
store now belong to `bo_forge_api`. Existing `bo_forge_app.api`,
`bo_forge_app.api_cli`, and `bo_forge_app.stages` imports remain available as
compatibility shims. The `bo-forge-api` command, optional `api` extra, routes,
payloads, status codes, and trusted-network boundary are unchanged.

## Streamlit Navigation

The workbench now uses three task-oriented areas:

| v2 panel | v3 area |
| --- | --- |
| `Overview`, `Data` | `Campaign` |
| `Suggest`, `Resolve` | `Run` |
| `Reports` | `Analyze` |

Stored panel state is mapped automatically. Navigation and Day/Night theme
changes do not invalidate staged suggestions; campaign, stage, context,
config, log, or staged-payload changes retain the existing invalidation rules.

## Scientific Figures

Figure data and output-path behavior are unchanged. Plot styling now uses a
scoped Matplotlib context, a shared semantic BO color registry, white opaque
export backgrounds, and 600 dpi PNG output. One requested plot path still
produces exactly one file; PDF output remains supported.
