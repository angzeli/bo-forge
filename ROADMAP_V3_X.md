# BO Forge v3.x Roadmap

Current baseline: `v3.0.0`.

v3.0.0 establishes an architecture and scientific-UX baseline for the existing
local CSV-backed campaign toolkit. It decomposes oversized implementation
modules, separates application and optional API ownership, simplifies the
Streamlit workbench, and standardizes scientific figures while preserving the
documented BO Forge workflows and durable YAML/CSV formats.

```mermaid
flowchart LR
    v30["v3.0.0<br/>Architecture + scientific UX reset"]
    class v30 baseline
    classDef baseline fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#111827;
```

The detailed v3.x expansion roadmap remains intentionally open. New optimizer,
schema, campaign-combination, authentication, persistence, or deployment work
requires a separate reviewed plan rather than being implied by this baseline.

## v3.0.0 - Architecture And Scientific UX Reset

Status: completed baseline

- Preserve the documented top-level `bo_forge` API and legacy implementation
  module imports through compatibility modules and focused internal ownership.
- Separate configuration, campaign, optimization, and diagnostic ownership.
- Move application workflow coordination to `bo_forge.application`.
- Isolate optional FastAPI ownership in `bo_forge_api` while retaining the
  compatibility imports and `bo-forge-api` command.
- Replace the five-panel workbench with `Campaign`, `Run`, and `Analyze`.
- Add native Day/Night theme state with URL persistence.
- Apply one scoped scientific figure and semantic-color contract.
- Enforce module, function, test-file, complexity, and layer-boundary gates.

See [docs/MIGRATION_V3.md](docs/MIGRATION_V3.md) for the behavior-preserving
module and UI mapping.
