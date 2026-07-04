# BO Forge v2.x Roadmap

This roadmap begins with the v2.0.0 hardening baseline. It is directional, not
a release promise. BO Forge v2.x should be a line of coherence and controlled expansion,
not a rewrite of the CSV-backed campaign model.

Current baseline: `v2.0.0`. The v2.0.0 release preserves v1 YAML/CSV/session,
CLI, notebook, Streamlit, service, and experimental API probe behavior while
adding a capability matrix, stronger package-boundary tests, and an active v2.x
roadmap. Real modeling expansion starts after the hardening baseline.

## Roadmap So Far

```mermaid
flowchart LR
    v10["v1.x<br/>Stable local BO Forge baseline"] --> v20["v2.0<br/>Hardening + capability matrix"] --> v21["v2.1<br/>Model profiles"] --> v22["v2.2<br/>Noisy + pending-aware BO"] --> v23["v2.3<br/>Controlled combinations"] --> v24["v2.4<br/>Multi-fidelity expansion"] --> v25["v2.5<br/>App/API operational hardening"]

    class v10,v20 done
    class v21,v22,v23,v24,v25 future

    classDef done fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px,color:#111827;
    classDef future fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#111827;
```

## v2.0.x - Stable v2 Baseline

Status: active

- Preserve v1 YAML, CSV, session, CLI, notebook, Streamlit, service, and
  experimental API probe behavior.
- Add [docs/CAPABILITY_MATRIX.md](docs/CAPABILITY_MATRIX.md) as the supported
  and deferred combination reference.
- Harden package and release tests for wheel/sdist boundaries.
- Keep the FastAPI probe experimental, optional, root-bound, and unauthenticated.
- Keep production auth, database storage, and public internet deployment out of
  scope.

## v2.1.x - Model Profiles And Advanced Surrogates

Status: planned

- Introduce curated model profiles instead of raw BoTorch kernel passthrough.
- Keep profile names stable and user-facing.
- Add advanced kernels only through validated BO Forge profiles.
- Preserve CSV schema compatibility unless a future profile requires explicit
  metadata.

## v2.2.x - Noisy And Pending-Aware BO

Status: planned

- Start with qLogNEI for noisy or pending-aware single-objective workflows.
- Consider qLogNEHVI only after the single-objective noisy path is stable.
- Keep learned-noise and replicate-derived variance semantics clear.
- Avoid adding noisy multi-objective scope before the data-safety model is
  explicit.

## v2.3.x - Controlled Feature Combinations

Status: planned

- Add selected combinations deliberately rather than enabling an unrestricted
  feature cross-product.
- Candidate combinations include contextual + review, contextual + cost,
  contextual + replicates, and structured + cost.
- Keep unsupported combinations documented in the capability matrix.

## v2.4.x - Multi-Fidelity Expansion

Status: planned

- Explore discrete fidelity levels.
- Expand fidelity diagnostics.
- Revisit stage/fidelity and context/fidelity interactions only after the
  conservative qMFKG baseline remains stable.

## v2.5.x - App/API Operational Hardening

Status: planned

- Consider server-side staged API state.
- Consider signed staged bundles.
- Improve safeguards around concurrent writes.
- Keep any production app/API direction explicit about auth, persistence, and
  deployment boundaries.

## Not Yet

- No mandatory database.
- No full authentication system.
- No SaaS/team workflows.
- No unrestricted feature cross-product.
- No raw low-level kernel API as the first modeling extension.
- No replacement of CSV logs as the source of truth.
