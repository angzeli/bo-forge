# Contributing To BO Forge

BO Forge accepts focused changes that preserve its local-first campaign model:
YAML configuration and CSV logs are durable user data, suggestion generation
is non-mutating, and writes remain explicit and failure-atomic.

## Supported development environments

Required CI covers CPython 3.11 and 3.12 on Linux. macOS 3.12 provides focused
coverage for path, symlink, locking, stale-fingerprint, rollback, file-mode,
and process-mutation behavior. Broad package requirements live in
`pyproject.toml`; generated development constraints live in `requirements/`.

Start from an empty environment using the matching commands in
[`requirements/README.md`](requirements/README.md). Do not rely on a stale
editable environment when preparing a release.

## Checks

For a focused change, run the directly affected tests and Ruff:

```bash
python -m pytest -p no:cacheprovider tests/path_to_relevant_test.py
ruff check . --no-cache
git diff --check
```

Before a release-assurance commit, run:

```bash
python -m pip check
python -m pytest -p no:cacheprovider
ruff check . --no-cache
python examples/quickstart.py
python -m bo_forge doctor
python -m build --outdir /tmp/bo-forge-dist
python -m twine check /tmp/bo-forge-dist/*
```

Run quickstart from a temporary copy when release hygiene matters so no working
log is created in the checkout. The release checklist describes external
wheel/sdist installation probes and packaged command smokes.

## Dependency changes

Keep end-user lower bounds and compatibility ranges in `pyproject.toml`. When a
runtime dependency or optional extra changes:

1. update the appropriate project dependency or extra;
2. regenerate every affected constraints file with pinned `uv` commands from
   `requirements/README.md`;
3. run clean bootstrap and `pip check`;
4. update package-boundary, optional-import, and release tests;
5. update only documentation whose install or capability statements changed.

Do not hand-edit generated constraints or commit virtual environments, caches,
build output, local reports, working logs, notebook execution output, or
machine-specific paths.

## Architecture and compatibility

Documented top-level `bo_forge` exports, `CampaignSession`, YAML keys, canonical
CSV schemas, CLI commands, and mutation semantics are compatibility contracts.
Core behavior belongs in `bo_forge`; Streamlit and FastAPI call the shared
application/session layers. Core imports must remain independent of Streamlit,
FastAPI, Uvicorn, and Matplotlib unless their functionality is requested.

Add behavior-freeze tests before changing numerical routing, candidate order,
seeds, retries, fallbacks, persistence, or error contracts. Avoid broad module
refactors during feature or release-hardening patches.

## Documentation and releases

Update tests with every fixed behavior defect. Update docs only when existing
wording becomes inaccurate. Release-facing files must not contain private
author-home paths.

Preparing a release is distinct from publishing it. A release preparation may
update version metadata, constraints, CI, tests, docs, and package checks. It
must not create a tag, GitHub Release, or registry upload unless that separate
action is explicitly authorized. Required CI for the exact release commit is
the authoritative gate; workstation checks are supporting evidence.
