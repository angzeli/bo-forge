# BO Forge Release Checklist

Preparing a release and publishing a release are separate operations. Required
CI for the exact release commit is the authoritative gate. Local checks support
that evidence but do not replace it.

## 1. Prepare An Isolated Environment

Use Python 3.11 or 3.12 on Linux, or Python 3.12 on macOS, and choose the
matching generated file from `requirements/`:

```bash
python3.12 -m venv /tmp/bo-forge-release
/tmp/bo-forge-release/bin/python -m pip install "uv==0.11.3"
/tmp/bo-forge-release/bin/uv pip install \
  --python /tmp/bo-forge-release/bin/python \
  --require-hashes --torch-backend cpu \
  -r requirements/constraints-py312-linux-x86_64.txt
/tmp/bo-forge-release/bin/uv pip install \
  --python /tmp/bo-forge-release/bin/python \
  --no-deps --no-build-isolation -e .
/tmp/bo-forge-release/bin/python -m pip check
```

On macOS, omit `--torch-backend cpu` and use the x86_64 or arm64 file matching
`uname -m`. See [`requirements/README.md`](../requirements/README.md) for exact
regeneration and freshness-verification commands.

Never use a stale repository `.venv` as the only release environment. Do not
commit virtual environments, caches, generated reports, local working logs, or
build directories.

## 2. Fast Local Checks

```bash
/tmp/bo-forge-release/bin/ruff check . --no-cache
/tmp/bo-forge-release/bin/python -m pytest -p no:cacheprovider \
  tests/test_v301_release_assurance.py \
  tests/test_release_artifacts.py \
  tests/test_release_dependency_resolution.py
git diff --check
```

These checks cover version consistency, constraints/workflow contracts,
release-facing path hygiene, package metadata, docs assets, and release
boundaries.

## 3. Full Local Preflight

```bash
/tmp/bo-forge-release/bin/python -m pytest -p no:cacheprovider
/tmp/bo-forge-release/bin/ruff check . --no-cache
/tmp/bo-forge-release/bin/python -m bo_forge --version
/tmp/bo-forge-release/bin/python -m bo_forge doctor
/tmp/bo-forge-release/bin/bo-forge --help
/tmp/bo-forge-release/bin/bo-forge-app --help
/tmp/bo-forge-release/bin/bo-forge-api --help
git diff --check
```

Run quickstart from a temporary copy so it cannot create a repository working
log:

```bash
probe=/tmp/bo-forge-quickstart
rm -rf "$probe"
mkdir -p "$probe"
cp -R configs examples "$probe/"
(cd "$probe" && PYTHONPATH="$OLDPWD" \
  /tmp/bo-forge-release/bin/python examples/quickstart.py)
```

Representative read-only CLI checks:

```bash
/tmp/bo-forge-release/bin/python -m bo_forge validate \
  --config configs/01_simple_2d_maximise_logei.yaml \
  --log examples/01_simple_2d_maximise_logei_campaign_log.csv
/tmp/bo-forge-release/bin/python -m bo_forge validate \
  --config configs/18_noisy_pending_qlognei.yaml \
  --log examples/18_noisy_pending_qlognei_campaign_log.csv
/tmp/bo-forge-release/bin/python -m bo_forge fidelity-coverage \
  --config configs/22_discrete_multi_fidelity_qmfkg.yaml \
  --log examples/22_discrete_multi_fidelity_qmfkg_campaign_log.csv
```

## 4. Required CI For The Exact Commit

`.github/workflows/ci.yml` must be green for the exact release commit.

| CI job | Required evidence |
| --- | --- |
| `Static validation` | Ruff, syntax, whitespace, generated-constraints freshness |
| `Core tests (Python 3.11)` | Complete Linux pytest suite under Python 3.11 |
| `Core tests (Python 3.12)` | Complete Linux pytest suite under Python 3.12 |
| `macOS filesystem and CLI` | Path, symlink, locking, fingerprint, rollback, mode, process, and CLI checks |
| `Bounded real numerical paths` | CPU-only representative real qMFKG execution with a bounded job timeout |
| `Build and external artifact probes` | PEP 517 build, Twine, package boundaries, external wheel/sdist installs, `pip check`, packaged entrypoints |

Workflow files use read-only repository permissions and bounded timeouts. They
do not use `pull_request_target`, publishing credentials, or non-loopback test
listeners.

Branch protection and tag protection are GitHub repository settings. This
repository documents the intended required jobs, but workflow files alone do
not prove those server-side settings are enabled. Verify them separately before
publishing.

Confirm the release and security guidance is tracked and the checkout contains
only intended work:

```bash
git status --short
git ls-files --error-unmatch CONTRIBUTING.md SECURITY.md \
  docs/API_SECURITY.md docs/STREAMLIT_DEPLOYMENT.md \
  requirements/README.md .github/workflows/ci.yml \
  .github/workflows/release-gate.yml
```

All intended files must be committed before evaluating the exact commit in CI.
Review [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md) and
[API_SECURITY.md](API_SECURITY.md) whenever launcher or API deployment behavior
changes.

## 5. Version And Release Identity

Confirm that these agree:

- `pyproject.toml` project version;
- `bo_forge.__version__`;
- README current version;
- changelog current entry;
- active roadmap baseline/status;
- installation and artifact filenames;
- package maturity classifier;
- intended tag `v<package-version>`.

Historical changelog and roadmap entries should retain their historical
versions. The release-facing scan must report no author-home absolute paths.

## 6. Local Artifact Build

Local builds are validation only. Build outside the checkout and never reuse an
old workstation `dist/` directory for publication:

```bash
artifact_dir=/tmp/bo-forge-dist
rm -rf "$artifact_dir"
mkdir -p "$artifact_dir"
/tmp/bo-forge-release/bin/python -m build --outdir "$artifact_dir"
/tmp/bo-forge-release/bin/python -m twine check "$artifact_dir"/*
```

Inspect the wheel and sdist with the repository contracts:

```bash
/tmp/bo-forge-release/bin/python -m pytest -p no:cacheprovider \
  tests/test_release_artifacts.py::test_built_distributions_install_from_outside_source_tree
```

The wheel contains only runtime packages and distribution metadata. The sdist
contains release docs, generated constraints, configs, seed logs, notebooks,
and tests required by the release contract.

Specifically, verify that the wheel contains the `bo_forge`, `bo_forge_app`,
and `bo_forge_api` packages, while release documentation, examples, notebooks,
tests, and generated constraints remain outside the wheel and inside the sdist.

## 7. External Installation Probes

Create each probe outside the source checkout, clear `PYTHONPATH`, install under
the matching constraints, and run `pip check`:

```bash
python3.12 -m venv /tmp/bo-forge-wheel-probe
/tmp/bo-forge-release/bin/uv pip install \
  --python /tmp/bo-forge-wheel-probe/bin/python \
  --torch-backend cpu \
  -c requirements/constraints-py312-linux-x86_64.txt \
  /tmp/bo-forge-dist/bo_forge-3.0.1-py3-none-any.whl
(cd /tmp && PYTHONPATH= /tmp/bo-forge-wheel-probe/bin/bo-forge --version)
/tmp/bo-forge-wheel-probe/bin/python -m pip check
```

Repeat with the sdist and with wheel extras `[app,api]`. Verify:

```bash
PYTHONPATH= /tmp/bo-forge-wheel-probe/bin/bo-forge --help
PYTHONPATH= /tmp/bo-forge-wheel-probe/bin/bo-forge doctor
PYTHONPATH= /tmp/bo-forge-wheel-probe/bin/bo-forge-app --help
PYTHONPATH= /tmp/bo-forge-wheel-probe/bin/bo-forge-api --help
```

Inspect `bo_forge.__file__` and confirm it is under the probe environment, not
the source checkout.

## 8. Future Tag Gate

`.github/workflows/release-gate.yml` is a validation-only future release path.
For a pushed `v*` tag or a manual run naming an existing tag, it:

1. checks out the exact tagged commit;
2. requires the tag to equal `v<pyproject version>`;
3. runs required tests and Ruff;
4. builds and verifies wheel/sdist in runner-temporary storage;
5. smoke-installs the exact wheel;
6. retains verified files as private GitHub Actions artifacts for 14 days.

It does not create a GitHub Release, publish to PyPI, use trusted publishing,
attach public files, or generate release prose.

## 9. Actual Publication Is Separate

Only after the exact release commit is pushed, clean, protected as intended,
and green in required CI should a maintainer create the matching tag. The files
used for a later manual release must come from the tag-gate run for that exact
tag, never from a workstation's old `dist/` directory.

Creating a tag, GitHub Release, final announcement, or registry upload requires
separate explicit authorization. Preparing v3.0.1 does none of those actions.
