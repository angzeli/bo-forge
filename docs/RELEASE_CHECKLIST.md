# BO Forge Release Checklist

Preparing a release and publishing a release are separate operations. Required
CI for the exact release commit is the authoritative gate. Local checks support
that evidence but do not replace it.

The v3.2.3 preparation closes predictive diagnostics with known-distribution
acceptance, adapter verification, and package probes. Deterministic correctness
and bounded fitted-model integration are distinct evidence levels.
Guidance is not calibration certification.
Commit only the reviewed release changes. Publication still requires passing required
CI on that exact commit; push, tagging, and release publication each require separate
authorization. Implementation completion does not grant publication approval.

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
  tests/test_release_assurance.py \
  tests/test_release_artifacts.py \
  tests/test_release_dependency_resolution.py
git diff --check
```

These checks cover version consistency, constraints/workflow contracts,
release-facing path hygiene, package metadata, docs assets, and release
boundaries.

For v3.1.x provenance releases, also run the managed-transaction and adapter
coverage directly:

```bash
/tmp/bo-forge-release/bin/python -m pytest -p no:cacheprovider \
  tests/test_provenance.py \
  tests/test_provenance_resume.py \
  tests/test_provenance_lifecycle.py \
  tests/test_provenance_lifecycle_adapters.py \
  tests/test_provenance_lifecycle_hardening.py \
  tests/test_provenance_acceptance.py \
  tests/test_provenance_closeout.py \
  tests/test_cli_core_and_analysis.py \
  tests/test_app_service.py \
  tests/test_api.py \
  tests/test_api_provenance.py
```

Verify that managed campaign fixtures keep each CSV with its manifest, cross-directory
initialization uses `~/...` rather than serializing author-home components, mismatched
managed files fail closed on load, and rollback-failure tests retain a recoverable CSV
backup.

For v3.1.3 acceptance, follow [Start Here](../START_HERE.md) from a freshly
extracted source archive outside the checkout, using a clean Python 3.12
environment and `python -m pip install ".[app]"`. Check `pip check`, version,
doctor, and the actual installed launcher in a browser: `Campaign`, `Run`, and
`Analyze` must render. Stop the server afterward. Record platform/interpreter,
archive identity, and any unverified OS instructions or environment limitations.
The wheel must exclude `START_HERE.md`; the sdist must contain it and the
lifecycle-hardening/acceptance tests. Existing artifact probes also exercise
bounded adopt, mutate, accept-config, and fork sequences without a model fit.

## 3. Full Local Preflight

For v3.2.3, run the output-free
`notebooks/23_predictive_diagnostics.ipynb` in an isolated working directory with
the installed package. Its temporary-directory setup creates 20 synthetic rows
through the existing config parser, evaluates `default` and `smooth` with three
folds, and explicitly exports/plots the result without a campaign fixture.
Check summary, predictions, fold outcomes, and metadata; verify original-unit,
noise-inclusive uncertainty and unchanged input CSV/config bytes. Keep the
committed notebook output-free. Inspect the same explicit evaluation route through
`bo-forge model-evaluate --profile default --profile smooth --folds 3 --seed 0`
with config/log paths, and Streamlit `Analyze`; loading or rerunning the page must
not start evaluation. Confirm in-sample labels and session-owned metadata, as well
as rejected unsupported campaigns, observation/fold bounds, and duplicate designs.
These focused checks supplement, not replace, full preflight and exact-commit CI.
Include `tests/test_predictive_exports.py`, `tests/test_predictive_hardening.py`,
`tests/test_fit_metadata.py`, `tests/test_cli_model_evaluate.py`, and
`tests/test_streamlit_predictive_evaluation.py`. Verify failures at every export
write and publication, concurrent no-overwrite publication, retry without refit,
incomplete summary messages, and legacy/managed file changes during evaluation.
The existing Linux full-suite and macOS filesystem jobs exercise the new export
publication path as well as unchanged provenance-fork publication. Inspect summary,
predictions, fold outcomes, and JSON from clean installed-artifact probes too.
Also run `tests/test_predictive_interpretation.py` and `tests/test_notebooks.py`:
check hand-worked arithmetic, NLPD unit rescaling, all unchanged notebook
computation/identity fields, and documentation links. Verify the collapsed
Streamlit interpretation reference for complete and incomplete results; rendering
it must not refit, export, mutate campaigns, or discard cached results. These
checks verify interpretation examples and software behavior, not calibration.

Run `tests/test_predictive_acceptance.py` and
`tests/test_predictive_workflow_acceptance.py` for v3.2.3 synthetic acceptance:
200 midpoint normal quantiles and independent arithmetic, plus complete and
incomplete session/service/CLI/Streamlit/export paths. New AppTests must restore
global state for later process tests. Inspect the existing notebook figures in
an isolated copy without changing its cells, IDs, six-fit workload, or outputs.
Run `tests/test_predictive_evaluation.py::test_bounded_real_gp_fit` explicitly in
numerical CI alongside qMFKG. Require finite predictions, positive observation
variance, and complete outputs, not a prescribed score or learned-model coverage.
External wheel and sdist probes must evaluate five rows with one profile and two
folds, then verify tables, metadata, no-overwrite export, and unchanged sources.
Run `tests/test_diagnostic_export_safety.py` for campaign-file plot aliases,
reserved legacy manifest directories, final-write guards, and Streamlit retry
without refitting. These filesystem checks also run in the macOS selection.

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
| `Bounded real numerical paths` | CPU-only real qMFKG and five-row/two-fold predictive-GP integration with a bounded job timeout |
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
The source distribution must include `docs/PROVENANCE.md`; the wheel must
include the provenance facade and internal campaign transaction modules.

## 7. External Installation Probes

Create each probe outside the source checkout, clear `PYTHONPATH`, install under
the matching constraints, and run `pip check`:

```bash
wheel=$(python3.12 - <<'PY'
from pathlib import Path

artifacts = sorted(Path("/tmp/bo-forge-dist").glob("bo_forge-*.whl"))
if len(artifacts) != 1:
    raise SystemExit(f"Expected exactly one wheel, found {len(artifacts)}")
print(artifacts[0])
PY
)
python3.12 -m venv /tmp/bo-forge-wheel-probe
/tmp/bo-forge-release/bin/uv pip install \
  --python /tmp/bo-forge-wheel-probe/bin/python \
  --torch-backend cpu \
  -c requirements/constraints-py312-linux-x86_64.txt \
  "$wheel"
(cd /tmp && PYTHONPATH= /tmp/bo-forge-wheel-probe/bin/bo-forge --version)
/tmp/bo-forge-wheel-probe/bin/python -m pip check
```

Discover and install the sdist with the same exact-one rule. Then install the
app and API extras in separate clean environments:

```bash
sdist=$(python3.12 - <<'PY'
from pathlib import Path

artifacts = sorted(Path("/tmp/bo-forge-dist").glob("bo_forge-*.tar.gz"))
if len(artifacts) != 1:
    raise SystemExit(f"Expected exactly one source distribution, found {len(artifacts)}")
print(artifacts[0])
PY
)
python3.12 -m venv /tmp/bo-forge-sdist-probe
/tmp/bo-forge-release/bin/uv pip install \
  --python /tmp/bo-forge-sdist-probe/bin/python \
  --torch-backend cpu \
  -c requirements/constraints-py312-linux-x86_64.txt \
  "$sdist"
/tmp/bo-forge-sdist-probe/bin/python -m pip check

python3.12 -m venv /tmp/bo-forge-app-probe
/tmp/bo-forge-release/bin/uv pip install \
  --python /tmp/bo-forge-app-probe/bin/python \
  --torch-backend cpu \
  -c requirements/constraints-py312-linux-x86_64.txt \
  "${wheel}[app]"
/tmp/bo-forge-app-probe/bin/python -m pip check

python3.12 -m venv /tmp/bo-forge-api-probe
/tmp/bo-forge-release/bin/uv pip install \
  --python /tmp/bo-forge-api-probe/bin/python \
  --torch-backend cpu \
  -c requirements/constraints-py312-linux-x86_64.txt \
  "${wheel}[api]"
/tmp/bo-forge-api-probe/bin/python -m pip check

PYTHONPATH= /tmp/bo-forge-wheel-probe/bin/bo-forge --help
PYTHONPATH= /tmp/bo-forge-wheel-probe/bin/bo-forge doctor
PYTHONPATH= /tmp/bo-forge-app-probe/bin/bo-forge-app --help
PYTHONPATH= /tmp/bo-forge-api-probe/bin/bo-forge-api --help
(cd /tmp && PYTHONPATH= /tmp/bo-forge-api-probe/bin/python -c \
  'from pathlib import Path; import bo_forge_api; from bo_forge_api.api import create_app; assert not hasattr(bo_forge_api, "create_app"); app = create_app(root=Path(".")); assert app.title')
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
5. smoke-installs the exact wheel, source distribution, app extra, and API extra;
6. retains verified files as private GitHub Actions artifacts for 14 days.

It does not create a GitHub Release, publish to PyPI, use trusted publishing,
attach public files, or generate release prose.

## 9. Actual Publication Is Separate

Only after the exact release commit is pushed, clean, protected as intended,
and green in required CI should a maintainer create the matching tag. The files
used for a later manual release must come from the tag-gate run for that exact
tag, never from a workstation's old `dist/` directory.

Creating a tag, GitHub Release, final announcement, or registry upload requires
separate explicit authorization. Preparing v3.2.3 does none of those actions.
