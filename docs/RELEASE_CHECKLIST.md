# 🚢 BO Forge Release Checklist

Use this checklist before publishing a GitHub release or PyPI package.

## ✅ Preflight

```bash
./.venv/bin/pytest
./.venv/bin/ruff check .
REPO_ROOT="$(pwd)"
rm -rf /tmp/bo_forge_quickstart_probe
mkdir -p /tmp/bo_forge_quickstart_probe
cp -R configs examples /tmp/bo_forge_quickstart_probe/
(cd /tmp/bo_forge_quickstart_probe && PYTHONPATH="$REPO_ROOT" "$REPO_ROOT/.venv/bin/python" examples/quickstart.py)
./.venv/bin/python -m bo_forge doctor
./.venv/bin/python -m bo_forge validate --config configs/15_multi_fidelity_qmfkg.yaml --log examples/15_multi_fidelity_qmfkg_campaign_log.csv
./.venv/bin/python -m bo_forge suggest --config configs/15_multi_fidelity_qmfkg.yaml --log examples/15_multi_fidelity_qmfkg_campaign_log.csv --batch-size 1
./.venv/bin/python -m bo_forge fidelity-summary --config configs/15_multi_fidelity_qmfkg.yaml --log examples/15_multi_fidelity_qmfkg_campaign_log.csv
./.venv/bin/python -m bo_forge fidelity-coverage --config configs/15_multi_fidelity_qmfkg.yaml --log examples/15_multi_fidelity_qmfkg_campaign_log.csv
./.venv/bin/python -m bo_forge plot --config configs/15_multi_fidelity_qmfkg.yaml --log examples/15_multi_fidelity_qmfkg_campaign_log.csv --kind fidelity-diagnostics --output /tmp/bo_forge_fidelity_diagnostics.png
./.venv/bin/python -m bo_forge plot --config configs/15_multi_fidelity_qmfkg.yaml --log examples/15_multi_fidelity_qmfkg_campaign_log.csv --kind fidelity-progress --output /tmp/bo_forge_fidelity_progress.png
./.venv/bin/python -m bo_forge validate --config configs/22_discrete_multi_fidelity_qmfkg.yaml --log examples/22_discrete_multi_fidelity_qmfkg_campaign_log.csv
./.venv/bin/python -m bo_forge suggest --config configs/22_discrete_multi_fidelity_qmfkg.yaml --log examples/22_discrete_multi_fidelity_qmfkg_campaign_log.csv --batch-size 2
./.venv/bin/python -m bo_forge suggest --config configs/22_discrete_multi_fidelity_qmfkg.yaml --log examples/22_discrete_multi_fidelity_qmfkg_campaign_log.csv --batch-size 4
./.venv/bin/python -m bo_forge fidelity-summary --config configs/22_discrete_multi_fidelity_qmfkg.yaml --log examples/22_discrete_multi_fidelity_qmfkg_campaign_log.csv
./.venv/bin/python -m bo_forge fidelity-coverage --config configs/22_discrete_multi_fidelity_qmfkg.yaml --log examples/22_discrete_multi_fidelity_qmfkg_campaign_log.csv
./.venv/bin/python -m bo_forge plot --config configs/22_discrete_multi_fidelity_qmfkg.yaml --log examples/22_discrete_multi_fidelity_qmfkg_campaign_log.csv --kind fidelity-diagnostics --output /tmp/bo_forge_discrete_fidelity_diagnostics.png
./.venv/bin/python -m bo_forge plot --config configs/22_discrete_multi_fidelity_qmfkg.yaml --log examples/22_discrete_multi_fidelity_qmfkg_campaign_log.csv --kind fidelity-progress --output /tmp/bo_forge_discrete_fidelity_progress.png
./.venv/bin/python -m bo_forge validate --config configs/16_contextual_logei.yaml --log examples/16_contextual_logei_campaign_log.csv
./.venv/bin/python -m bo_forge suggest --config configs/16_contextual_logei.yaml --log examples/16_contextual_logei_campaign_log.csv --context feedstock_acidity=0.25 --batch-size 1
./.venv/bin/python -m bo_forge context-summary --config configs/16_contextual_logei.yaml --log examples/16_contextual_logei_campaign_log.csv
./.venv/bin/python -m bo_forge plot --config configs/16_contextual_logei.yaml --log examples/16_contextual_logei_campaign_log.csv --kind context-diagnostics --output /tmp/bo_forge_context_diagnostics.png
./.venv/bin/python -m bo_forge validate --config configs/17_model_profile_logei.yaml --log examples/17_model_profile_campaign_log.csv
./.venv/bin/python -m bo_forge model-summary --config configs/17_model_profile_logei.yaml --log examples/17_model_profile_campaign_log.csv
./.venv/bin/python -m bo_forge model-compare --config configs/17_model_profile_logei.yaml --log examples/17_model_profile_campaign_log.csv
./.venv/bin/python -m bo_forge plot --config configs/17_model_profile_logei.yaml --log examples/17_model_profile_campaign_log.csv --kind model-diagnostics --output /tmp/bo_forge_model_diagnostics.png
./.venv/bin/python -m bo_forge plot --config configs/17_model_profile_logei.yaml --log examples/17_model_profile_campaign_log.csv --kind model-comparison --output /tmp/bo_forge_model_comparison.png
./.venv/bin/python -m bo_forge validate --config configs/18_noisy_pending_qlognei.yaml --log examples/18_noisy_pending_qlognei_campaign_log.csv
./.venv/bin/python -m bo_forge qlog-nei-summary --config configs/18_noisy_pending_qlognei.yaml --log examples/18_noisy_pending_qlognei_campaign_log.csv
./.venv/bin/python -m bo_forge suggest --config configs/18_noisy_pending_qlognei.yaml --log examples/18_noisy_pending_qlognei_campaign_log.csv --batch-size 1
./.venv/bin/python -m bo_forge plot --config configs/18_noisy_pending_qlognei.yaml --log examples/18_noisy_pending_qlognei_campaign_log.csv --kind qlog-nei-diagnostics --output /tmp/bo_forge_qlog_nei_diagnostics.png
./.venv/bin/python -m bo_forge validate --config configs/19_multi_objective_qlognehvi.yaml --log examples/19_multi_objective_qlognehvi_campaign_log.csv
./.venv/bin/python -m bo_forge suggest --config configs/19_multi_objective_qlognehvi.yaml --log examples/19_multi_objective_qlognehvi_campaign_log.csv --batch-size 1
./.venv/bin/python -m streamlit --version
git diff --check
git status --short
git ls-files --error-unmatch docs/API_SECURITY.md
```

Confirm:

- `LICENSE` exists.
- `README.md` includes install commands for the core package, app/API extras, `bo-forge`, `bo-forge-app`, and `bo-forge-api`.
- `docs/CAPABILITY_MATRIX.md` lists supported, read-only, rejected, and deferred workflow combinations.
- `docs/STREAMLIT_DEPLOYMENT.md` describes local-only, trusted-LAN, SSH/VPN, and authenticated reverse-proxy modes.
- subprocess import-boundary tests prove package import, version, help,
  validation, fidelity-summary, and fidelity-coverage avoid Torch, BoTorch, GPyTorch, Matplotlib,
  Streamlit, and FastAPI;
- `docs/PERFORMANCE_BENCHMARKS.md` records five-run warm-cache startup and
  representative qMFKG `q=1/2/4` medians without machine-dependent CI
  thresholds.
- `docs/API_PROBE.md` describes the experimental API probe, root-bound paths, and no-auth safety model.
- `docs/API_SECURITY.md` documents API assets, trust boundaries, deployment safeguards, and deferred production requirements.
- The release worktree is clean, all intended files are committed, and
  `docs/API_SECURITY.md` is tracked before building or tagging.
- Network launcher tests require `--allow-network-access` for wildcard and
  non-loopback binds before Streamlit or Uvicorn imports.
- API tests verify `--server-stages-only`, `--no-docs`, additive deployment
  health metadata, and the absence of permissive wildcard CORS headers.
- API stage lifecycle tests cover create, recover, list/filter, explicit
  renewal, discard, expiration, metadata-only tombstones, early capacity
  rejection, compatibility retirement, exactly-once append, stale/ambiguous
  outcomes, campaign-kind round trips, recovery guidance, and process restart
  loss.
- Cross-process log tests cover stale fingerprints, lock timeouts, canonical
  path locking, process-stable lock roots, and serialized same-process and
  cross-process appends without expected fingerprints. Atomic-write tests also
  cover post-write rollback and preservation of existing log file modes.
- No tracked caches, working logs, latest-suggestion CSVs, notebook outputs, or runtime reports are present.
- `docs/QLOGNEHVI_FEASIBILITY.md` states that qLogNEHVI is implemented only
  for coupled `2 <= m <= 4` objective campaigns and that cost, replicates,
  structured stages, context, multi-fidelity, decoupled, and asynchronous
  qLogNEHVI remain deferred.
- `configs/19_multi_objective_qlognehvi.yaml` and
  `examples/19_multi_objective_qlognehvi_campaign_log.csv` validate and are
  included in the source distribution.
- `configs/20_contextual_cost_review_logei.yaml`,
  `examples/20_contextual_cost_review_campaign_log.csv`, and
  `notebooks/20_contextual_cost_review_logei_campaign.ipynb` validate and are
  included in the source distribution.
- `configs/21_contextual_replicate_logei.yaml` and
  `examples/21_contextual_replicate_campaign_log.csv` validate and are included
  in the source distribution.
- `configs/22_discrete_multi_fidelity_qmfkg.yaml`,
  `examples/22_discrete_multi_fidelity_qmfkg_campaign_log.csv`, and
  `notebooks/22_discrete_multi_fidelity_qmfkg_campaign.ipynb` validate and are
  included in the source distribution.
- Notebook 22 remains output-free, targets 15 observed rows, and demonstrates
  fidelity summary, coverage, diagnostics, progress, and batched observation
  updates.
- The v2.4.3 behavior-freeze matrix covers continuous/discrete initial and
  mocked model-based qMFKG batches for `q=1`, `q=2`, and `q=4`.

## 📦 Build

```bash
rm -rf dist/ build/
./.venv/bin/python -m build
./.venv/bin/python -m twine check dist/*
```

`dist/` and `build/` are generated release artifacts and should remain ignored.

Package-data boundary:

- the wheel should contain only the importable `bo_forge` and `bo_forge_app` packages, entrypoint metadata, and license metadata;
- release docs, configs, examples, notebooks, and tests belong in the sdist, not in the wheel;
- ignored working logs, latest-suggestion CSVs, and runtime reports should not appear in either artifact.

`twine check` is the README/PyPI rendering gate.

The normal test suite uses the already-installed development dependencies for
fast artifact-boundary checks. Before publishing, run the same distribution
test in isolated environments with normal dependency resolution and `pip check`:

```bash
BO_FORGE_RELEASE_RESOLVE_DEPENDENCIES=1 ./.venv/bin/python -m pytest \
  tests/test_release_readiness.py::test_built_distributions_install_from_outside_source_tree \
  -p no:cacheprovider
```

## 🧪 Fresh Core Wheel Smoke

Run the core wheel check outside the source checkout:

```bash
python3 -m venv /tmp/bo_forge_release_probe
/tmp/bo_forge_release_probe/bin/pip install dist/bo_forge-2.5.3-py3-none-any.whl
cd /tmp
/tmp/bo_forge_release_probe/bin/python -c "import bo_forge, bo_forge_app; print(bo_forge.__version__)"
/tmp/bo_forge_release_probe/bin/python -m bo_forge --version
/tmp/bo_forge_release_probe/bin/bo-forge --version
/tmp/bo_forge_release_probe/bin/bo-forge doctor
/tmp/bo_forge_release_probe/bin/pip check
```

`doctor` should work without importing optional Streamlit app dependencies.

## 🖥️ Fresh App Extra Smoke

Test the app extra separately:

```bash
python3 -m venv /tmp/bo_forge_app_release_probe
/tmp/bo_forge_app_release_probe/bin/pip install "dist/bo_forge-2.5.3-py3-none-any.whl[app]"
cd /tmp
/tmp/bo_forge_app_release_probe/bin/python -c "import bo_forge_app, streamlit"
/tmp/bo_forge_app_release_probe/bin/python -c "from bo_forge_app.cli import packaged_streamlit_app_path; print(packaged_streamlit_app_path())"
/tmp/bo_forge_app_release_probe/bin/python -m bo_forge_app --help
/tmp/bo_forge_app_release_probe/bin/bo-forge-app --help
/tmp/bo_forge_app_release_probe/bin/pip check
```

For `bo-forge-app`, it is enough to confirm the command resolves the packaged `bo_forge_app/streamlit_app.py`; manual browser use can happen from a normal development environment.

## 🧪 Fresh API Extra Smoke

Test the experimental API extra separately:

```bash
python3 -m venv /tmp/bo_forge_api_release_probe
/tmp/bo_forge_api_release_probe/bin/pip install "dist/bo_forge-2.5.3-py3-none-any.whl[api]"
cd /tmp
/tmp/bo_forge_api_release_probe/bin/python -c "import bo_forge_app.api"
/tmp/bo_forge_api_release_probe/bin/bo-forge-api --help
/tmp/bo_forge_api_release_probe/bin/pip check
```

## 📦 Fresh Sdist Smoke

Install the source distribution outside the source checkout:

```bash
python3 -m venv /tmp/bo_forge_sdist_release_probe
/tmp/bo_forge_sdist_release_probe/bin/pip install dist/bo_forge-2.5.3.tar.gz
cd /tmp
/tmp/bo_forge_sdist_release_probe/bin/python -c "import bo_forge, bo_forge_app; print(bo_forge.__version__)"
/tmp/bo_forge_sdist_release_probe/bin/python -m bo_forge --version
/tmp/bo_forge_sdist_release_probe/bin/bo-forge doctor
/tmp/bo_forge_sdist_release_probe/bin/pip check
```

Also extract the sdist and run at least one fixture-dependent test so packaged
test support such as `tests/conftest.py` is verified, not only listed.

## 🖥️ Manual App Smoke

Run:

```bash
bo-forge-app
python -m bo_forge_app --help
bo-forge-api --help
```

Confirm `--stage-ttl-seconds` and `--max-staged-batches` are documented in API
help; `/campaign/stages` lists metadata without payloads; explicit renewal does
not change suggestions; and `/health` reports capacity, ages, retained terminal
tombstone counts, and restart-reset lifecycle totals without reading campaign files.

Optional trusted-LAN smoke from a controlled network:

```bash
bo-forge-app --host 0.0.0.0 --port 8501 --no-browser --allow-network-access
```

Confirm the startup output warns that the app has no built-in authentication,
is for trusted LAN/VPN/SSH tunnel use only, and reads/writes host files. Do not
expose the app directly to the public internet.

Review [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md) and
[API_PROBE.md](API_PROBE.md) and [API_SECURITY.md](API_SECURITY.md) before release and
confirm the trusted-LAN manual smoke matches the documented safety guidance.

Optional macOS launcher smoke:

```bash
bo-forge-app --make-launcher ~/Desktop/BO-Forge.command
```

Confirm the full local loop still works:

- create or load a campaign;
- create a `Campaign kind = Multi-fidelity qMFKG` campaign and confirm the
  generated YAML contains `fidelity:` and `bo.acquisition: qmf_kg`; repeat with
  ordered discrete levels and batch size 2; verify optimizer max iterations,
  the opt-in timeout, and timeout omission when its toggle is off;
- create a `Campaign kind = Contextual LogEI` campaign and confirm the
  generated YAML contains `context:` and `bo.acquisition: log_ei`;
- create a supported single-objective campaign with `model.profile: smooth` and
  confirm Model Summary, Model Diagnostics, and Model Comparison are available;
- open `notebooks/17_model_profile_logei_campaign.ipynb` or inspect its source
  to confirm it uses the current model-profile config and seed log;
- load `configs/16_contextual_logei.yaml`, generate a dry-run contextual
  suggestion with `feedstock_acidity`, and confirm append is blocked if the
  context value changes after staging;
- load `configs/20_contextual_cost_review_logei.yaml`, generate a contextual
  cost-aware dry-run suggestion, review/observe it with `actual_cost`, and
  confirm Context Summary, Cost Summary, Context Diagnostics, and Cost Progress
  remain available;
- load `configs/21_contextual_replicate_logei.yaml`, request
  `feedstock_acidity=0.25`, and confirm active repeats stay within that context
  while Context Summary, Replicate Summary, and both diagnostics remain available;
- generate staged suggestions;
- append;
- mark observed;
- export report;
- export plot.

## 🏷️ GitHub Release

- Final closeout: confirm `ROADMAP_V1_X.md` remains completed history,
  `ROADMAP_V2_X.md` is the active roadmap, and `README.md`, `CHANGELOG.md`,
  install paths, and the release tag all agree on `v2.5.3`.
- Tag the release as `v2.5.3`.
- Use `CHANGELOG.md` and the final release note as the release description.
- Attach built distributions only if needed.

## 📤 PyPI

After all checks pass:

```bash
./.venv/bin/python -m twine upload dist/*
```

Uploading is a manual release action and is not part of normal implementation work.
