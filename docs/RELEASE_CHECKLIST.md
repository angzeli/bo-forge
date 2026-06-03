# 🚢 BO Forge Release Checklist

Use this checklist before publishing a GitHub release or PyPI package.

## ✅ Preflight

```bash
./.venv/bin/pytest
./.venv/bin/ruff check .
./.venv/bin/python examples/quickstart.py
./.venv/bin/python -m bo_forge doctor
./.venv/bin/python -m streamlit --version
git diff --check
```

Confirm:

- `LICENSE` exists.
- `README.md` includes install commands for the core package, app extra, `bo-forge`, and `bo-forge-app`.
- No tracked caches, working logs, latest-suggestion CSVs, notebook outputs, or runtime reports are present.

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

## 🧪 Fresh Core Wheel Smoke

Run the core wheel check outside the source checkout:

```bash
python3 -m venv /tmp/bo_forge_release_probe
/tmp/bo_forge_release_probe/bin/pip install dist/bo_forge-1.1.3-py3-none-any.whl
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
/tmp/bo_forge_app_release_probe/bin/pip install "dist/bo_forge-1.1.3-py3-none-any.whl[app]"
cd /tmp
/tmp/bo_forge_app_release_probe/bin/python -c "import bo_forge_app, streamlit"
/tmp/bo_forge_app_release_probe/bin/python -c "from bo_forge_app.cli import packaged_streamlit_app_path; print(packaged_streamlit_app_path())"
/tmp/bo_forge_app_release_probe/bin/pip check
```

For `bo-forge-app`, it is enough to confirm the command resolves the packaged `bo_forge_app/streamlit_app.py`; manual browser use can happen from a normal development environment.

## 📦 Fresh Sdist Smoke

Install the source distribution outside the source checkout:

```bash
python3 -m venv /tmp/bo_forge_sdist_release_probe
/tmp/bo_forge_sdist_release_probe/bin/pip install dist/bo_forge-1.1.3.tar.gz
cd /tmp
/tmp/bo_forge_sdist_release_probe/bin/python -c "import bo_forge, bo_forge_app; print(bo_forge.__version__)"
/tmp/bo_forge_sdist_release_probe/bin/python -m bo_forge --version
/tmp/bo_forge_sdist_release_probe/bin/bo-forge doctor
/tmp/bo_forge_sdist_release_probe/bin/pip check
```

## 🖥️ Manual App Smoke

Run:

```bash
bo-forge-app
```

Confirm the full local loop still works:

- create or load a campaign;
- generate staged suggestions;
- append;
- mark observed;
- export report;
- export plot.

## 🏷️ GitHub Release

- Tag the release as `v1.1.3`.
- Use `CHANGELOG.md` and the final release note as the release description.
- Attach built distributions only if needed.

## 📤 PyPI

After all checks pass:

```bash
./.venv/bin/python -m twine upload dist/*
```

Uploading is a manual release action and is not part of normal implementation work.
