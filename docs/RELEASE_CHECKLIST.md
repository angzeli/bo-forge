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
- `docs/STREAMLIT_DEPLOYMENT.md` describes local-only, trusted-LAN, SSH/VPN, and authenticated reverse-proxy modes.
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
/tmp/bo_forge_release_probe/bin/pip install dist/bo_forge-1.2.2-py3-none-any.whl
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
/tmp/bo_forge_app_release_probe/bin/pip install "dist/bo_forge-1.2.2-py3-none-any.whl[app]"
cd /tmp
/tmp/bo_forge_app_release_probe/bin/python -c "import bo_forge_app, streamlit"
/tmp/bo_forge_app_release_probe/bin/python -c "from bo_forge_app.cli import packaged_streamlit_app_path; print(packaged_streamlit_app_path())"
/tmp/bo_forge_app_release_probe/bin/python -m bo_forge_app --help
/tmp/bo_forge_app_release_probe/bin/bo-forge-app --help
/tmp/bo_forge_app_release_probe/bin/pip check
```

For `bo-forge-app`, it is enough to confirm the command resolves the packaged `bo_forge_app/streamlit_app.py`; manual browser use can happen from a normal development environment.

## 📦 Fresh Sdist Smoke

Install the source distribution outside the source checkout:

```bash
python3 -m venv /tmp/bo_forge_sdist_release_probe
/tmp/bo_forge_sdist_release_probe/bin/pip install dist/bo_forge-1.2.2.tar.gz
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
python -m bo_forge_app --help
```

Optional trusted-LAN smoke from a controlled network:

```bash
bo-forge-app --host 0.0.0.0 --port 8501 --no-browser
```

Confirm the startup output warns that the app has no built-in authentication,
is for trusted LAN/VPN/SSH tunnel use only, and reads/writes host files. Do not
expose the app directly to the public internet.

Review [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md) before release and
confirm the trusted-LAN manual smoke matches the documented safety guidance.

Optional macOS launcher smoke:

```bash
bo-forge-app --make-launcher ~/Desktop/BO-Forge.command
```

Confirm the full local loop still works:

- create or load a campaign;
- generate staged suggestions;
- append;
- mark observed;
- export report;
- export plot.

## 🏷️ GitHub Release

- Tag the release as `v1.2.2`.
- Use `CHANGELOG.md` and the final release note as the release description.
- Attach built distributions only if needed.

## 📤 PyPI

After all checks pass:

```bash
./.venv/bin/python -m twine upload dist/*
```

Uploading is a manual release action and is not part of normal implementation work.
