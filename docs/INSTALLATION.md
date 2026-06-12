# 🧰 BO Forge Installation Tutorial

This page shows the recommended `pip install` paths for BO Forge v1.3.0.

Use the normal package install when you want BO Forge as a command-line or Python package. Use the app extra when you also want the local Streamlit workbench. Use the API extra only when you want the experimental FastAPI probe.

## ✅ Option 1: Core Package And CLI

Create and activate a fresh environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install BO Forge:

```bash
pip install bo-forge
```

Check the install:

```bash
bo-forge --version
bo-forge doctor
python -m bo_forge --version
pip check
```

This installs the backend package, `CampaignSession`, and the `bo-forge` CLI.

## 🖥️ Option 2: Core Package Plus Streamlit App

If you want the local app as well, install the app extra:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "bo-forge[app]"
```

Check the install:

```bash
bo-forge --version
bo-forge doctor
bo-forge-app
python -m bo_forge_app
pip check
```

`bo-forge-app` launches the packaged Streamlit workbench. `python -m bo_forge_app`
uses the same launcher path when console scripts are less visible.

For trusted LAN access:

```bash
bo-forge-app --host 0.0.0.0 --port 8501
```

BO Forge has no built-in authentication. Read
[STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md) before sharing the app beyond
one local machine.

On macOS, create an optional double-click launcher:

```bash
bo-forge-app --make-launcher ~/Desktop/BO-Forge.command
```

## 🧪 Option 3: Experimental API Probe

If you want the optional FastAPI probe, install the API extra:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "bo-forge[api]"
```

Check the launcher:

```bash
bo-forge-api --help
```

Launch local-only:

```bash
bo-forge-api --root . --host 127.0.0.1 --port 8765
```

The API probe is experimental, has no built-in authentication, and is not a
production backend. Read [API_PROBE.md](API_PROBE.md) before exposing it beyond
localhost.

## 🛠️ Option 4: Development From A Clone

From the repository root:

```bash
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev]"
```

Run the standard development checks:

```bash
./.venv/bin/pytest
./.venv/bin/ruff check .
./.venv/bin/python examples/quickstart.py
./.venv/bin/python -m bo_forge doctor
```

Launch the app from a source checkout:

```bash
./.venv/bin/bo-forge-app
```

The raw Streamlit command remains a development fallback:

```bash
./.venv/bin/python -m streamlit run bo_forge_app/streamlit_app.py
```

## 🧪 Install From Local Release Artifacts

After building local release artifacts:

```bash
./.venv/bin/python -m build
```

Install the wheel in a fresh environment:

```bash
python3 -m venv /tmp/bo_forge_probe
/tmp/bo_forge_probe/bin/pip install dist/bo_forge-1.3.0-py3-none-any.whl
/tmp/bo_forge_probe/bin/bo-forge doctor
/tmp/bo_forge_probe/bin/pip check
```

Install the source distribution similarly:

```bash
python3 -m venv /tmp/bo_forge_sdist_probe
/tmp/bo_forge_sdist_probe/bin/pip install dist/bo_forge-1.3.0.tar.gz
/tmp/bo_forge_sdist_probe/bin/bo-forge doctor
/tmp/bo_forge_sdist_probe/bin/pip check
```

The release checklist in [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) contains the full GitHub/PyPI release smoke checks.

## 🧯 Common Install Checks

If a command is not found, confirm you are using the environment where BO Forge was installed:

```bash
which python
which pip
which bo-forge
```

If notebook or editable environments are confusing command lookup, use module invocation:

```bash
python -m bo_forge --version
python -m bo_forge doctor
```

If `pip check` reports dependency conflicts, fix those before using the environment for release validation.

If `bo-forge-app` reports that Streamlit is not installed, install the app extra with `pip install "bo-forge[app]"`.
