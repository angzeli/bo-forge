import os
import re
import subprocess
import sys
import sysconfig
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

import bo_forge
from bo_forge.session import CampaignSession
from bo_forge_app.cli import packaged_streamlit_app_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_license_file_exists() -> None:
    assert (PROJECT_ROOT / "LICENSE").is_file()


def test_no_duplicate_release_artifacts_in_worktree() -> None:
    duplicate_artifacts = [
        path
        for directory in ["configs", "examples", "notebooks"]
        for path in (PROJECT_ROOT / directory).glob("* 2.*")
    ]

    assert duplicate_artifacts == []


def test_readme_contains_current_install_commands() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "pip install bo-forge" in readme
    assert 'pip install "bo-forge[app]"' in readme
    assert 'pip install "bo-forge[api]"' in readme
    assert "bo-forge --version" in readme
    assert "bo-forge-app" in readme
    assert "bo-forge-api" in readme
    assert "python -m bo_forge_app" in readme
    assert "docs/STREAMLIT_DEPLOYMENT.md" in readme
    assert "docs/API_PROBE.md" in readme
    assert "docs/INSTALLATION.md" in readme


def test_streamlit_deployment_guide_exists_and_covers_safety_model() -> None:
    guide = (PROJECT_ROOT / "docs" / "STREAMLIT_DEPLOYMENT.md").read_text(
        encoding="utf-8"
    )

    required_phrases = [
        "no built-in auth",
        "trusted LAN",
        "VPN",
        "SSH tunnel",
        "no safe unauthenticated public internet exposure",
        "host filesystem access",
        "Back up CSV logs",
        "Avoid simultaneous writes",
        "dedicated campaign working directory",
    ]
    for phrase in required_phrases:
        assert phrase in guide


def test_core_docs_link_streamlit_deployment_guide() -> None:
    docs = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs" / "STREAMLIT_APP.md",
        PROJECT_ROOT / "docs" / "INSTALLATION.md",
        PROJECT_ROOT / "docs" / "QUICKSTART.md",
        PROJECT_ROOT / "docs" / "RELEASE_CHECKLIST.md",
    ]

    for path in docs:
        assert "STREAMLIT_DEPLOYMENT.md" in path.read_text(encoding="utf-8")


def test_api_probe_guide_exists_and_covers_safety_model() -> None:
    guide = (PROJECT_ROOT / "docs" / "API_PROBE.md").read_text(encoding="utf-8")

    required_phrases = [
        "experimental",
        "not a stable public API",
        'pip install "bo-forge[api]"',
        "bo-forge-api --root . --host 127.0.0.1 --port 8765",
        "root-bound",
        "no built-in auth",
        "trusted LAN",
        "SSH tunnel",
        "Do not expose it directly to the public internet",
        "Streamlit remains the recommended local UI",
    ]
    for phrase in required_phrases:
        assert phrase in guide


def test_v1_3_roadmap_line_is_active_after_v1_2_closeout() -> None:
    roadmap = (PROJECT_ROOT / "ROADMAP_V1_X.md").read_text(encoding="utf-8")

    assert "Current baseline: `v1.3.2`" in roadmap
    assert "explicit stage-aware backend/session/CLI suggestions" in roadmap
    assert "read-only stage reports and diagnostics" in roadmap
    assert (
        "`v1.3.2` | Minor | Read-only stage summaries, report sections, "
        "CLI inspection, and stage diagnostics"
    ) in roadmap
    assert re.search(
        r"## 🏗️ v1\.2 - App Launcher And Access Path\s+Status: completed",
        roadmap,
    )
    assert re.search(r"## 🧩 v1\.3 - Structured Campaigns\s+Status: active", roadmap)


def test_streamlit_service_layer_is_documented_as_internal_non_http() -> None:
    repository_structure = (PROJECT_ROOT / "docs" / "REPOSITORY_STRUCTURE.md").read_text(
        encoding="utf-8"
    )
    streamlit_app_docs = (PROJECT_ROOT / "docs" / "STREAMLIT_APP.md").read_text(
        encoding="utf-8"
    )
    public_api = (PROJECT_ROOT / "docs" / "PUBLIC_API.md").read_text(encoding="utf-8")

    assert "bo_forge_app/service.py" in repository_structure
    assert "internal, non-HTTP app service layer" in repository_structure
    assert "not a stable public API" in repository_structure
    assert "internal non-HTTP service layer" in streamlit_app_docs
    assert "CampaignAppService" not in public_api
    assert "bo_forge_app.api" not in public_api


def test_release_checklist_includes_fresh_install_pip_check() -> None:
    checklist = (PROJECT_ROOT / "docs" / "RELEASE_CHECKLIST.md").read_text(encoding="utf-8")

    assert "/tmp/bo_forge_release_probe/bin/pip check" in checklist
    assert "/tmp/bo_forge_app_release_probe/bin/pip check" in checklist
    assert "/tmp/bo_forge_api_release_probe/bin/pip check" in checklist
    assert "/tmp/bo_forge_sdist_release_probe/bin/pip check" in checklist


def test_installation_tutorial_covers_pip_install_paths() -> None:
    tutorial = (PROJECT_ROOT / "docs" / "INSTALLATION.md").read_text(encoding="utf-8")

    assert "pip install bo-forge" in tutorial
    assert 'pip install "bo-forge[app]"' in tutorial
    assert 'pip install "bo-forge[api]"' in tutorial
    assert 'pip install -e ".[dev]"' in tutorial
    assert "dist/bo_forge-1.3.2-py3-none-any.whl" in tutorial
    assert "dist/bo_forge-1.3.2.tar.gz" in tutorial
    assert "pip check" in tutorial


def test_quickstart_has_no_stale_v0_4_current_feature_wording() -> None:
    quickstart = (PROJECT_ROOT / "docs" / "QUICKSTART.md").read_text(encoding="utf-8")

    stale_phrases = [
        "v0.4.3 adds optional deterministic cost",
        "v0.4.3 uses greedy",
        "v0.4.4 adds optional explicit replicate",
    ]
    for phrase in stale_phrases:
        assert phrase not in quickstart


def test_structured_stage_docs_use_working_log_suggestion_flow() -> None:
    cli_docs = (PROJECT_ROOT / "docs" / "CLI.md").read_text(encoding="utf-8")
    quickstart = (PROJECT_ROOT / "docs" / "QUICKSTART.md").read_text(encoding="utf-8")
    csv_schema = (PROJECT_ROOT / "docs" / "CSV_SCHEMA.md").read_text(encoding="utf-8")
    repository_structure = (PROJECT_ROOT / "docs" / "REPOSITORY_STRUCTURE.md").read_text(
        encoding="utf-8"
    )

    for content in (cli_docs, quickstart):
        assert "bo-forge init-log" in content
        assert "13_structured_campaign_core_working_log.csv" in content
        assert "--stage screen" in content
        assert "stage-summary" in content
        assert "stage-diagnostics" in content
    assert "manually staged rows" not in quickstart
    assert "manually staged rows" not in repository_structure
    normalized_csv_schema = " ".join(csv_schema.split())
    assert "In v1.3.2, `stages:` cannot be combined with `cost:`." in normalized_csv_schema
    assert "source,[stage],review_status" not in csv_schema


def test_app_created_campaign_tutorial_uses_current_streamlit_labels() -> None:
    tutorial = (PROJECT_ROOT / "docs" / "09_APP_CREATED_CAMPAIGN_TUTORIAL.md").read_text(
        encoding="utf-8"
    )

    assert "Campaign file action" in tutorial
    assert "Create Campaign" in tutorial
    assert "Update YAML preview from form" in tutorial
    assert "Campaign Files" not in tutorial
    assert "Create Campaign tab" not in tutorial
    assert "Regenerate YAML from structured fields" not in tutorial


def test_cost_aware_multi_objective_notebook_uses_current_version_wording() -> None:
    notebook_text = (
        PROJECT_ROOT / "notebooks" / "12_cost_aware_multi_objective_qlogehvi_campaign.ipynb"
    ).read_text(encoding="utf-8")

    assert "v1.1 backend workflow" in notebook_text
    assert "v1.1.3 backend workflow" not in notebook_text


def test_replicate_ready_cli_demo_exercises_repeat_path() -> None:
    quickstart = (PROJECT_ROOT / "docs" / "QUICKSTART.md").read_text(encoding="utf-8")

    assert "/tmp/bo_forge_08_repeat_ready.csv" in quickstart
    assert "uncertain_best" in quickstart
    assert "rep_seed_3a" in quickstart
    assert "configs/08_replicate_aware_logei.yaml" in quickstart


def test_replicate_ready_demo_executes_repeat_path(tmp_path: Path) -> None:
    config_path = PROJECT_ROOT / "configs" / "08_replicate_aware_logei.yaml"
    seed_log_path = PROJECT_ROOT / "examples" / "08_replicate_aware_campaign_log.csv"
    working_log_path = tmp_path / "bo_forge_08_repeat_ready.csv"
    working_log_path.write_bytes(seed_log_path.read_bytes())
    df = pd.read_csv(working_log_path, keep_default_na=False)
    df.loc[len(df)] = [
        "rep_seed_3a",
        3,
        "observed",
        "manual",
        "rep_3",
        0,
        0.85,
        430,
        1.10,
        "",
        "",
        "",
    ]
    df.to_csv(working_log_path, index=False)

    campaign = CampaignSession.from_files(config_path, working_log_path)
    suggestions = campaign.suggest_next(batch_size=3)

    existing_groups = set(df["replicate_group"].astype(str))
    repeat_rows = suggestions[
        suggestions["replicate_group"].astype(str).isin(existing_groups)
    ]
    exploration_rows = suggestions[
        ~suggestions["replicate_group"].astype(str).isin(existing_groups)
    ]

    assert len(suggestions) == 3
    assert not repeat_rows.empty
    for group, group_suggestions in repeat_rows.groupby("replicate_group"):
        existing_indexes = df.loc[df["replicate_group"] == group, "replicate_index"].astype(
            int
        )
        expected_indexes = list(
            range(
                int(existing_indexes.max()) + 1,
                int(existing_indexes.max()) + 1 + len(group_suggestions),
            )
        )
        assert sorted(group_suggestions["replicate_index"].astype(int).tolist()) == (
            expected_indexes
        )
    assert len(exploration_rows) == 1
    exploration = exploration_rows.iloc[0]
    assert exploration["replicate_group"] == exploration["row_id"]
    assert int(exploration["replicate_index"]) == 0


def test_public_api_exports_are_importable() -> None:
    public_api = (PROJECT_ROOT / "docs" / "PUBLIC_API.md").read_text(encoding="utf-8")
    section = public_api.split("## ✅ Public Package Exports", maxsplit=1)[1].split(
        "## ",
        maxsplit=1,
    )[0]
    exports = re.findall(r"^- `([^`]+)`", section, flags=re.MULTILINE)

    assert exports
    for export in exports:
        assert hasattr(bo_forge, export), f"Missing public export from bo_forge: {export}"


def test_app_console_entrypoint_resolves_packaged_script() -> None:
    app_path = packaged_streamlit_app_path()

    assert app_path.name == "streamlit_app.py"
    assert app_path.is_file()
    assert app_path.parent.name == "bo_forge_app"


def test_built_distributions_install_from_outside_source_tree(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(dist_dir),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        text=True,
    )
    wheels = sorted(dist_dir.glob("bo_forge-1.3.2-*.whl"))
    sdists = sorted(dist_dir.glob("bo_forge-1.3.2.tar.gz"))
    assert wheels, "No v1.3.2 wheel was built."
    assert sdists, "No v1.3.2 sdist was built."

    _assert_wheel_package_boundaries(wheels[0])
    _assert_sdist_contains_release_assets(sdists[0])
    subprocess.run(
        [sys.executable, "-m", "twine", "check", str(wheels[0]), str(sdists[0])],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        text=True,
    )
    _install_distribution_and_probe(
        artifact=wheels[0],
        probe_root=tmp_path / "wheel_probe",
        env=env,
        install_args=[],
    )
    _install_core_only_app_missing_streamlit_probe(
        wheel=wheels[0],
        probe_root=tmp_path / "core_app_probe",
        env=env,
    )
    _install_app_extra_and_probe(
        wheel=wheels[0],
        probe_root=tmp_path / "app_probe",
        env=env,
    )
    _install_api_extra_and_probe(
        wheel=wheels[0],
        probe_root=tmp_path / "api_probe",
        env=env,
    )
    _install_distribution_and_probe(
        artifact=sdists[0],
        probe_root=tmp_path / "sdist_probe",
        env=env,
        install_args=["--no-build-isolation"],
    )


def _assert_wheel_package_boundaries(wheel_path: Path) -> None:
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        metadata = wheel.read("bo_forge-1.3.2.dist-info/METADATA").decode("utf-8")

    assert "bo_forge/__init__.py" in names
    assert "bo_forge/structured.py" in names
    assert "bo_forge_app/streamlit_app.py" in names
    assert "bo_forge_app/cli.py" in names
    assert "bo_forge_app/service.py" in names
    assert "bo_forge_app/api.py" in names
    assert "bo_forge_app/api_cli.py" in names
    assert "bo_forge_app/__main__.py" in names
    assert "bo_forge-1.3.2.dist-info/entry_points.txt" in names
    assert "bo_forge-1.3.2.dist-info/licenses/LICENSE" in names
    excluded_prefixes = ("docs/", "configs/", "examples/", "notebooks/", "tests/")
    assert not any(name.startswith(excluded_prefixes) for name in names)
    assert "Provides-Extra: app" in metadata
    assert "Provides-Extra: api" in metadata
    assert 'Requires-Dist: streamlit>=1.57; extra == "app"' in metadata
    assert 'Requires-Dist: fastapi>=0.115; extra == "api"' in metadata
    assert 'Requires-Dist: uvicorn>=0.30; extra == "api"' in metadata
    assert 'Requires-Dist: streamlit>=1.57\n' not in metadata
    assert 'Requires-Dist: fastapi>=0.115\n' not in metadata


def _assert_sdist_contains_release_assets(sdist_path: Path) -> None:
    with tarfile.open(sdist_path) as sdist:
        names = set(sdist.getnames())

    assert "bo_forge-1.3.2/README.md" in names
    assert "bo_forge-1.3.2/ROADMAP_V0_TO_V1.md" in names
    assert "bo_forge-1.3.2/ROADMAP_V1_X.md" in names
    assert "bo_forge-1.3.2/docs/PUBLIC_API.md" in names
    assert "bo_forge-1.3.2/docs/STREAMLIT_DEPLOYMENT.md" in names
    assert "bo_forge-1.3.2/docs/API_PROBE.md" in names
    assert "bo_forge-1.3.2/examples/quickstart.py" in names
    assert "bo_forge-1.3.2/examples/01_simple_2d_maximise_logei_campaign_log.csv" in names
    assert (
        "bo_forge-1.3.2/examples/10_multi_objective_mixed_constrained_campaign_log.csv"
        in names
    )
    assert (
        "bo_forge-1.3.2/examples/11_four_objective_mixed_constrained_campaign_log.csv"
        in names
    )
    assert "bo_forge-1.3.2/examples/12_cost_aware_multi_objective_campaign_log.csv" in names
    assert "bo_forge-1.3.2/examples/13_structured_campaign_core_campaign_log.csv" in names
    assert "bo_forge-1.3.2/configs/10_multi_objective_mixed_constrained_qlogehvi.yaml" in names
    assert "bo_forge-1.3.2/configs/11_four_objective_mixed_constrained_qlogehvi.yaml" in names
    assert "bo_forge-1.3.2/configs/12_cost_aware_multi_objective_qlogehvi.yaml" in names
    assert "bo_forge-1.3.2/configs/13_structured_campaign_core.yaml" in names
    assert "bo_forge-1.3.2/notebooks/01_maximisation_logei_campaign.ipynb" in names
    assert "bo_forge-1.3.2/notebooks/10_multi_objective_qlogehvi_campaign.ipynb" in names
    assert "bo_forge-1.3.2/notebooks/11_four_objective_qlogehvi_campaign.ipynb" in names
    assert "bo_forge-1.3.2/notebooks/12_cost_aware_multi_objective_qlogehvi_campaign.ipynb" in names
    assert not any("working_log" in name or "latest_suggestions" in name for name in names)


def _install_distribution_and_probe(
    *,
    artifact: Path,
    probe_root: Path,
    env: dict[str, str],
    install_args: list[str],
) -> None:
    venv_dir = probe_root / "venv"
    probe_dir = probe_root / "probe"
    probe_dir.mkdir(parents=True)

    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
        env=env,
        check=True,
        text=True,
    )
    python = venv_dir / "bin" / "python"
    pip = venv_dir / "bin" / "pip"
    install_env = env
    if "--no-build-isolation" in install_args:
        install_env = dict(env)
        install_env["PYTHONPATH"] = sysconfig.get_paths()["purelib"]
    subprocess.run(
        [str(pip), "install", "--no-deps", *install_args, str(artifact)],
        cwd=probe_dir,
        env=install_env,
        check=True,
        text=True,
    )
    completed = subprocess.run(
        [str(venv_dir / "bin" / "bo-forge"), "--version"],
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    assert completed.stdout == "bo-forge 1.3.2\n"
    subprocess.run(
        [str(venv_dir / "bin" / "bo-forge-api"), "--help"],
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    source_root = str(PROJECT_ROOT.resolve())
    script = f"""
import builtins
from pathlib import Path
from importlib.metadata import entry_points
import bo_forge
import bo_forge_app

source_root = Path({source_root!r})
scripts = {{ep.name: ep.value for ep in entry_points(group="console_scripts")}}
assert scripts["bo-forge-app"] == "bo_forge_app.cli:main"
assert scripts["bo-forge-api"] == "bo_forge_app.api_cli:main"
for module in (bo_forge, bo_forge_app):
    module_path = Path(module.__file__).resolve()
    assert source_root not in module_path.parents, module_path
assert bo_forge.__version__ == "1.3.2"

real_import = builtins.__import__
def block_optional_app_deps(name, *args, **kwargs):
    if name == "streamlit" or name.startswith("streamlit."):
        raise AssertionError("doctor imported optional Streamlit dependencies")
    if name in {{"fastapi", "uvicorn"}} or name.startswith(("fastapi.", "uvicorn.")):
        raise AssertionError("doctor imported optional API dependencies")
    return real_import(name, *args, **kwargs)
builtins.__import__ = block_optional_app_deps
from bo_forge.cli import run
assert run(["doctor"]) == 0
"""
    subprocess.run(
        [str(python), "-c", script],
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
    )


def _install_app_extra_and_probe(
    *,
    wheel: Path,
    probe_root: Path,
    env: dict[str, str],
) -> None:
    venv_dir = probe_root / "venv"
    probe_dir = probe_root / "probe"
    probe_dir.mkdir(parents=True)
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
        env=env,
        check=True,
        text=True,
    )
    python = venv_dir / "bin" / "python"
    pip = venv_dir / "bin" / "pip"
    subprocess.run(
        [str(pip), "install", "--no-deps", f"{wheel}[app]"],
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
    )
    source_root = str(PROJECT_ROOT.resolve())
    script = f"""
from pathlib import Path
import streamlit
from bo_forge_app.cli import packaged_streamlit_app_path

source_root = Path({source_root!r})
app_path = packaged_streamlit_app_path()
assert app_path.name == "streamlit_app.py"
assert source_root not in app_path.resolve().parents, app_path
assert streamlit.__version__
"""
    subprocess.run(
        [str(python), "-c", script],
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
    )
    subprocess.run(
        [str(python), "-m", "bo_forge_app", "--help"],
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [str(venv_dir / "bin" / "bo-forge-app"), "--help"],
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


def _install_api_extra_and_probe(
    *,
    wheel: Path,
    probe_root: Path,
    env: dict[str, str],
) -> None:
    venv_dir = probe_root / "venv"
    probe_dir = probe_root / "probe"
    probe_dir.mkdir(parents=True)
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
        env=env,
        check=True,
        text=True,
    )
    python = venv_dir / "bin" / "python"
    pip = venv_dir / "bin" / "pip"
    subprocess.run(
        [str(pip), "install", "--no-deps", f"{wheel}[api]"],
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
    )
    source_root = str(PROJECT_ROOT.resolve())
    probe_env = dict(env)
    probe_env["PYTHONPATH"] = sysconfig.get_paths()["purelib"]
    script = f"""
from pathlib import Path
import fastapi
import uvicorn
import bo_forge_app.api

source_root = Path({source_root!r})
api_path = Path(bo_forge_app.api.__file__).resolve()
assert source_root not in api_path.parents, api_path
assert fastapi.__version__
assert uvicorn.__version__
"""
    subprocess.run(
        [str(python), "-c", script],
        cwd=probe_dir,
        env=probe_env,
        check=True,
        text=True,
    )
    subprocess.run(
        [str(venv_dir / "bin" / "bo-forge-api"), "--help"],
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


def _install_core_only_app_missing_streamlit_probe(
    *,
    wheel: Path,
    probe_root: Path,
    env: dict[str, str],
) -> None:
    venv_dir = probe_root / "venv"
    probe_dir = probe_root / "probe"
    probe_dir.mkdir(parents=True)
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        env=env,
        check=True,
        text=True,
    )
    subprocess.run(
        [str(venv_dir / "bin" / "pip"), "install", "--no-deps", str(wheel)],
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
    )
    for command in [
        [str(venv_dir / "bin" / "bo-forge-app")],
        [str(venv_dir / "bin" / "python"), "-m", "bo_forge_app"],
    ]:
        completed = subprocess.run(
            command,
            cwd=probe_dir,
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )

        assert completed.returncode == 1
        assert 'pip install "bo-forge[app]"' in completed.stderr
        assert "Traceback" not in completed.stderr
    completed = subprocess.run(
        [str(venv_dir / "bin" / "bo-forge-api")],
        cwd=probe_dir,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    assert 'pip install "bo-forge[api]"' in completed.stderr
    assert "Traceback" not in completed.stderr
