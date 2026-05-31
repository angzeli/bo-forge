import os
import re
import subprocess
import sys
import sysconfig
import tarfile
import zipfile
from pathlib import Path

import bo_forge
from bo_forge_app.cli import packaged_streamlit_app_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_license_file_exists() -> None:
    assert (PROJECT_ROOT / "LICENSE").is_file()


def test_readme_contains_current_install_commands() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "pip install bo-forge" in readme
    assert 'pip install "bo-forge[app]"' in readme
    assert "bo-forge --version" in readme
    assert "bo-forge-app" in readme
    assert "docs/INSTALLATION.md" in readme


def test_release_checklist_includes_fresh_install_pip_check() -> None:
    checklist = (PROJECT_ROOT / "docs" / "RELEASE_CHECKLIST.md").read_text(encoding="utf-8")

    assert "/tmp/bo_forge_release_probe/bin/pip check" in checklist
    assert "/tmp/bo_forge_app_release_probe/bin/pip check" in checklist
    assert "/tmp/bo_forge_sdist_release_probe/bin/pip check" in checklist


def test_installation_tutorial_covers_pip_install_paths() -> None:
    tutorial = (PROJECT_ROOT / "docs" / "INSTALLATION.md").read_text(encoding="utf-8")

    assert "pip install bo-forge" in tutorial
    assert 'pip install "bo-forge[app]"' in tutorial
    assert 'pip install -e ".[dev]"' in tutorial
    assert "dist/bo_forge-1.1.0-py3-none-any.whl" in tutorial
    assert "dist/bo_forge-1.1.0.tar.gz" in tutorial
    assert "pip check" in tutorial


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


def test_streamlit_docs_screenshot_assets_exist() -> None:
    docs_dir = PROJECT_ROOT / "docs"
    screenshot_refs: set[str] = set()
    for doc_path in docs_dir.glob("*.md"):
        text = doc_path.read_text(encoding="utf-8")
        screenshot_refs.update(re.findall(r"\]\((assets/[^)]+\.png)\)", text))

    assert screenshot_refs
    missing = [ref for ref in sorted(screenshot_refs) if not (docs_dir / ref).is_file()]
    assert not missing, f"Missing documentation screenshot assets: {missing}"


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
    wheels = sorted(dist_dir.glob("bo_forge-1.1.0-*.whl"))
    sdists = sorted(dist_dir.glob("bo_forge-1.1.0.tar.gz"))
    assert wheels, "No v1.1.0 wheel was built."
    assert sdists, "No v1.1.0 sdist was built."

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
    _install_distribution_and_probe(
        artifact=sdists[0],
        probe_root=tmp_path / "sdist_probe",
        env=env,
        install_args=["--no-build-isolation"],
    )


def _assert_wheel_package_boundaries(wheel_path: Path) -> None:
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        metadata = wheel.read("bo_forge-1.1.0.dist-info/METADATA").decode("utf-8")

    assert "bo_forge/__init__.py" in names
    assert "bo_forge_app/streamlit_app.py" in names
    assert "bo_forge_app/cli.py" in names
    assert "bo_forge-1.1.0.dist-info/entry_points.txt" in names
    assert "bo_forge-1.1.0.dist-info/licenses/LICENSE" in names
    excluded_prefixes = ("docs/", "configs/", "examples/", "notebooks/", "tests/")
    assert not any(name.startswith(excluded_prefixes) for name in names)
    assert "Provides-Extra: app" in metadata
    assert 'Requires-Dist: streamlit>=1.57; extra == "app"' in metadata
    assert 'Requires-Dist: streamlit>=1.57\n' not in metadata


def _assert_sdist_contains_release_assets(sdist_path: Path) -> None:
    with tarfile.open(sdist_path) as sdist:
        names = set(sdist.getnames())

    assert "bo_forge-1.1.0/README.md" in names
    assert "bo_forge-1.1.0/ROADMAP_PRE_V1.md" in names
    assert "bo_forge-1.1.0/ROADMAP_AFTER_V1.md" in names
    assert "bo_forge-1.1.0/docs/PUBLIC_API.md" in names
    assert "bo_forge-1.1.0/docs/assets/streamlit_campaign_panel.png" in names
    assert "bo_forge-1.1.0/examples/quickstart.py" in names
    assert "bo_forge-1.1.0/examples/01_simple_2d_maximise_logei_campaign_log.csv" in names
    assert (
        "bo_forge-1.1.0/examples/10_multi_objective_mixed_constrained_campaign_log.csv"
        in names
    )
    assert "bo_forge-1.1.0/configs/10_multi_objective_mixed_constrained_qlogehvi.yaml" in names
    assert "bo_forge-1.1.0/notebooks/01_maximisation_logei_campaign.ipynb" in names
    assert "bo_forge-1.1.0/notebooks/10_multi_objective_qlogehvi_campaign.ipynb" in names
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
    assert completed.stdout == "bo-forge 1.1.0\n"

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
for module in (bo_forge, bo_forge_app):
    module_path = Path(module.__file__).resolve()
    assert source_root not in module_path.parents, module_path
assert bo_forge.__version__ == "1.1.0"

real_import = builtins.__import__
def block_streamlit(name, *args, **kwargs):
    if name == "streamlit" or name.startswith("streamlit."):
        raise AssertionError("doctor imported optional Streamlit dependencies")
    return real_import(name, *args, **kwargs)
builtins.__import__ = block_streamlit
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
    completed = subprocess.run(
        [str(venv_dir / "bin" / "bo-forge-app")],
        cwd=probe_dir,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    assert 'pip install "bo-forge[app]"' in completed.stderr
    assert "Traceback" not in completed.stderr
