import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pandas as pd

import bo_forge
from bo_forge.session import CampaignSession
from bo_forge_app.cli import packaged_streamlit_app_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads(
    (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
)
PROJECT_VERSION = PYPROJECT["project"]["version"]
SDIST_ROOT = f"bo_forge-{PROJECT_VERSION}"
DIST_INFO_ROOT = f"bo_forge-{PROJECT_VERSION}.dist-info"
RELEASE_DEPENDENCY_PROBE_ENV = "BO_FORGE_RELEASE_RESOLVE_DEPENDENCIES"


























































































def _assert_wheel_package_boundaries(wheel_path: Path) -> None:
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        metadata = wheel.read(f"{DIST_INFO_ROOT}/METADATA").decode("utf-8")

    assert "bo_forge/__init__.py" in names
    assert "bo_forge/contextual.py" in names
    assert "bo_forge/multifidelity.py" in names
    assert "bo_forge/plot_registry.py" in names
    assert "bo_forge/structured.py" in names
    assert "bo_forge/application.py" in names
    assert "bo_forge/predictive.py" in names
    assert "bo_forge/_filesystem.py" in names
    assert "bo_forge/_config/parser.py" in names
    assert "bo_forge/_campaign/validation.py" in names
    assert "bo_forge/_optimization/router.py" in names
    assert "bo_forge/_diagnostics/standard.py" in names
    assert "bo_forge_app/streamlit_app.py" in names
    assert "bo_forge_app/streamlit_entry.py" in names
    assert "bo_forge_app/views/campaign.py" in names
    assert "bo_forge_app/ui/theme.py" in names
    assert "bo_forge_app/cli.py" in names
    assert "bo_forge_app/service.py" in names
    assert "bo_forge_app/api.py" in names
    assert "bo_forge_app/api_cli.py" in names
    assert "bo_forge_app/stages.py" in names
    assert "bo_forge_app/__main__.py" in names
    assert "bo_forge_api/api.py" in names
    assert "bo_forge_api/cli.py" in names
    assert "bo_forge_api/contracts.py" in names
    assert "bo_forge_api/stages.py" in names
    assert f"{DIST_INFO_ROOT}/entry_points.txt" in names
    assert f"{DIST_INFO_ROOT}/licenses/LICENSE" in names
    excluded_prefixes = (
        "benchmarks/",
        "notebook_assurance/",
        "docs/",
        "configs/",
        "examples/",
        "notebooks/",
        "requirements/",
        "tests/",
    )
    assert not any(name.startswith(excluded_prefixes) for name in names)
    assert "START_HERE.md" not in names
    assert "Provides-Extra: app" in metadata
    assert "Provides-Extra: api" in metadata
    assert 'Requires-Dist: streamlit>=1.57; extra == "app"' in metadata
    assert 'Requires-Dist: fastapi>=0.115; extra == "api"' in metadata
    assert 'Requires-Dist: uvicorn>=0.30; extra == "api"' in metadata
    assert "Requires-Dist: filelock<4,>=3.16" in metadata
    assert 'Requires-Dist: streamlit>=1.57\n' not in metadata
    assert 'Requires-Dist: fastapi>=0.115\n' not in metadata


def _assert_sdist_contains_release_assets(sdist_path: Path) -> None:
    with tarfile.open(sdist_path) as sdist:
        names = set(sdist.getnames())

    required_paths = {
        "notebook_assurance/__init__.py",
        "notebook_assurance/__main__.py",
        "docs/NOTEBOOK_EXECUTION.md",
        "benchmarks/__init__.py",
        "benchmarks/__main__.py",
        "benchmarks/specs/smoke.yaml",
        "benchmarks/specs/standard.yaml",
        "benchmarks/specs/mixed_smoke.yaml",
        "benchmarks/specs/mixed_standard.yaml",
        "benchmarks/specs/constrained_mixed_smoke.yaml",
        "benchmarks/specs/constrained_mixed_standard.yaml",
        "benchmarks/specs/pending_noisy_smoke.yaml",
        "benchmarks/specs/pending_noisy_standard.yaml",
        "benchmarks/specs/multi_objective_smoke.yaml",
        "benchmarks/specs/multi_objective_standard.yaml",
        "benchmarks/specs/multi_fidelity_smoke.yaml",
        "benchmarks/specs/multi_fidelity_standard.yaml",
        "benchmarks/multi.py",
        "benchmarks/multi_scores.py",
        "benchmarks/multi_evidence.py",
        "benchmarks/multi_report.py",
        "docs/BENCHMARKS.md",
        "notebooks/24_closed_loop_benchmarks.ipynb",
        "README.md",
        "START_HERE.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "SECURITY.md",
        "ROADMAP_V0_TO_V1.md",
        "ROADMAP_V1_X.md",
        "ROADMAP_V2_X.md",
        "ROADMAP_V3_X.md",
        "docs/PUBLIC_API.md",
        "docs/PROVENANCE.md",
        "tests/test_provenance_acceptance.py",
        "tests/test_provenance_closeout.py",
        "tests/fixtures/provenance_v1.json",
        "tests/test_provenance_lifecycle_hardening.py",
        "tests/__init__.py",
        "tests/_session_support.py",
        "docs/MIGRATION_V3.md",
        "docs/STREAMLIT_DEPLOYMENT.md",
        "docs/API_PROBE.md",
        "docs/API_SECURITY.md",
        "docs/CAPABILITY_MATRIX.md",
        "docs/QLOGNEHVI_FEASIBILITY.md",
        "requirements/README.md",
        "requirements/constraints-py311-linux-x86_64.txt",
        "requirements/constraints-py312-linux-x86_64.txt",
        "requirements/constraints-py312-macos-arm64.txt",
        "requirements/constraints-py312-macos-x86_64.txt",
        "examples/quickstart.py",
        "examples/01_simple_2d_maximise_logei_campaign_log.csv",
        "examples/10_multi_objective_mixed_constrained_campaign_log.csv",
        "examples/11_four_objective_mixed_constrained_campaign_log.csv",
        "examples/12_cost_aware_multi_objective_campaign_log.csv",
        "examples/13_structured_campaign_core_campaign_log.csv",
        "examples/14_structured_campaign_tutorial_campaign_log.csv",
        "examples/15_multi_fidelity_qmfkg_campaign_log.csv",
        "examples/16_contextual_logei_campaign_log.csv",
        "examples/17_model_profile_campaign_log.csv",
        "examples/18_noisy_pending_qlognei_campaign_log.csv",
        "examples/19_multi_objective_qlognehvi_campaign_log.csv",
        "examples/20_contextual_cost_review_campaign_log.csv",
        "examples/21_contextual_replicate_campaign_log.csv",
        "examples/22_discrete_multi_fidelity_qmfkg_campaign_log.csv",
        "configs/10_multi_objective_mixed_constrained_qlogehvi.yaml",
        "configs/11_four_objective_mixed_constrained_qlogehvi.yaml",
        "configs/12_cost_aware_multi_objective_qlogehvi.yaml",
        "configs/13_structured_campaign_core.yaml",
        "configs/14_structured_campaign_tutorial.yaml",
        "configs/15_multi_fidelity_qmfkg.yaml",
        "configs/16_contextual_logei.yaml",
        "configs/17_model_profile_logei.yaml",
        "configs/18_noisy_pending_qlognei.yaml",
        "configs/19_multi_objective_qlognehvi.yaml",
        "configs/20_contextual_cost_review_logei.yaml",
        "configs/21_contextual_replicate_logei.yaml",
        "configs/22_discrete_multi_fidelity_qmfkg.yaml",
        "notebooks/01_maximisation_logei_campaign.ipynb",
        "notebooks/10_multi_objective_qlogehvi_campaign.ipynb",
        "notebooks/11_four_objective_qlogehvi_campaign.ipynb",
        "notebooks/12_cost_aware_multi_objective_qlogehvi_campaign.ipynb",
        "notebooks/14_structured_campaign_tutorial.ipynb",
        "notebooks/15_multi_fidelity_qmfkg_campaign.ipynb",
        "notebooks/16_contextual_logei_campaign.ipynb",
        "notebooks/17_model_profile_logei_campaign.ipynb",
        "notebooks/18_noisy_pending_qlognei_campaign.ipynb",
        "notebooks/20_contextual_cost_review_logei_campaign.ipynb",
        "notebooks/22_discrete_multi_fidelity_qmfkg_campaign.ipynb",
        "notebooks/23_predictive_diagnostics.ipynb",
        "tests/test_predictive_exports.py",
        "tests/test_predictive_hardening.py",
        "tests/test_predictive_acceptance.py",
        "tests/test_predictive_workflow_acceptance.py",
        "tests/test_diagnostic_export_safety.py",
        "tests/conftest.py",
        "tests/test_v253_operational_freeze.py",
    }
    assert {f"{SDIST_ROOT}/{path}" for path in required_paths}.issubset(names)
    benchmark_sources = {
        f"{SDIST_ROOT}/{path.relative_to(PROJECT_ROOT).as_posix()}"
        for path in (PROJECT_ROOT / "benchmarks").rglob("*.py")
    }
    assert benchmark_sources.issubset(names)
    assurance_sources = {
        f"{SDIST_ROOT}/{path.relative_to(PROJECT_ROOT).as_posix()}"
        for path in (PROJECT_ROOT / "notebook_assurance").rglob("*.py")
    }
    assert assurance_sources.issubset(names)
    benchmark_specs = {
        f"{SDIST_ROOT}/{path.relative_to(PROJECT_ROOT).as_posix()}"
        for path in (PROJECT_ROOT / "benchmarks" / "specs").glob("*.yaml")
    }
    assert benchmark_specs.issubset(names)
    assert not any("working_log" in name or "latest_suggestions" in name for name in names)


def _assert_sdist_test_fixture_works(
    sdist_path: Path,
    extract_root: Path,
    env: dict[str, str],
) -> None:
    with tarfile.open(sdist_path) as sdist:
        sdist.extractall(extract_root)
    source_root = extract_root / SDIST_ROOT
    benchmark_env = env.copy()
    benchmark_env.pop("PYTHONPATH", None)
    for arguments in (["--help"], ["run", "--help"], ["report", "--help"]):
        completed = subprocess.run(
            [sys.executable, "-m", "benchmarks", *arguments],
            cwd=source_root,
            env=benchmark_env,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert "usage:" in completed.stdout.lower()
    for arguments in (["--help"], ["run", "--help"], ["aggregate", "--help"]):
        completed = subprocess.run(
            [sys.executable, "-m", "notebook_assurance", *arguments],
            cwd=source_root,
            env=benchmark_env,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert "usage:" in completed.stdout.lower()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            "tests/test_app_service.py::test_app_service_review_and_single_objective_mark_observed",
            "tests/test_provenance_acceptance.py",
            "tests/test_provenance_closeout.py::test_original_v310_writer_fixture_remains_mutable_and_migratable",
            "tests/test_benchmark_runner.py::test_runner_pairs_strategies_and_reloads_complete_managed_histories",
            "tests/test_benchmark_multi_scores.py",
            "tests/test_benchmark_multi_evidence.py::test_complete_roundtrip_preserves_campaigns_and_reports_without_objectives",
        ],
        cwd=source_root,
        env=env,
        check=True,
        text=True,
    )


def _install_distribution_and_probe(
    *,
    artifact: Path,
    probe_root: Path,
    env: dict[str, str],
    install_args: list[str],
    resolve_dependencies: bool,
) -> None:
    venv_dir = probe_root / "venv"
    probe_dir = probe_root / "probe"
    probe_dir.mkdir(parents=True)

    subprocess.run(
        _venv_create_command(venv_dir, resolve_dependencies=resolve_dependencies),
        env=env,
        check=True,
        text=True,
    )
    python = venv_dir / "bin" / "python"
    pip = venv_dir / "bin" / "pip"
    effective_install_args = list(install_args)
    if resolve_dependencies:
        effective_install_args = [
            argument
            for argument in effective_install_args
            if argument != "--no-build-isolation"
        ]
    install_env = dict(env)
    if not resolve_dependencies:
        install_env["PYTHONPATH"] = sysconfig.get_paths()["purelib"]
    if "--no-build-isolation" in effective_install_args:
        install_env["PYTHONPATH"] = sysconfig.get_paths()["purelib"]
    subprocess.run(
        _pip_install_command(
            pip,
            artifact,
            install_args=effective_install_args,
            resolve_dependencies=resolve_dependencies,
        ),
        cwd=probe_dir,
        env=install_env,
        check=True,
        text=True,
    )
    if resolve_dependencies:
        _run_pip_check(pip, probe_dir, env)
    completed = subprocess.run(
        [str(venv_dir / "bin" / "bo-forge"), "--version"],
        cwd=probe_dir,
        env=install_env,
        check=True,
        text=True,
        capture_output=True,
    )
    assert completed.stdout == f"bo-forge {PROJECT_VERSION}\n"
    api_help = subprocess.run(
        [str(venv_dir / "bin" / "bo-forge-api"), "--help"],
        cwd=probe_dir,
        env=install_env,
        check=True,
        text=True,
        capture_output=True,
    )
    for option in ("--allow-network-access", "--server-stages-only", "--no-docs"):
        assert option in api_help.stdout

    source_root = str(PROJECT_ROOT.resolve())
    script = f"""
import builtins
import sys
import sysconfig
from pathlib import Path
from importlib.metadata import entry_points

installed_site = sysconfig.get_paths()["purelib"]
if installed_site in sys.path:
    sys.path.remove(installed_site)
sys.path.insert(0, installed_site)
import bo_forge
import bo_forge_app

source_root = Path({source_root!r})
scripts = {{ep.name: ep.value for ep in entry_points(group="console_scripts")}}
assert scripts["bo-forge-app"] == "bo_forge_app.cli:main"
assert scripts["bo-forge-api"] == "bo_forge_api.cli:main"
for module in (bo_forge, bo_forge_app):
    module_path = Path(module.__file__).resolve()
    assert source_root not in module_path.parents, module_path
assert bo_forge.__version__ == {PROJECT_VERSION!r}

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
    script += '''
from bo_forge import (
    CampaignSession, adopt_provenance, accept_provenance_config, fork_campaign,
)
from bo_forge.io import empty_campaign_log
import pandas as pd

config_path, log_path = Path("campaign.yaml"), Path("campaign.csv")
config_path.write_text("""campaign_name: artifact
objective: {name: score, direction: maximize}
variables: [{name: x, type: continuous, lower: 0, upper: 1}]
bo: {initial_design_size: 4}
""", encoding="utf-8")
from bo_forge import CampaignConfig
config = CampaignConfig.from_yaml(config_path)
empty_campaign_log(config).to_csv(log_path, index=False)
def apply_lifecycle(helper, *args):
    preview = helper(*args)
    return helper(*args, apply=True, reason="Artifact acceptance",
                  expected_identities=preview["expected_identities"])
apply_lifecycle(adopt_provenance, config_path, log_path)
session = CampaignSession.from_files(config_path, log_path, provenance_policy="required")
row = dict.fromkeys(session.df.columns, "")
row.update(row_id="artifact_1", status="suggested", source="sobol", iteration=0, x=0.25)
session.append_suggestions(pd.DataFrame([row], columns=session.df.columns))
session.mark_observed("artifact_1", 1.5)
config_path.write_bytes(config_path.read_bytes() + b"\\n# formatting only\\n")
apply_lifecycle(accept_provenance_config, config_path, log_path)
apply_lifecycle(fork_campaign, config_path, log_path, Path("child"))
child = CampaignSession.from_files("child/campaign.yaml", "child/campaign.csv",
                                   provenance_policy="required")
child.validate()
assert child.log_path.read_bytes() == log_path.read_bytes()
assert dict(child.provenance_summary().values)["history"] == "inherited_parent_data"
assert len(child.df) == 1
print("Installed provenance lifecycle acceptance passed")
import json
import math

rows = []
for index, x in enumerate((0.4, 0.55, 0.7, 0.85), start=2):
    row = dict.fromkeys(child.df.columns, "")
    row.update(row_id=f"artifact_{index}", status="suggested", source="sobol", iteration=1, x=x)
    rows.append(row)
child.append_suggestions(pd.DataFrame(rows, columns=child.df.columns))
for row in rows:
    child.mark_observed(row["row_id"], 2 * row["x"] + 1)
sources = (config_path, log_path, Path("campaign.csv.manifest.json"),
           child.config_path, child.log_path, Path("child/campaign.csv.manifest.json"))
before = {path: path.read_bytes() for path in sources}
diagnostics = child.model_predictive_evaluation(profiles=["default"], folds=2, seed=0)
assert diagnostics.summary.fit_status.tolist() == ["complete"]
assert diagnostics.fold_outcomes.fit_status.tolist() == ["complete", "complete"]
assert len(diagnostics.predictions) == 5
assert diagnostics.metadata["requested_fit_count"] == 2
assert diagnostics.predictions.predicted_mean.map(math.isfinite).all()
assert diagnostics.predictions.predicted_variance.map(math.isfinite).all()
assert diagnostics.predictions.predicted_variance.gt(0).all()
exported = diagnostics.export(Path("diagnostics/evaluation"))
assert {p.name for p in exported.iterdir()} == {
    "summary.csv", "predictions.csv", "fold_outcomes.csv", "metadata.json",
}
assert pd.read_csv(exported / "summary.csv").rmse.map(math.isfinite).all()
assert len(pd.read_csv(exported / "predictions.csv")) == 5
assert pd.read_csv(exported / "fold_outcomes.csv").fit_status.eq("complete").all()
assert json.loads((exported / "metadata.json").read_text()) == diagnostics.metadata
try:
    diagnostics.export(exported)
except FileExistsError:
    pass
else:
    raise AssertionError("Installed predictive export overwrote its destination")
assert before == {path: path.read_bytes() for path in sources}
print("Installed predictive evaluation and export acceptance passed (5 rows, 2 fits)")
'''
    subprocess.run(
        [str(python), "-c", script],
        cwd=probe_dir,
        env=install_env,
        check=True,
        text=True,
    )


def _install_app_extra_and_probe(
    *,
    wheel: Path,
    probe_root: Path,
    env: dict[str, str],
    resolve_dependencies: bool,
) -> None:
    venv_dir = probe_root / "venv"
    probe_dir = probe_root / "probe"
    probe_dir.mkdir(parents=True)
    subprocess.run(
        _venv_create_command(venv_dir, resolve_dependencies=resolve_dependencies),
        env=env,
        check=True,
        text=True,
    )
    python = venv_dir / "bin" / "python"
    pip = venv_dir / "bin" / "pip"
    subprocess.run(
        _pip_install_command(
            pip,
            f"{wheel}[app]",
            install_args=[],
            resolve_dependencies=resolve_dependencies,
        ),
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
    )
    if resolve_dependencies:
        _run_pip_check(pip, probe_dir, env)
    source_root = str(PROJECT_ROOT.resolve())
    probe_env = dict(env)
    if not resolve_dependencies:
        probe_env["PYTHONPATH"] = sysconfig.get_paths()["purelib"]
    script = f"""
import sys
import sysconfig
from pathlib import Path

installed_site = sysconfig.get_paths()["purelib"]
if installed_site in sys.path:
    sys.path.remove(installed_site)
sys.path.insert(0, installed_site)
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
        env=probe_env,
        check=True,
        text=True,
    )
    module_help = subprocess.run(
        [str(python), "-m", "bo_forge_app", "--help"],
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    script_help = subprocess.run(
        [str(venv_dir / "bin" / "bo-forge-app"), "--help"],
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "--allow-network-access" in module_help.stdout
    assert "--allow-network-access" in script_help.stdout


def _install_api_extra_and_probe(
    *,
    wheel: Path,
    probe_root: Path,
    env: dict[str, str],
    resolve_dependencies: bool,
) -> None:
    venv_dir = probe_root / "venv"
    probe_dir = probe_root / "probe"
    probe_dir.mkdir(parents=True)
    subprocess.run(
        _venv_create_command(venv_dir, resolve_dependencies=resolve_dependencies),
        env=env,
        check=True,
        text=True,
    )
    python = venv_dir / "bin" / "python"
    pip = venv_dir / "bin" / "pip"
    subprocess.run(
        _pip_install_command(
            pip,
            f"{wheel}[api]",
            install_args=[],
            resolve_dependencies=resolve_dependencies,
        ),
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
    )
    if resolve_dependencies:
        _run_pip_check(pip, probe_dir, env)
    source_root = str(PROJECT_ROOT.resolve())
    probe_env = dict(env)
    if not resolve_dependencies:
        probe_env["PYTHONPATH"] = sysconfig.get_paths()["purelib"]
    script = f"""
import sys
import sysconfig
from pathlib import Path

installed_site = sysconfig.get_paths()["purelib"]
if installed_site in sys.path:
    sys.path.remove(installed_site)
sys.path.insert(0, installed_site)
import fastapi
import uvicorn
import bo_forge_api
import bo_forge_api.api
import bo_forge_app.api
from bo_forge_api.api import create_app

source_root = Path({source_root!r})
api_path = Path(bo_forge_app.api.__file__).resolve()
assert source_root not in api_path.parents, api_path
assert source_root not in Path(bo_forge_api.api.__file__).resolve().parents
assert not hasattr(bo_forge_api, "create_app")
try:
    exec("from bo_forge_api import create_app")
except ImportError:
    pass
else:
    raise AssertionError("bo_forge_api must not export create_app")
app = create_app(root=Path("."))
assert app.title
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
    api_help = subprocess.run(
        [str(venv_dir / "bin" / "bo-forge-api"), "--help"],
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    for option in ("--allow-network-access", "--server-stages-only", "--no-docs"):
        assert option in api_help.stdout


def _venv_create_command(
    venv_dir: Path,
    *,
    resolve_dependencies: bool,
) -> list[str]:
    command = [sys.executable, "-m", "venv"]
    if not resolve_dependencies:
        command.append("--system-site-packages")
    return [*command, str(venv_dir)]


def _pip_install_command(
    pip: Path,
    artifact: str | Path,
    *,
    install_args: list[str],
    resolve_dependencies: bool,
) -> list[str]:
    command = [str(pip), "install"]
    if not resolve_dependencies:
        command.append("--no-deps")
    return [*command, *install_args, str(artifact)]


def _run_pip_check(pip: Path, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(
        [str(pip), "check"],
        cwd=cwd,
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

__all__ = [
    'CampaignSession',
    'DIST_INFO_ROOT',
    'PROJECT_ROOT',
    'PROJECT_VERSION',
    'PYPROJECT',
    'Path',
    'RELEASE_DEPENDENCY_PROBE_ENV',
    'SDIST_ROOT',
    '_assert_sdist_contains_release_assets',
    '_assert_sdist_test_fixture_works',
    '_assert_wheel_package_boundaries',
    '_install_api_extra_and_probe',
    '_install_app_extra_and_probe',
    '_install_core_only_app_missing_streamlit_probe',
    '_install_distribution_and_probe',
    '_pip_install_command',
    '_run_pip_check',
    '_venv_create_command',
    'bo_forge',
    'os',
    'packaged_streamlit_app_path',
    'pd',
    're',
    'shutil',
    'subprocess',
    'sys',
    'sysconfig',
    'tarfile',
    'tomllib',
    'zipfile',
]
