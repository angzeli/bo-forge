import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_MODULES = {"torch", "botorch", "gpytorch", "matplotlib", "streamlit", "fastapi"}


def _probe(source: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _blocked_probe(operation: str) -> dict[str, object]:
    return _probe(
        f"""
import json
import sys
{operation}
blocked = {sorted(BLOCKED_MODULES)!r}
print(json.dumps({{"blocked": sorted(set(blocked) & set(sys.modules))}}))
"""
    )


def test_import_bo_forge_avoids_heavy_dependencies() -> None:
    assert _blocked_probe("import bo_forge")["blocked"] == []


def test_lazy_campaign_session_export_remains_lightweight() -> None:
    result = _blocked_probe(
        """
from bo_forge import CampaignSession
assert CampaignSession.__module__ == "bo_forge.session"
"""
    )
    assert result["blocked"] == []


def test_module_version_path_avoids_heavy_dependencies() -> None:
    result = _blocked_probe(
        """
import runpy
sys.argv = ["bo_forge", "--version"]
try:
    runpy.run_module("bo_forge", run_name="__main__")
except SystemExit as exc:
    assert exc.code == 0
"""
    )
    assert result["blocked"] == []


def test_cli_help_path_avoids_heavy_dependencies() -> None:
    result = _blocked_probe(
        """
import runpy
sys.argv = ["bo_forge", "--help"]
try:
    runpy.run_module("bo_forge", run_name="__main__")
except SystemExit as exc:
    assert exc.code == 0
"""
    )
    assert result["blocked"] == []


def test_campaign_validation_path_avoids_heavy_dependencies() -> None:
    result = _blocked_probe(
        """
from bo_forge.cli import run
code = run([
    "validate",
    "--config", "configs/22_discrete_multi_fidelity_qmfkg.yaml",
    "--log", "examples/22_discrete_multi_fidelity_qmfkg_campaign_log.csv",
])
assert code == 0
"""
    )
    assert result["blocked"] == []


def test_single_objective_summary_path_avoids_heavy_dependencies() -> None:
    result = _blocked_probe(
        """
from bo_forge.cli import run
code = run([
    "summary",
    "--config", "configs/01_simple_2d_maximise_logei.yaml",
    "--log", "examples/01_simple_2d_maximise_logei_campaign_log.csv",
])
assert code == 0
"""
    )
    assert result["blocked"] == []


def test_fidelity_summary_path_avoids_heavy_dependencies() -> None:
    result = _blocked_probe(
        """
from bo_forge.cli import run
code = run([
    "fidelity-summary",
    "--config", "configs/22_discrete_multi_fidelity_qmfkg.yaml",
    "--log", "examples/22_discrete_multi_fidelity_qmfkg_campaign_log.csv",
])
assert code == 0
"""
    )
    assert result["blocked"] == []


def test_lazy_public_exports_preserve_star_import_and_dir() -> None:
    result = _probe(
        """
import json
import bo_forge
namespace = {}
exec("from bo_forge import *", namespace)
missing = sorted(set(bo_forge.__all__) - set(namespace))
print(json.dumps({
    "missing": missing,
    "dir_has_all": set(bo_forge.__all__).issubset(dir(bo_forge)),
    "session_module": bo_forge.CampaignSession.__module__,
}))
"""
    )
    assert result == {
        "missing": [],
        "dir_has_all": True,
        "session_module": "bo_forge.session",
    }
