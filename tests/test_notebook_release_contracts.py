"""CI and source-package contracts for notebook execution assurance."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ".github/workflows/notebook-execution.yml"
PR_PREFIXES = {"01", "04", "08", "12", "14", "18", "20", "22", "23", "24"}


def _workflow(name: str = WORKFLOW) -> dict:
    return yaml.load((ROOT / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _commands(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def test_notebook_matrix_bounds_and_failure_evidence() -> None:
    workflow = _workflow()
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["on"]) == {"workflow_call"}
    assert set(workflow["on"]["workflow_call"]["inputs"]) == {
        "profile", "artifact-name", "source-ref",
    }
    jobs = workflow["jobs"]
    assert set(jobs) == {"schedule", "execute", "aggregate"}
    execute = jobs["execute"]
    assert execute["timeout-minutes"] == "90"
    assert execute["needs"] == "schedule"
    assert execute["strategy"] == {
        "fail-fast": "false", "max-parallel": "2",
        "matrix": {"notebook": "${{ fromJSON(needs.schedule.outputs.notebooks) }}"},
    }
    command = _commands(execute)
    for fragment in (
        'python -m pip install "uv==0.11.3"', "--require-hashes --torch-backend cpu",
        "requirements/constraints-py312-linux-x86_64.txt", "python -m pip check",
        "python -m notebook_assurance run --sdist", '--profile "$PROFILE"',
        '--notebook "$NOTEBOOK"', '--output "$RUNNER_TEMP/notebook-result"',
    ):
        assert fragment in command
    assert command.count("--notebook ") == 1
    assert "--upgrade" not in command
    assert "python -m build" not in command
    assert "continue-on-error" not in execute
    assert all("continue-on-error" not in step for step in execute["steps"])
    upload, = [step for step in execute["steps"]
               if step.get("uses") == "actions/upload-artifact@v4"]
    assert upload["if"] == "always()"
    assert upload["with"] == {
        "name": "notebook-${{ inputs.profile }}-${{ matrix.notebook }}",
        "path": "${{ runner.temp }}/notebook-result/", "include-hidden-files": "true",
        "if-no-files-found": "warn", "retention-days": "14",
    }


def test_aggregate_cannot_skip_failed_or_missing_jobs() -> None:
    jobs = _workflow()["jobs"]
    aggregate = jobs["aggregate"]
    assert aggregate["needs"] == ["schedule", "execute"]
    assert aggregate["if"] == "${{ always() && needs.schedule.result == 'success' }}"
    assert "continue-on-error" not in aggregate
    tolerant, = [step for step in aggregate["steps"] if "continue-on-error" in step]
    assert tolerant["uses"] == "actions/download-artifact@v4"
    assert tolerant["continue-on-error"] == "true"
    assert tolerant["with"]["merge-multiple"] == "false"
    assert tolerant["with"]["pattern"] == "notebook-${{ inputs.profile }}-*.ipynb"
    command = _commands(aggregate)
    for fragment in (
        'mkdir -p "$RUNNER_TEMP/notebook-evidence"',
        "python -m notebook_assurance aggregate --sdist", '--profile "$PROFILE"',
        '--evidence "$RUNNER_TEMP/notebook-evidence"',
        '--output "$RUNNER_TEMP/notebook-aggregate"',
    ):
        assert fragment in command
    assert "--notebook" not in command
    assert "python -m build" not in command
    upload = aggregate["steps"][-1]
    assert upload["uses"] == "actions/upload-artifact@v4"
    assert upload["if"] == "always()"
    assert upload["with"]["retention-days"] == "14"
    for name in ("execute", "aggregate"):
        steps = jobs[name]["steps"]
        checkout, = [step for step in steps if step.get("uses") == "actions/checkout@v4"]
        assert checkout["with"]["ref"] == "${{ inputs.source-ref }}"
    for job in jobs.values():
        download, = [step for step in job["steps"]
                     if step.get("uses") == "actions/download-artifact@v4"
                     and "name" in step["with"]]
        assert download["with"]["name"] == "${{ inputs.artifact-name }}"


@pytest.mark.parametrize("profile", ["pr", "full", "invalid"])
@pytest.mark.parametrize("missing", [False, True])
def test_archive_scheduler_selects_complete_profile(tmp_path: Path, profile: str, missing: bool):
    notebooks = sorted(path.name for path in (ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 20
    with tarfile.open(tmp_path / "source.tar.gz", "w:gz") as archive:
        for name in notebooks[1:] if missing else notebooks:
            member = tarfile.TarInfo(f"bo_forge/notebooks/{name}")
            member.size = 2
            archive.addfile(member, io.BytesIO(b"{}"))
    command, = [step["run"] for step in _workflow()["jobs"]["schedule"]["steps"]
                if "run" in step]
    script = command.removeprefix("python - <<'PY'\n").removesuffix("PY\n")
    output = tmp_path / "output"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PROFILE": profile, "DIST_DIR": str(tmp_path),
             "GITHUB_OUTPUT": str(output)},
        capture_output=True, text=True, check=False,
    )
    if missing or profile == "invalid":
        assert completed.returncode != 0
        assert ("Expected exactly 20 unique archive notebooks" if missing
                else "Unknown notebook profile") in completed.stderr
        assert not output.exists()
    else:
        assert completed.returncode == 0, completed.stderr
        scheduled = json.loads(output.read_text().removeprefix("notebooks="))
        expected = notebooks if profile == "full" else [
            name for name in notebooks if name[:2] in PR_PREFIXES
        ]
        assert scheduled == expected
        assert len(scheduled) == (20 if profile == "full" else 10)


def test_pr_manual_and_exact_tag_share_built_artifacts() -> None:
    for filename, producer, profile in (
        ("ci.yml", "package", "pr"),
        ("notebook-full.yml", "source", "full"),
        ("release-gate.yml", "validate-tag", "full"),
    ):
        workflow = _workflow(f".github/workflows/{filename}")
        jobs = workflow["jobs"]
        consumer = jobs["notebooks"]
        assert consumer["needs"] == producer
        assert consumer["uses"] == f"./{WORKFLOW}"
        assert consumer["with"]["profile"] == profile
        assert _commands(jobs[producer]).count("python -m build ") == 1
        uploads = [step["with"] for step in jobs[producer]["steps"]
                   if step.get("uses") == "actions/upload-artifact@v4"]
        if profile == "pr" or producer == "source":
            assert consumer["with"]["artifact-name"] == "notebook-source"
            assert any(upload["name"] == "notebook-source" for upload in uploads)
            assert consumer["with"]["source-ref"] == "${{ github.sha }}"
        else:
            assert uploads[0]["name"] == "bo-forge-${{ env.RELEASE_REF }}-verified"
            assert consumer["with"]["artifact-name"] == (
                "bo-forge-${{ github.event_name == 'push' "
                "&& github.ref_name || inputs.ref }}-verified"
            )
            assert consumer["with"]["source-ref"] == "${{ needs.validate-tag.outputs.source-ref }}"
            assert jobs[producer]["outputs"]["source-ref"] == "${{ steps.identity.outputs.commit }}"
            assert '["git", "rev-parse", "HEAD"]' in _commands(jobs[producer])
    assert set(_workflow(".github/workflows/notebook-full.yml")["on"]) == {"workflow_dispatch"}


def test_assurance_is_source_only_and_not_exempt_from_static_checks() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "notebook_assurance/**/*.py" in project["tool"]["ruff"]["include"]
    packages = project["tool"]["setuptools"]["packages"]["find"]["include"]
    assert not any(pattern.startswith("notebook_assurance") for pattern in packages)
    assert "recursive-include notebook_assurance *.py" in (ROOT / "MANIFEST.in").read_text()
    assert (ROOT / "notebook_assurance/__main__.py").is_file()
    static = _commands(_workflow(".github/workflows/ci.yml")["jobs"]["static"])
    syntax, = [line for line in static.splitlines() if "compileall" in line]
    assert "notebook_assurance" in syntax
    from tests.test_v3_architecture import GOVERNED_ROOTS

    assert ROOT / "notebook_assurance" in GOVERNED_ROOTS


def test_execution_guide_documents_contract_without_granting_acceptance() -> None:
    guide = (ROOT / "docs/NOTEBOOK_EXECUTION.md").read_text()
    for fragment in (
        "--sdist", "--profile", "--notebook", "--evidence", "--output",
        "archive_sha256", "scheduled", "notebooks", "result.json", "source.tar.gz",
        "`passed`, `failed`, `interrupted`, `timeout`, and `not_run`",
        "missing jobs", "executed proof", "mismatched identities",
        "Execution acceptance requires successful full-profile results",
        "notebook-full.yml", "release-gate.yml",
    ):
        assert fragment in guide
    for filename in ("README.md", "CONTRIBUTING.md", "ROADMAP_V3_X.md",
                     "docs/RELEASE_CHECKLIST.md"):
        assert "NOTEBOOK_EXECUTION.md" in (ROOT / filename).read_text()
