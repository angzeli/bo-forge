"""Manual performance CI and release-evidence documentation contracts only."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ".github/workflows/performance.yml"
NOTEBOOK_ARCHIVE = "466a48c15347e5d8c4d2635afddbeb85acf53c693bd26dc856e47aa6b0fc9f40"


def _workflow(path: str = WORKFLOW) -> dict:
    return yaml.load((ROOT / path).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_performance_is_manual_single_host_and_bounded() -> None:
    workflow = _workflow()
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    job, = workflow["jobs"].values()
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["timeout-minutes"] == "120"
    assert "strategy" not in job and "continue-on-error" not in job
    assert all("continue-on-error" not in step for step in job["steps"])
    setup, = [s for s in job["steps"] if s.get("uses") == "actions/setup-python@v5"]
    assert setup["with"]["python-version"] == "3.12"
    checkout, = [s for s in job["steps"] if s.get("uses") == "actions/checkout@v4"]
    assert checkout["with"]["fetch-depth"] == "0"
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        assert workflow["env"][name] == "1"
    for filename in (".github/workflows/ci.yml", ".github/workflows/release-gate.yml"):
        text = (ROOT / filename).read_text(encoding="utf-8")
        assert "benchmarks.performance run" not in text
        assert f"./{WORKFLOW}" not in text


def test_performance_uses_constrained_separate_archive_builds() -> None:
    job, = _workflow()["jobs"].values()
    commands = "\n".join(s.get("run", "") for s in job["steps"])
    for fragment in (
        'python -m pip install "uv==0.11.3"',
        "uv pip install --system --require-hashes --torch-backend cpu",
        "requirements/constraints-py312-linux-x86_64.txt", "python -m pip check",
        'git worktree add --detach "$RUNNER_TEMP/performance-baseline" 6c0d57db',
        'git -C "$RUNNER_TEMP/performance-baseline" rev-parse HEAD',
        'git rev-parse HEAD > "$RUNNER_TEMP/performance-inputs/candidate-commit.txt"',
        '--outdir "$RUNNER_TEMP/performance-inputs/baseline" "$RUNNER_TEMP/performance-baseline"',
        '--outdir "$RUNNER_TEMP/performance-inputs/candidate"',
        'sha256sum "${baselines[0]}" "${candidates[0]}"',
        '[[ ${#baselines[@]} -eq 1 && -f ${baselines[0]} ]]',
        '[[ ${#candidates[@]} -eq 1 && -f ${candidates[0]} ]]',
    ):
        assert fragment in commands
    assert commands.count("python -m build --sdist --no-isolation") == 2
    for forbidden in ("--upgrade", " -e .", "--editable", "git checkout", "git reset"):
        assert forbidden not in commands


def test_performance_run_report_and_failure_upload_contract() -> None:
    job, = _workflow()["jobs"].values()
    steps = job["steps"]
    run, = [s for s in steps if s.get("id") == "compare"]
    for fragment in (
        "python -m benchmarks.performance run",
        '--baseline "${baselines[0]}" --candidate "${candidates[0]}"',
        '--output "$RUNNER_TEMP/performance-run"',
    ):
        assert fragment in run["run"]
    report, = [s for s in steps if "benchmarks.performance report" in s.get("run", "")]
    assert report["if"] == "${{ always() && steps.compare.outcome != 'skipped' }}"
    assert '--input "$RUNNER_TEMP/performance-run"' in report["run"]
    assert '--output "$RUNNER_TEMP/performance-report"' in report["run"]
    assert steps.index(run) < steps.index(report)
    upload = steps[-1]
    assert upload["uses"] == "actions/upload-artifact@v4"
    assert upload["if"] == "always()"
    assert upload["with"]["retention-days"] == "14"
    assert upload["with"]["if-no-files-found"] == "warn"
    assert upload["with"]["include-hidden-files"] == "true"
    assert upload["with"]["path"].splitlines() == [
        "${{ runner.temp }}/performance-inputs/", "${{ runner.temp }}/performance-run/",
        "${{ runner.temp }}/performance-report/",
        "!${{ runner.temp }}/performance-run/*/venv/**",
        "!${{ runner.temp }}/performance-run/*/extracted/**",
    ]
    for step in steps:
        if "run" in step:
            checked = subprocess.run(
                ["bash", "-n"], input=step["run"], text=True, capture_output=True, check=False,
            )
            assert checked.returncode == 0, checked.stderr


def test_performance_docs_separate_implementation_closeout_from_release_gates() -> None:
    guide = (ROOT / "docs/PERFORMANCE_BENCHMARKS.md").read_text(encoding="utf-8")
    current, historical = guide.split("## Historical Timings: Not Comparable To v3.3.4")
    for fragment in (
        "Performance acceptance: passed (156/156 processes; 13/13 valid cases)", "6c0d57db",
        "13 cases x 2 versions x 6 samples = 156 fresh processes", "600 seconds",
        "90 minutes", "120 minutes", "Isolated, non-editable", "excluded from sample timings",
        "Advisory candidate/baseline", "Full performance runs are not PR CI",
        "python -m benchmarks.performance run --baseline PATH --candidate PATH --output PATH",
        "python -m benchmarks.performance report --input PATH --output NEWPATH",
        "v3.3.x implementation is complete", "not exact-commit CI",
        "The local release gate must pass before commit", "exact-commit CI remains pending",
        "report-validation and interruption-checkpoint fixes plus closeout docs",
        "3de4067a70fb32a85ca2d92e1ae30e41a68414e74df16615f76f622d23065c56",
        "e24137cc7ea94b287c613658dcc741c79489462059164a7b1c229e85e324e971",
        "median (IQR)", "inclusive quartiles", "not cold-cache startup",
        "not incremental model memory", "0.9660-1.0822", "0.9012-1.0794", "0.9939-1.0086",
        "reports/performance/v3.3.4/paired-acceptance/",
        "reports/performance/v3.3.4/paired-report/", "ignored local evidence",
    ):
        assert fragment in current
    for fragment in (
        "median of five warm-cache subprocess runs", "Python 3.11.14 and BoTorch 0.17.2",
        "| `bo_forge --version` | 2.2599 | 0.3595 | -84.1% |",
        "| Discrete qMFKG `q=4` suggestion | 14.6199 | 12.3885 | -15.3% |",
    ):
        assert fragment in historical
    for filename in ("ROADMAP_V3_X.md", "docs/RELEASE_CHECKLIST.md", "docs/NOTEBOOK_EXECUTION.md"):
        content = (ROOT / filename).read_text(encoding="utf-8")
        assert NOTEBOOK_ARCHIVE in content
        assert "reports/notebooks/v3.3.3/seed-baseline-fix/" in content
    roadmap = (ROOT / "ROADMAP_V3_X.md").read_text(encoding="utf-8")
    assert "| `v3.3.x` | completed |" in roadmap
    assert "| `v3.3.4` | implementation complete |" in roadmap
    assert "class v33 majorDone" in roadmap
    assert "class v334 patchDone" in roadmap
