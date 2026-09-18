"""One specification snapshot must describe both scheduling and retained evidence."""

import json
import shutil
from unittest.mock import patch

import pytest
import yaml

from benchmarks import runner
from benchmarks.__main__ import main
from benchmarks.report import generate_report, load_evidence
from benchmarks.spec import SpecError, load_spec, validate_spec
from benchmarks.storage import sha256, write_json
from tests._benchmark_support import controlled_execute, smoke_spec, spec_file


@pytest.fixture(scope="module")
def seed_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("spec-binding")
    with patch.object(runner, "_execute_trial", controlled_execute):
        return runner.run_suite(spec_file(root), root / "run")


@pytest.mark.parametrize("change", ["name", "seeds", "invalid", "duplicate"])
def test_report_rejects_contradictory_snapshot_even_with_matching_hash(
    seed_run, tmp_path, change,
):
    run = shutil.copytree(seed_run, tmp_path / "run")
    path = run / "spec.yaml"
    spec = load_spec(path)
    if change == "invalid":
        payload = "schema_version: 1\nname: incomplete\n"
    elif change == "duplicate":
        payload = path.read_text() + "name: duplicate\n"
    else:
        spec[change] = "different" if change == "name" else [99]
        payload = yaml.safe_dump(spec)
    path.write_text(payload)
    metadata = json.loads((run / "run.json").read_text())
    metadata["spec_sha256"] = sha256(path.read_bytes())
    write_json(run / "run.json", metadata)
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="(specification|spec|unique)"):
        generate_report(run, tmp_path / "report")
    assert all(p.read_bytes() == value for p, value in before.items())
    assert not (tmp_path / "report").exists()


def test_report_accepts_equivalent_snapshot_formatting(seed_run, tmp_path):
    run = shutil.copytree(seed_run, tmp_path / "run")
    path = run / "spec.yaml"
    path.write_text("# Equivalent formatting\n" + yaml.safe_dump(load_spec(path), sort_keys=False))
    metadata = json.loads((run / "run.json").read_text())
    metadata["spec_sha256"] = sha256(path.read_bytes())
    write_json(run / "run.json", metadata)
    assert load_evidence(run)[2].status.eq("complete").all()


def test_runner_uses_the_bytes_it_parsed_even_if_source_changes(tmp_path, monkeypatch):
    path = spec_file(tmp_path)
    original_bytes = path.read_bytes()
    read_bytes = type(path).read_bytes
    read_text = type(path).read_text

    def changed_read_bytes(self, *args, **kwargs):
        result = read_bytes(self, *args, **kwargs)
        if self == path:
            self.write_text("not the scheduled specification")
        return result

    def changed_read_text(self, *args, **kwargs):
        result = read_text(self, *args, **kwargs)
        if self == path:
            self.write_text("not the scheduled specification")
        return result

    monkeypatch.setattr(type(path), "read_bytes", changed_read_bytes)
    monkeypatch.setattr(type(path), "read_text", changed_read_text)
    monkeypatch.setattr(runner, "_execute_trial", controlled_execute)
    run = runner.run_suite(path, tmp_path / "run")
    assert (run / "spec.yaml").read_bytes() == original_bytes
    assert load_evidence(run)[2].status.eq("complete").all()


@pytest.mark.parametrize("route,expected", [
    (None, "branin, hartmann3, hartmann6"), ("mixed", "mixed_quadratic"),
    ("constrained_mixed", "mixed_quadratic"), ("pending_noisy", "branin"),
])
def test_unknown_problem_cli_guidance_matches_route(tmp_path, capsys, route, expected):
    spec = smoke_spec() if route is None else load_spec(
        f"benchmarks/specs/{route}_smoke.yaml",
    )
    spec["problems"][0]["name"] = "typo"
    with pytest.raises(SpecError, match=expected):
        validate_spec(spec)
    assert main(["run", "--spec", str(spec_file(tmp_path, spec)),
                 "--output", str(tmp_path / "run")]) == 1
    assert expected in capsys.readouterr().err
    assert not (tmp_path / "run").exists()
