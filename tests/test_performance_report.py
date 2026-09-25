"""Deterministic retained-evidence reporting, without installs or model execution."""

import csv
import json
import subprocess
from pathlib import Path

import pytest

from benchmarks.performance import HARNESS_VERSION, report, schedule
from benchmarks.storage import write_json
from notebook_assurance.archive import digest


def _rows():
    rows = []
    for item in schedule():
        value = 10000 if item["warmup"] else item["repetition"]
        seconds = value * (2 if item["version"] == "baseline" else 1)
        rows.append({**item, "status": "complete", "returncode": 0,
                     "process_seconds": seconds,
                     "call_seconds": seconds / 4 if item["kind"] == "suggest" else None,
                     "peak_rss_bytes": seconds * 1024})
    return rows


def _fake_installation(root, label, version):
    directory = root / label
    (directory / "wheels").mkdir(parents=True)
    archive, wheel = directory / "source.tar.gz", directory / "wheels/core.whl"
    archive.write_text(f"synthetic archive {version}\n")
    wheel.write_text(f"synthetic wheel {version}\n")
    prefix = directory / "venv"
    identity = {
        "version": version, "python": "3.12.8", "requirements": ["numpy>=2"],
        "packages": {"bo-forge": version, "numpy": "2.1.0"},
        "prefix": str(prefix), "executable": str(prefix / "bin/python"),
        "import_path": str(prefix / "lib/site-packages/bo_forge/__init__.py"),
        "files": {}, "archive": "source.tar.gz", "wheel": "wheels/core.whl",
        "archive_sha256": digest(archive), "wheel_sha256": digest(wheel),
    }
    write_json(directory / "identity.json", identity)
    write_json(directory / "postflight.json", {
        key: identity[key] for key in ("files", "packages", "python", "import_path")})
    return identity


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    # This fixture tests serialized evidence, not campaign semantics; those have real inputs below.
    monkeypatch.setattr(report, "_suggestions", lambda *args: None)
    root = tmp_path / "evidence"
    (root / "harness").mkdir(parents=True)
    harness = root / "harness/worker.py"
    harness.write_text("# Synthetic retained harness; never executed.\n")
    constraints = root / "constraints.txt"
    constraints.write_text("numpy==2.1.0\n")
    metadata = {
        "schema_version": 1, "harness_version": HARNESS_VERSION, "status": "complete",
        "machine": {"machine": "synthetic-host", "python": "3.12.8"},
        "schedule": schedule(), "harness_files": {"worker.py": digest(harness)},
        "constraints_sha256": digest(constraints), "inputs": {},
        "installations": {label: _fake_installation(root, label, version)
                          for label, version in (("baseline", "3.3.3"),
                                                 ("candidate", "3.3.4"))},
    }
    for route in ("log_ei", "qlog_ehvi", "qmf_kg"):
        directory = root / "inputs" / route
        directory.mkdir(parents=True)
        (directory / "campaign.yaml").write_text(f"route: {route}\n")
        (directory / "campaign.csv").write_text("row_id,status\n0,observed\n")
        metadata["inputs"][route] = {"expected_source": route}
        for key, filename in (("config", "campaign.yaml"), ("log", "campaign.csv")):
            metadata["inputs"][route].update({
                f"{key}_path": f"{route}/{filename}",
                f"{key}_sha256": digest(directory / filename)})
    for row in _rows():
        directory = root / "samples" / row["sample_id"]
        directory.mkdir(parents=True)
        inputs = metadata["inputs"][row.get("route", "qmf_kg")]
        identity = metadata["installations"][row["version"]]
        item = next(item for item in metadata["schedule"]
                    if item["sample_id"] == row["sample_id"])
        request = {**item, "installation_root": identity["prefix"],
                   "expected_source": inputs["expected_source"],
                   "result_path": str(directory / "result.json"),
                   "suggestions_path": str(directory / "suggestions.csv")}
        for key in ("config", "log"):
            source = root / "inputs" / inputs[f"{key}_path"]
            target = directory / source.name
            target.write_bytes(source.read_bytes())
            request.update({f"{key}_path": str(target),
                            f"{key}_sha256": inputs[f"{key}_sha256"]})
        write_json(directory / "request.json", request)
        (directory / "process.log").write_text("Synthetic sample completed.\n")
        if row["kind"] == "suggest":
            worker = {
                **{key: row[key] for key in ("sample_id", "repetition", "warmup", "version")},
                "status": "complete", "case_id": row["case_id"],
                "batch_size": row["batch_size"], "expected_source": request["expected_source"],
                "observed_sources": [request["expected_source"]],
                "call_seconds": row["call_seconds"], "peak_rss_bytes": row["peak_rss_bytes"],
                "inputs_unchanged": True,
                "input_sha256": {key: inputs[f"{key}_sha256"] for key in ("config", "log")},
                "package_locations": {"bo_forge": identity["import_path"]},
                "package_versions": {"bo_forge": identity["version"]}, "warnings": [],
                "validation": {key: True for key in (
                    "expected_source", "batch_size", "canonical_schema", "unobserved",
                    "feasible", "unique")},
            }
            write_json(directory / "result.json", worker)
            (directory / "suggestions.csv").write_text(
                "row_id,status,source\n" + "".join(
                    f"{index + 1},suggested,{request['expected_source']}\n"
                    for index in range(row["batch_size"])))
            row["worker"] = worker
        row["artifact_hashes"] = {path.name: digest(path) for path in directory.iterdir()}
        write_json(directory / "status.json", row)
    write_json(root / "run.json", metadata)
    return root


def _read(path):
    return json.loads(path.read_text())


def _sample(root):
    return root / "samples/log_ei-q1-1-baseline"


def _rewrite_artifact(directory, name, value):
    write_json(directory / name, value)
    status = _read(directory / "status.json")
    status["artifact_hashes"][name] = digest(directory / name)
    write_json(directory / "status.json", status)


def _assert_rejected(root, target, match):
    with pytest.raises(ValueError, match=match):
        report.report(root, target)
    assert not target.exists()
    assert not list(target.parent.glob(f".{target.name}-*"))


def test_statistics_use_inclusive_quartiles_and_exact_five_measurements():
    assert report.statistics_for([10, 2, 8, 4, 6]) == {
        "median": 6, "iqr": 4, "minimum": 2, "maximum": 10}
    for values in ([], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6], [1, 2, None, 4, 5]):
        assert report.statistics_for(values) is None


def test_summary_excludes_warmups_and_reports_candidate_over_baseline():
    summaries = report.summarize(_rows())
    assert len(summaries) == 13
    for summary in summaries:
        assert summary["comparable"] is True
        assert summary["complete_samples"] == summary["scheduled_samples"] == 12
        process = summary["measurements"]["process_seconds"]
        assert process["baseline"] == {"median": 6, "iqr": 4, "minimum": 2, "maximum": 10}
        assert process["candidate"] == {"median": 3, "iqr": 2, "minimum": 1, "maximum": 5}
        assert process["candidate_baseline_ratio"] == 0.5
        call = summary["measurements"]["call_seconds"]
        if "-q" in summary["case_id"]:
            assert call["baseline"]["median"] == 1.5
            assert call["candidate_baseline_ratio"] == 0.5
        else:
            assert call == {"baseline": None, "candidate": None, "candidate_baseline_ratio": None}
        assert summary["measurements"]["peak_rss_bytes"]["baseline"]["median"] == 6144


@pytest.mark.parametrize("warmup", [False, True], ids=["measured", "warmup"])
@pytest.mark.parametrize("outcome", ["missing", "failed", "timeout", "interrupted", "not_run"])
def test_incomplete_pair_never_gets_a_ratio(warmup, outcome):
    rows = _rows()
    row = next(row for row in rows if row["case_id"] == "log_ei-q1"
               and row["version"] == "baseline" and row["warmup"] == warmup)
    if outcome == "missing":
        rows.remove(row)
    else:
        row["status"] = outcome
    summaries = {row["case_id"]: row for row in report.summarize(rows)}
    affected = summaries["log_ei-q1"]
    assert affected["comparable"] is False
    assert affected["complete_samples"] == 11
    assert all(metric["candidate_baseline_ratio"] is None
               for metric in affected["measurements"].values())
    assert summaries["qmf_kg-q4"]["comparable"] is True


def test_report_regeneration_reads_only_retained_evidence(evidence, tmp_path, monkeypatch):
    from benchmarks.performance import execution, preparation, runner, worker

    def forbidden(*args, **kwargs):
        pytest.fail("Reporting must not execute, prepare installations, or fit a model")

    for module, name in ((execution, "execute"), (preparation, "prepare"),
                         (preparation, "checked"), (runner, "run"), (runner, "execute"),
                         (worker, "run_request"), (subprocess, "Popen")):
        monkeypatch.setattr(module, name, forbidden)
    metadata, rows = report.load(evidence)
    assert len(rows) == 156 and all(row["status"] == "complete" for row in rows)
    assert sum(row["warmup"] for row in rows) == 26
    before = {path.relative_to(evidence): digest(path)
              for path in evidence.rglob("*") if path.is_file()}
    first, second = tmp_path / "first", tmp_path / "regenerated"
    assert report.report(evidence, first) is True
    assert report.report(evidence, second) is True
    assert {path.name for path in first.iterdir()} == {"summary.json", "samples.csv", "report.md"}
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    assert before == {path.relative_to(evidence): digest(path)
                      for path in evidence.rglob("*") if path.is_file()}
    summary = _read(first / "summary.json")
    assert summary["valid"] is True and summary["comparison"] == "advisory"
    assert summary["archive_sha256"] == {
        name: identity["archive_sha256"] for name, identity in metadata["installations"].items()}
    with (first / "samples.csv").open() as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 156
    assert next(row for row in exported if row["case_id"] == "import")["call_seconds"] == ""


@pytest.mark.parametrize("relative", [
    "baseline/source.tar.gz", "candidate/wheels/core.whl", "harness/worker.py",
    "inputs/log_ei/campaign.yaml", "inputs/qmf_kg/campaign.csv",
    "samples/log_ei-q1-1-baseline/campaign.csv",
    "samples/log_ei-q1-1-baseline/suggestions.csv",
])
def test_corrupt_hashed_evidence_is_rejected_without_publication(evidence, tmp_path, relative):
    (evidence / relative).write_text("corrupted bytes\n")
    _assert_rejected(evidence, tmp_path / "report", "identity mismatch")


@pytest.mark.parametrize("corrupt", [False, True])
def test_optional_shared_harness_files_are_hash_checked(evidence, tmp_path, corrupt):
    shared = evidence / "harness/shared/notebook_assurance/archive.py"
    shared.parent.mkdir(parents=True)
    shared.write_text("# Retained shared helper.\n")
    metadata = _read(evidence / "run.json")
    metadata["shared_harness_files"] = {"notebook_assurance/archive.py": digest(shared)}
    write_json(evidence / "run.json", metadata)
    if corrupt:
        shared.write_text("# Changed shared helper.\n")
        _assert_rejected(evidence, tmp_path / "report", "identity mismatch")
    else:
        _, rows = report.load(evidence)
        assert len(rows) == 156


def test_installation_identity_must_match_retained_copy(evidence, tmp_path):
    path = evidence / "baseline/identity.json"
    identity = _read(path)
    identity["version"] = "0.0.0"
    write_json(path, identity)
    _assert_rejected(evidence, tmp_path / "report", "installation identity mismatch")


@pytest.mark.parametrize("key,value", [("sample_id", "other"), ("version", "candidate"),
                                       ("case_id", "qmf_kg-q4"), ("warmup", True)])
def test_status_must_match_scheduled_identity(evidence, tmp_path, key, value):
    path = _sample(evidence) / "status.json"
    status = _read(path)
    status[key] = value
    write_json(path, status)
    _assert_rejected(evidence, tmp_path / "report", "identity mismatch")


@pytest.mark.parametrize("key,value", [("case_id", "qmf_kg-q4"), ("version", "candidate"),
                                       ("sample_id", "other"), ("warmup", True),
                                       ("repetition", 4), ("kind", "import"),
                                       ("config_sha256", "0" * 64)])
def test_rehashed_request_must_match_schedule_and_inputs(evidence, tmp_path, key, value):
    directory = _sample(evidence)
    request = _read(directory / "request.json")
    request[key] = value
    _rewrite_artifact(directory, "request.json", request)
    _assert_rejected(evidence, tmp_path / "report", "mismatch|differs|source|identity")


def test_rehashed_worker_must_match_saved_status(evidence, tmp_path):
    directory = _sample(evidence)
    worker = _read(directory / "result.json")
    worker["call_seconds"] += 1
    _rewrite_artifact(directory, "result.json", worker)
    _assert_rejected(evidence, tmp_path / "report", "Suggestion evidence mismatch")


@pytest.mark.parametrize("postflight", ["missing", "matching"])
def test_failed_postflight_run_has_no_comparable_cases_or_ratios(evidence, tmp_path, postflight):
    metadata = _read(evidence / "run.json")
    metadata.update(status="failed", message="Postflight did not complete")
    write_json(evidence / "run.json", metadata)
    if postflight == "missing":
        for version in ("baseline", "candidate"):
            (evidence / version / "postflight.json").unlink(missing_ok=True)
    _, rows = report.load(evidence)
    assert len(rows) == 156 and all(row["status"] == "complete" for row in rows)
    target = tmp_path / "report"
    assert report.report(evidence, target) is False
    summary = _read(target / "summary.json")
    assert summary["valid"] is False
    assert all(case["comparable"] is False for case in summary["cases"])
    assert all(metric["candidate_baseline_ratio"] is None
               for case in summary["cases"] for metric in case["measurements"].values())
    assert all(case["measurements"]["process_seconds"]["baseline"]["median"] == 6
               for case in summary["cases"])
    with (target / "samples.csv").open() as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 156 and all(row["status"] == "complete" for row in exported)


@pytest.mark.parametrize("version", ["baseline", "candidate"])
def test_complete_run_requires_both_postflight_records(evidence, tmp_path, version):
    (evidence / version / "postflight.json").unlink(missing_ok=True)
    _assert_rejected(evidence, tmp_path / "report", rf"{version}:.*postflight")


@pytest.mark.parametrize("run_status", ["complete", "failed"])
@pytest.mark.parametrize("version", ["baseline", "candidate"])
@pytest.mark.parametrize("field,value", [
    ("files", {"lib/bo_forge/__init__.py": "changed"}),
    ("packages", {"bo-forge": "0.0.0"}),
    ("python", "0.0.0"), ("import_path", "/outside/bo_forge/__init__.py"),
])
def test_present_postflight_must_match_even_for_failed_run(
    evidence, tmp_path, run_status, version, field, value,
):
    metadata = _read(evidence / "run.json")
    metadata["status"] = run_status
    write_json(evidence / "run.json", metadata)
    after = {key: metadata["installations"][version][key]
             for key in ("files", "packages", "python", "import_path")}
    after[field] = value
    write_json(evidence / version / "postflight.json", after)
    _assert_rejected(evidence, tmp_path / "report", rf"{version}:.*postflight {field}")


@pytest.mark.parametrize("rss", [None, 999999])
def test_top_level_rss_must_match_retained_worker(evidence, tmp_path, rss):
    path = _sample(evidence) / "status.json"
    status = _read(path)
    assert status["worker"]["peak_rss_bytes"] != rss
    status["peak_rss_bytes"] = rss
    write_json(path, status)
    _assert_rejected(evidence, tmp_path / "report", "Suggestion evidence mismatch|RSS|rss")


@pytest.mark.parametrize("field,value", [
    ("sample_id", "log_ei-q1-2-baseline"), ("repetition", 2),
    ("warmup", True), ("version", "candidate"),
])
def test_rehashed_worker_identity_must_match_schedule(evidence, tmp_path, field, value):
    directory = _sample(evidence)
    worker = _read(directory / "result.json")
    worker[field] = value
    _rewrite_artifact(directory, "result.json", worker)
    path = directory / "status.json"
    status = _read(path)
    status["worker"] = worker
    write_json(path, status)
    _assert_rejected(evidence, tmp_path / "report", "Worker sample identity mismatch")


@pytest.mark.parametrize("metric", ["process_seconds", "call_seconds"])
def test_complete_sample_requires_timing(evidence, tmp_path, metric):
    path = _sample(evidence) / "status.json"
    status = _read(path)
    status[metric] = None
    write_json(path, status)
    _assert_rejected(evidence, tmp_path / "report", "timing|Suggestion evidence mismatch")


def test_missing_status_and_rss_stay_unavailable(evidence, tmp_path):
    missing = evidence / "samples/import-0-baseline/status.json"
    missing.unlink()
    path = evidence / "samples/import-1-candidate/status.json"
    status = _read(path)
    status["peak_rss_bytes"] = None
    write_json(path, status)
    _, rows = report.load(evidence)
    row = next(row for row in rows if row["sample_id"] == "import-0-baseline")
    assert row["status"] == "not_run"
    assert row["process_seconds"] is row["call_seconds"] is row["peak_rss_bytes"] is None
    target = tmp_path / "report"
    assert report.report(evidence, target) is False
    summary = _read(target / "summary.json")
    assert summary["valid"] is False
    imported = next(row for row in summary["cases"] if row["case_id"] == "import")
    assert imported["measurements"]["peak_rss_bytes"]["candidate"] is None
    assert imported["measurements"]["process_seconds"]["candidate_baseline_ratio"] is None
    with (target / "samples.csv").open() as stream:
        missing_row = next(row for row in csv.DictReader(stream)
                           if row["sample_id"] == "import-0-baseline")
    assert missing_row["process_seconds"] == ""
    assert "FAILED / incomplete" in (target / "report.md").read_text()


@pytest.mark.parametrize("collision", ["existing", "source", "inside", "ancestor", "symlink"])
def test_report_destination_cannot_overwrite_or_overlap_source(tmp_path, monkeypatch, collision):
    source = tmp_path / "evidence"
    source.mkdir()
    sentinel = source / "retained.txt"
    sentinel.write_text("retain me\n")
    if collision == "existing":
        target = tmp_path / "existing"
        target.mkdir()
    elif collision == "source":
        target = source
    elif collision == "inside":
        target = source / "report"
    elif collision == "ancestor":
        target = tmp_path
    else:
        target = tmp_path / "alias"
        target.symlink_to(source, target_is_directory=True)

    def forbidden(*args):
        pytest.fail("Destination collision must be rejected before reading evidence")

    monkeypatch.setattr(report, "load", forbidden)
    with pytest.raises(FileExistsError, match="new directory outside"):
        report.report(source, target)
    assert sentinel.read_text() == "retain me\n"
    assert list(source.iterdir()) == [sentinel]


@pytest.fixture(scope="module")
def real_inputs(tmp_path_factory):
    from benchmarks.performance.workloads import materialize

    root = tmp_path_factory.mktemp("report-workloads")
    return root, materialize(root)


@pytest.fixture
def suggestion_evidence(tmp_path, real_inputs, request):
    import pandas as pd

    from bo_forge.config import CampaignConfig
    from bo_forge.validation import canonical_columns

    route = getattr(request, "param", "log_ei")
    inputs_root, inputs = real_inputs
    item = inputs[route]
    root, base = tmp_path, Path("sample")
    directory = root / base
    directory.mkdir()
    for key in ("config", "log"):
        source = inputs_root / item[f"{key}_path"]
        (directory / source.name).write_bytes(source.read_bytes())
    config = CampaignConfig.from_yaml(directory / "campaign.yaml")
    observed_source = "qlog_ei" if route == "log_ei" else route
    rows = []
    for index, fraction in enumerate((.31, .38)):
        row = dict.fromkeys(canonical_columns(config), "")
        row.update(row_id=f"report_{index}", iteration=10, status="suggested",
                   source=observed_source)
        row.update({variable.name: variable.lower + fraction * (variable.upper - variable.lower)
                    for variable in config.variables})
        if config.fidelity is not None:
            row[config.fidelity.variable] = 1.0
        rows.append(row)
    pd.DataFrame(rows, columns=canonical_columns(config)).to_csv(
        directory / "suggestions.csv", index=False)
    row = next(item for item in schedule() if item["case_id"] == f"{route}-q2"
               and item["version"] == "baseline" and item["repetition"] == 1)
    installation = {"prefix": "/synthetic/venv", "version": "3.3.3"}
    request_data = {**row, "expected_source": route, "installation_root": installation["prefix"],
                    "config_path": str(directory / "campaign.yaml"),
                    "log_path": str(directory / "campaign.csv"),
                    "config_sha256": item["config_sha256"], "log_sha256": item["log_sha256"]}
    worker = {
        "status": "complete", "inputs_unchanged": True, "case_id": row["case_id"],
        "batch_size": 2, "expected_source": route, "observed_sources": [observed_source],
        "package_locations": {"bo_forge": "/synthetic/venv/lib/bo_forge/__init__.py"},
        "package_versions": {"bo_forge": "3.3.3"},
        "input_sha256": {key: item[f"{key}_sha256"] for key in ("config", "log")},
        "validation": {key: True for key in (
            "expected_source", "batch_size", "canonical_schema", "unobserved",
            "feasible", "unique")},
    }
    return {"root": root, "base": base, "row": row, "request": request_data,
            "worker": worker, "inputs": item,
            "metadata": {"installations": {"baseline": installation}}}


@pytest.mark.parametrize("suggestion_evidence", ["log_ei", "qlog_ehvi", "qmf_kg"], indirect=True)
def test_suggestion_validation_accepts_real_route_inputs_without_fitting(suggestion_evidence):
    report._suggestions(**suggestion_evidence)


@pytest.mark.parametrize("key,value", [
    ("case_id", "qmf_kg-q4"), ("batch_size", 4), ("status", "failed"),
    ("expected_source", "random"), ("inputs_unchanged", False),
    ("package_versions", {"bo_forge": "0.0.0"}),
    ("package_locations", {"bo_forge": "/synthetic/venv-other/bo_forge/__init__.py"}),
])
def test_suggestions_reject_wrong_worker_binding(suggestion_evidence, key, value):
    suggestion_evidence["worker"][key] = value
    with pytest.raises(ValueError):
        report._suggestions(**suggestion_evidence)


def test_suggestions_reject_request_source_mismatch(suggestion_evidence):
    suggestion_evidence["request"]["expected_source"] = "random"
    with pytest.raises(ValueError, match="suggestion evidence"):
        report._suggestions(**suggestion_evidence)


@pytest.mark.parametrize("fault", ["source", "count", "schema", "duplicate", "initial-design"])
def test_suggestions_reject_malformed_retained_csv(suggestion_evidence, fault):
    import pandas as pd

    from bo_forge.config import CampaignConfig
    from bo_forge.errors import BOForgeError

    directory = suggestion_evidence["root"] / suggestion_evidence["base"]
    config = CampaignConfig.from_yaml(directory / "campaign.yaml")
    path = directory / "suggestions.csv"
    frame = pd.read_csv(path)
    if fault == "source":
        frame["source"] = "random"
    elif fault == "count":
        frame = frame.iloc[:1]
    elif fault == "schema":
        frame = frame.drop(columns="source")
    elif fault == "duplicate":
        frame.loc[1, config.variable_names] = frame.loc[0, config.variable_names].to_numpy()
    else:
        initial = pd.read_csv(directory / "campaign.csv")
        frame.loc[0, config.variable_names] = initial.loc[0, config.variable_names].to_numpy()
    frame.to_csv(path, index=False)
    with pytest.raises((ValueError, BOForgeError)):
        report._suggestions(**suggestion_evidence)
