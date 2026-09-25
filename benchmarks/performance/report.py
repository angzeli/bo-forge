"""Validate retained identities and summarize paired samples without execution."""

import csv
import json
import math
import shutil
import statistics
import tempfile
from pathlib import Path

from benchmarks.performance import HARNESS_VERSION, cases, schedule
from benchmarks.performance.preparation import match_environments
from benchmarks.storage import write_json
from bo_forge._filesystem import rename_directory_exclusive
from notebook_assurance.archive import digest


def _path(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError(f"Evidence path escapes run: {relative}")
    return path


def _read(root, relative):
    return json.loads(_path(root, relative).read_text())


def _hash(root, relative, expected):
    if digest(_path(root, relative)) != expected:
        raise ValueError(f"Evidence identity mismatch: {relative}")


def load(root):
    root = Path(root).resolve()
    metadata = _read(root, "run.json")
    if metadata.get("schema_version") != 1 or metadata.get("harness_version") != HARNESS_VERSION:
        raise ValueError("Unsupported performance evidence/harness version.")
    if metadata.get("schedule") != schedule():
        raise ValueError("Performance schedule differs from the fixed acceptance matrix.")
    for name, value in metadata["harness_files"].items():
        _hash(root, f"harness/{name}", value)
    for name, value in metadata.get("shared_harness_files", {}).items():
        _hash(root, f"harness/shared/{name}", value)
    identities = metadata.get("installations", {})
    if set(identities) != {"baseline", "candidate"}:
        raise ValueError("Both installed environments are required for comparison.")
    match_environments(identities)
    for version, identity in identities.items():
        if _read(root, f"{version}/identity.json") != identity:
            raise ValueError(f"{version}: installation identity mismatch")
        _hash(root, f"{version}/{identity['archive']}", identity["archive_sha256"])
        _hash(root, f"{version}/{identity['wheel']}", identity["wheel_sha256"])
        _postflight(root, version, identity, metadata["status"])
    for value in metadata["inputs"].values():
        for key in ("config", "log"):
            _hash(root, f"inputs/{value[f'{key}_path']}", value[f"{key}_sha256"])
    rows = []
    for item in metadata["schedule"]:
        rows.append(_sample(root, item, metadata))
    return metadata, rows


def _postflight(root, version, identity, status):
    path = _path(root, f"{version}/postflight.json")
    if not path.exists():
        if status == "complete":
            raise ValueError(f"{version}: completed run lacks postflight evidence")
        return
    after = _read(root, f"{version}/postflight.json")
    for key in ("files", "packages", "python", "import_path"):
        if after.get(key) != identity[key]:
            raise ValueError(f"{version}: postflight {key} differs from installation")


def _sample(root, item, metadata):
    base = Path("samples") / item["sample_id"]
    path = _path(root, base / "status.json")
    if not path.exists():
        return {**item, "status": "not_run", "process_seconds": None,
                "call_seconds": None, "peak_rss_bytes": None}
    row = _read(root, base / "status.json")
    if any(row.get(key) != value for key, value in item.items()):
        raise ValueError(f"Sample identity mismatch: {item['sample_id']}")
    if row["status"] not in ("complete", "failed", "timeout", "interrupted", "not_run"):
        raise ValueError(f"Invalid sample status: {item['sample_id']}")
    for key in ("process_seconds", "call_seconds", "peak_rss_bytes"):
        value = row.get(key)
        if value is not None and (
                isinstance(value, bool) or not math.isfinite(value) or value <= 0):
            raise ValueError(f"Invalid measurement {key}: {item['sample_id']}")
    if row["status"] == "complete":
        _complete_sample(root, base, row, metadata)
    return row


def _complete_sample(root, base, row, metadata):
    if row.get("process_seconds") is None:
        raise ValueError(f"Complete sample lacks timing: {row['sample_id']}")
    hashes = row.get("artifact_hashes", {})
    required = {"request.json", "process.log"}
    if row["kind"] == "suggest":
        required.update(("result.json", "suggestions.csv"))
    if not required.issubset(hashes):
        raise ValueError(f"Complete sample lacks artifacts: {row['sample_id']}")
    for name, expected in hashes.items():
        _hash(root, base / name, expected)
    request = _read(root, base / "request.json")
    for key in ("sample_id", "case_id", "version", "repetition", "warmup", "kind"):
        if request.get(key) != row[key]:
            raise ValueError(f"Request identity mismatch: {row['sample_id']}: {key}")
    route = row.get("route", "qmf_kg")
    inputs = metadata["inputs"][route]
    for key in ("config", "log"):
        if request[f"{key}_sha256"] != inputs[f"{key}_sha256"]:
            raise ValueError(f"Sample input mismatch: {row['sample_id']}: {key}")
        _hash(root, base / Path(request[f"{key}_path"]).name, inputs[f"{key}_sha256"])
    if row["kind"] == "suggest":
        worker = _read(root, base / "result.json")
        _worker_identity(row, worker)
        _suggestions(root, base, row, request, worker, inputs, metadata)


def _worker_identity(row, worker):
    if (worker != row.get("worker")
            or any(row.get(key) != worker.get(key)
                   for key in ("call_seconds", "peak_rss_bytes"))):
        raise ValueError(f"Suggestion evidence mismatch: {row['sample_id']}")
    for key in ("sample_id", "version", "repetition", "warmup"):
        if worker.get(key) != row[key]:
            raise ValueError(f"Worker sample identity mismatch: {row['sample_id']}: {key}")
    if row["call_seconds"] is None:
        raise ValueError(f"Missing public suggestion-call timing: {row['sample_id']}")


def _suggestions(root, base, row, request, worker, inputs, metadata):
    import pandas as pd

    from bo_forge.config import CampaignConfig
    from bo_forge.validation import canonical_columns, design_tuples, validate_campaign_data

    expected = inputs["expected_source"]
    required = "qlog_ei" if expected == "log_ei" and row["batch_size"] > 1 else expected
    if (worker.get("status") != "complete" or worker.get("inputs_unchanged") is not True
            or worker.get("case_id") != row["case_id"]
            or worker.get("batch_size") != row["batch_size"]
            or request.get("expected_source") != expected
            or worker.get("expected_source") != expected):
        raise ValueError(f"Invalid suggestion evidence: {row['sample_id']}")
    installation = metadata["installations"][row["version"]]
    for name, location in worker["package_locations"].items():
        if not Path(location).is_relative_to(installation["prefix"]):
            raise ValueError(f"{name} import escaped measured installation")
    if worker["package_versions"]["bo_forge"] != installation["version"]:
        raise ValueError("Worker imported the wrong BO Forge version")
    config = CampaignConfig.from_yaml(_path(root, base / Path(request["config_path"]).name))
    initial = pd.read_csv(_path(root, base / Path(request["log_path"]).name))
    suggestions = pd.read_csv(_path(root, base / "suggestions.csv"))
    if (len(suggestions) != row["batch_size"]
            or list(suggestions.columns) != canonical_columns(config)
            or not suggestions.source.eq(required).all()
            or not suggestions.status.eq("suggested").all()):
        raise ValueError(f"Invalid retained suggestion batch: {row['sample_id']}")
    validate_campaign_data(config, pd.concat([initial, suggestions], ignore_index=True))
    designs = design_tuples(config, suggestions)
    if len(designs) != len(suggestions) or designs & design_tuples(config, initial):
        raise ValueError(f"Duplicate retained suggestion design: {row['sample_id']}")


def statistics_for(values):
    if len(values) != 5 or any(value is None for value in values):
        return None
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return {"median": statistics.median(values), "iqr": quartiles[2] - quartiles[0],
            "minimum": min(values), "maximum": max(values)}


def summarize(rows):
    summaries = []
    for case in cases():
        selected = [row for row in rows if row["case_id"] == case["case_id"]]
        complete = len(selected) == 12 and all(row["status"] == "complete" for row in selected)
        summary = {"case_id": case["case_id"], "comparable": complete,
                   "complete_samples": sum(row["status"] == "complete" for row in selected),
                   "scheduled_samples": 12, "measurements": {}}
        for metric in ("process_seconds", "call_seconds", "peak_rss_bytes"):
            groups = {version: statistics_for([row.get(metric) for row in selected
                       if row["version"] == version and not row["warmup"]])
                      for version in ("baseline", "candidate")}
            comparable = complete and all(groups.values())
            summary["measurements"][metric] = {
                **groups, "candidate_baseline_ratio":
                groups["candidate"]["median"] / groups["baseline"]["median"]
                if comparable else None}
        summaries.append(summary)
    return summaries


def report(source, destination):
    root, target = Path(source).resolve(), Path(destination).resolve()
    if target.exists() or target.is_relative_to(root) or root.is_relative_to(target):
        raise FileExistsError("Report must use a new directory outside retained evidence.")
    metadata, rows = load(root)
    summaries = summarize(rows)
    if metadata["status"] != "complete":
        for summary in summaries:
            summary["comparable"] = False
            for metric in summary["measurements"].values():
                metric["candidate_baseline_ratio"] = None
    valid = metadata["status"] == "complete" and all(row["comparable"] for row in summaries)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        write_json(temporary / "summary.json", {"valid": valid, "cases": summaries,
                   "machine": metadata["machine"], "comparison": "advisory",
                   "reporter_sha256": digest(Path(__file__)),
                   "archive_sha256": {k: v["archive_sha256"]
                                      for k, v in metadata["installations"].items()}})
        columns = ["sample_id", "case_id", "version", "repetition", "warmup", "status",
                   "process_seconds", "call_seconds", "peak_rss_bytes", "message"]
        with (temporary / "samples.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        (temporary / "report.md").write_text(_markdown(metadata, summaries, valid))
        rename_directory_exclusive(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return valid


def _markdown(metadata, summaries, valid):
    lines = ["# Paired Performance Evidence", "",
             f"Measurement validation: {'passed' if valid else 'FAILED / incomplete'}.", "",
             "Timing comparisons are advisory, not scientific or statistical superiority claims.",
             "Fresh processes on warmed filesystems; not cold-cache startup. Five measured repeats",
             "per version/case plus one retained warm-up. IQR uses inclusive quartiles.",
             "Process elapsed includes startup/shutdown; call elapsed covers suggestion only.",
             "RSS is worker lifetime peak resident memory in bytes, not incremental model memory.",
             "Missing memory/timing is unavailable, never zero. Incomplete pairs have no ratio.",
             "",
             f"Host: `{metadata['machine']}`", "",
             "| Case | Valid pair | Baseline median s | Candidate median s | Ratio |",
             "|---|---|---:|---:|---:|"]
    for row in summaries:
        metric = row["measurements"]["process_seconds"]
        values = [metric[version]["median"] if metric[version] else None
                  for version in ("baseline", "candidate")]
        values.append(metric["candidate_baseline_ratio"])
        rendered = [f"{value:.4f}" if value is not None else "not available" for value in values]
        lines.append(f"| {row['case_id']} | {row['comparable']} | " + " | ".join(rendered) + " |")
    lines.extend(["", "Raw samples: samples.csv. All metric distributions: summary.json.",
                  "Do not pool machines. Five repetitions do not establish significance.", ""])
    return "\n".join(lines)
