"""Standalone installed-package sample: ``python -I /absolute/worker.py request.json``.

The caller sets cwd to a fresh sample directory and supplies case_id,
config_path, log_path, batch_size, expected_source, result_path,
suggestions_path, config_sha256, log_sha256, and installation_root (the venv
prefix). Relative paths resolve against cwd; output paths must be distinct
fresh files inside cwd.
The conventional names are result.json and suggestions.csv. No repo imports
or sys.path modifications are permitted; all initial imports are stdlib.

Result API: status (complete/failed), case_id, batch_size, expected_source,
call_seconds (only suggest_next, null if not called), peak_rss_bytes,
peak_rss_scope (whole worker lifetime, including imports/loading/validation),
peak_rss_native_unit, peak_rss_status (measured/unavailable; unavailable bytes
are null), package_locations, package_versions, warnings,
input_sha256, inputs_unchanged, validation, required_source, observed_sources,
and error on failure. Runner identity fields are echoed unchanged. For the
log_ei route, required_source is qlog_ei when batch_size > 1. Failed
samples exit 1 and never publish suggestions.csv. No campaign mutations or
manifest writes are performed. Warnings cover imports, loading and the call.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import sys
import time
import warnings
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None

PACKAGES = ("bo_forge", "torch", "botorch", "gpytorch", "numpy", "pandas", "scipy", "yaml")
RSS_SCOPE = "worker_process_lifetime_including_imports_session_loading_call_and_validation"


def _installed_packages(root: Path) -> tuple[dict, dict]:
    if Path(sys.prefix).resolve() != root or sys.prefix == sys.base_prefix:
        raise ValueError("installation_root must identify the active virtual environment.")
    locations, versions = {}, {}
    for name in PACKAGES:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"{name} imported outside installation_root: {path}")
        locations[name] = str(path)
        distribution = {"bo_forge": "bo-forge", "yaml": "PyYAML"}.get(name, name)
        versions[name] = importlib.metadata.version(distribution)
    _check_bo_modules(root)
    return locations, versions


def _check_bo_modules(root: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if name == "bo_forge" or name.startswith("bo_forge."):
            location = getattr(module, "__file__", None)
            if location is None or not Path(location).resolve().is_relative_to(root):
                raise ValueError(f"{name} is not from the requested installation: {location}")


def _input_hashes(paths: dict) -> dict:
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def _peak_rss() -> dict:
    result = {"peak_rss_bytes": None, "peak_rss_native_unit": None,
              "peak_rss_scope": RSS_SCOPE, "peak_rss_status": "unavailable"}
    if resource is None or sys.platform not in {"darwin", "linux"}:
        return result
    try:
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        value = int(raw * (1024 if sys.platform == "linux" else 1))
    except (AttributeError, OSError, TypeError, ValueError, OverflowError):
        return result
    if value <= 0:
        return result
    result.update(peak_rss_bytes=value, peak_rss_status="measured",
                  peak_rss_native_unit="KiB" if sys.platform == "linux" else "bytes")
    return result


def _validate_suggestions(session, suggestions, q: int, expected: str) -> dict:
    import pandas as pd

    from bo_forge.validation import canonical_columns, design_tuples, validate_campaign_data

    if not isinstance(suggestions, pd.DataFrame) or len(suggestions) != q:
        raise ValueError("Suggestion count differs from requested batch_size.")
    if list(suggestions.columns) != canonical_columns(session.config):
        raise ValueError("Suggestions do not have the exact canonical schema.")
    if not suggestions.source.eq(expected).all():
        raise ValueError("Suggestion source differs from required source; fallback rejected.")
    objectives = (session.config.objective_names if session.config.is_multi_objective
                  else [session.config.objective.name])
    blank = suggestions[objectives].isna() | suggestions[objectives].eq("")
    if not suggestions.status.eq("suggested").all() or not blank.all().all():
        raise ValueError("Suggestions must be unobserved; no objective evaluation is allowed.")
    validate_campaign_data(session.config, suggestions)
    validate_campaign_data(session.config, pd.concat([session.df, suggestions], ignore_index=True))
    designs = design_tuples(session.config, suggestions)
    if len(designs) != q or designs & design_tuples(session.config, session.df):
        raise ValueError("Duplicate suggestion design, within batch or against initial data.")
    return {"expected_source": True, "batch_size": True, "canonical_schema": True,
            "unobserved": True, "feasible": True, "unique": True}


def _sample(request: dict, result: dict, inputs: dict):
    root = Path(request["installation_root"]).resolve()
    locations, versions = _installed_packages(root)
    result.update(package_locations=locations, package_versions=versions)
    from bo_forge import CampaignSession

    q = request["batch_size"]
    if type(q) is not int or q < 1:
        raise ValueError("batch_size must be a positive integer.")
    if request["expected_source"] not in {"log_ei", "qlog_ei", "qlog_ehvi", "qmf_kg"}:
        raise ValueError("expected_source must be a supported model-based source.")
    session = CampaignSession.from_files(inputs["config"], inputs["log"])
    if not session.df.status.eq("observed").all():
        raise ValueError("Input log must contain observed initial rows only.")
    if len(session.df) != session.config.bo.initial_design_size:
        raise ValueError("Input count must equal initial_design_size.")
    required = ("qlog_ei" if request["expected_source"] == "log_ei" and q > 1
                else request["expected_source"])
    result["required_source"] = required
    start = time.perf_counter()
    try:
        suggestions = session.suggest_next(batch_size=q)
    finally:
        result["call_seconds"] = time.perf_counter() - start
    _check_bo_modules(root)
    result["validation"] = _validate_suggestions(session, suggestions, q, required)
    result["observed_sources"] = sorted(set(suggestions.source))
    return suggestions


def _output_paths(request: dict, inputs: dict) -> tuple[Path, Path]:
    paths = tuple(Path(request[key]).resolve() for key in ("result_path", "suggestions_path"))
    cwd = Path.cwd().resolve()
    for path in paths:
        if path.parent != cwd or path in inputs.values() or path.exists():
            raise ValueError("Outputs must be fresh files in the sample cwd, distinct from inputs.")
    if paths[0] == paths[1]:
        raise ValueError("Result and suggestions paths must differ.")
    return paths


def run_request(request: dict) -> dict:
    """Run one sample; persist failure evidence as well as successful results."""
    inputs = {key: Path(request[f"{key}_path"]).resolve() for key in ("config", "log")}
    result_path, suggestions_path = _output_paths(request, inputs)
    result = {"case_id": request["case_id"], "batch_size": request["batch_size"],
              "expected_source": request["expected_source"], "status": "failed",
              "call_seconds": None, "package_locations": {}, "package_versions": {},
              "validation": {}, "inputs_unchanged": False, "input_sha256": {}}
    for key in ("kind", "route", "version", "repetition", "warmup", "sample_id"):
        if key in request:
            result[key] = request[key]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result["input_sha256"] = _input_hashes(inputs)
            for key, actual in result["input_sha256"].items():
                if request[f"{key}_sha256"] != actual:
                    raise ValueError(f"{key} SHA-256 differs from the frozen request input.")
            suggestions = _sample(request, result, inputs)
            result["status"] = "complete"
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                result["inputs_unchanged"] = _input_hashes(inputs) == result["input_sha256"]
                if not result["inputs_unchanged"]:
                    raise ValueError("Input bytes changed during the sample.")
            except Exception as exc:
                result["status"] = "failed"
                result["error"] = f"{result.get('error', '')} {type(exc).__name__}: {exc}".strip()
        if result["status"] == "complete":
            try:
                suggestions.to_csv(suggestions_path, index=False)
            except Exception as exc:
                result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                suggestions_path.unlink(missing_ok=True)
    result["warnings"] = [{"category": item.category.__name__, "message": str(item.message),
                           "filename": item.filename, "lineno": item.lineno} for item in caught]
    result.update(_peak_rss())
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    if not sys.flags.isolated:
        raise ValueError("Launch this worker with python -I.")
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    return 0 if run_request(request)["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
