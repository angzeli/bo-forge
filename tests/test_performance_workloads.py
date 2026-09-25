"""Bounded input generation and controlled first-call worker contracts, without fitting."""

import ast
import hashlib
import json
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from benchmarks.definitions import campaign_config
from benchmarks.performance import worker, workloads
from benchmarks.spec import load_spec, schedule
from bo_forge import CampaignSession
from bo_forge.validation import canonical_columns


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    root = tmp_path_factory.mktemp("performance-inputs")
    return root, workloads.materialize(root)


def test_materialized_inputs_match_specs_and_exact_config22(frozen):
    root, routes = frozen
    assert set(routes) == {"log_ei", "qlog_ehvi", "qmf_kg"}
    for route, metadata in routes.items():
        config, log = (root / metadata[key] for key in ("config_path", "log_path"))
        session = CampaignSession.from_files(config, log)
        assert list(session.df.columns) == canonical_columns(session.config)
        assert len(session.df) == metadata["initial_observations"] == 6
        assert session.df.status.eq("observed").all()
        assert metadata["expected_source"] == route
        assert metadata["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
        assert metadata["log_sha256"] == hashlib.sha256(log.read_bytes()).hexdigest()
        assert metadata["settings"] == yaml.safe_load(config.read_text())
        assert sorted(path.name for path in config.parent.iterdir()) == [
            "campaign.csv", "campaign.yaml",
        ]
    for route, filename, name in [
        ("log_ei", "standard.yaml", "branin"),
        ("qlog_ehvi", "multi_objective_standard.yaml", "branin_currin"),
    ]:
        spec = load_spec(workloads.ROOT / "benchmarks/specs" / filename)
        trial = next(item for item in schedule(spec)
                     if item["seed"] == 0 and item["strategy"] == "bo"
                     and item["mode"] == "deterministic" and item["problem"]["name"] == name)
        assert routes[route]["settings"] == campaign_config(trial)
        assert routes[route]["seed"] == 0
        assert routes[route]["seeds"] == trial["seeds"]
    mf = routes["qmf_kg"]
    for key in ("config", "log"):
        original = workloads.ROOT / mf["origin"][f"{key}_path"]
        assert (root / mf[f"{key}_path"]).read_bytes() == original.read_bytes()
    assert mf["seed"] == mf["settings"]["bo"]["random_seed"] == 22


def test_initial_only_reproducible_generation(frozen, tmp_path, monkeypatch):
    calls = {"so": 0, "mo": 0}
    true_value, evaluate = workloads.true_value, workloads.evaluate

    def so(problem, point):
        calls["so"] += 1
        return true_value(problem, point)

    def mo(problem, point):
        calls["mo"] += 1
        return evaluate(problem, point)

    def forbidden(*args, **kwargs):
        pytest.fail("Input materialization must not initialize manifests or suggest candidates.")

    monkeypatch.setattr(workloads, "true_value", so)
    monkeypatch.setattr(workloads, "evaluate", mo)
    monkeypatch.setattr(CampaignSession, "suggest_next", forbidden)
    monkeypatch.setattr(CampaignSession, "initialize", forbidden)
    assert workloads.materialize(tmp_path) == frozen[1]
    assert calls == {"so": 6, "mo": 6}
    with pytest.raises(FileExistsError):
        workloads.materialize(tmp_path)


def _request(frozen, sample, route="log_ei", q=1):
    root, metadata = frozen
    item = metadata[route]
    return {"case_id": f"{route}-q{q}", "config_path": str(root / item["config_path"]),
            "log_path": str(root / item["log_path"]), "batch_size": q,
            "expected_source": route, "route": route, "kind": "suggest",
            "version": "baseline", "repetition": 0, "warmup": True,
            "sample_id": f"{route}-q{q}-0-baseline",
            "config_sha256": item["config_sha256"], "log_sha256": item["log_sha256"],
            "result_path": str(sample / "result.json"),
            "suggestions_path": str(sample / "suggestions.csv"), "installation_root": sys.prefix}


def _suggestions(session, q, source):
    rows = []
    for index in range(q):
        row = dict.fromkeys(canonical_columns(session.config), "")
        row.update(row_id=f"controlled_{index}", iteration=10, status="suggested", source=source)
        fraction = .31 + index * .07
        row.update({variable.name: variable.lower + fraction * (variable.upper - variable.lower)
                    for variable in session.config.variables})
        if session.config.fidelity is not None:
            row[session.config.fidelity.variable] = 1.0
        rows.append(row)
    return pd.DataFrame(rows, columns=canonical_columns(session.config))


@pytest.fixture
def controlled(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(worker, "_installed_packages",
                        lambda root: ({"bo_forge": "controlled"}, {}))
    monkeypatch.setattr(worker, "_check_bo_modules", lambda root: None)
    return tmp_path


@pytest.mark.parametrize("route,q", [("log_ei", 1), ("log_ei", 2), ("qlog_ehvi", 2), ("qmf_kg", 2)])
def test_worker_timer_warnings_read_only_success(frozen, controlled, monkeypatch, route, q):
    request = _request(frozen, controlled, route, q)
    clock = [10.0]
    original_load = CampaignSession.from_files
    before = {path: path.read_bytes() for path in Path(request["config_path"]).parent.iterdir()}

    def load(config, log):
        clock[0] += 100.0
        warnings.warn("controlled loading warning", UserWarning, stacklevel=1)
        return original_load(config, log)

    def suggest(session, *, batch_size):
        assert batch_size == q
        clock[0] += 2.5
        warnings.warn("controlled call warning", RuntimeWarning, stacklevel=1)
        source = "qlog_ei" if route == "log_ei" and q > 1 else route
        return _suggestions(session, q, source)

    monkeypatch.setattr(CampaignSession, "from_files", load)
    monkeypatch.setattr(CampaignSession, "suggest_next", suggest)
    monkeypatch.setattr(worker.time, "perf_counter", lambda: clock[0])
    result = worker.run_request(request)
    assert result["status"] == "complete"
    assert result["call_seconds"] == 2.5
    assert result["inputs_unchanged"] is True
    for key in ("kind", "route", "version", "repetition", "warmup", "sample_id"):
        assert result[key] == request[key]
    assert result["observed_sources"] == [result["required_source"]]
    assert all(result["validation"].values())
    assert result["peak_rss_bytes"] > 0
    assert result["peak_rss_scope"] == worker.RSS_SCOPE
    assert {item["message"] for item in result["warnings"]} >= {
        "controlled loading warning", "controlled call warning",
    }
    assert len(pd.read_csv(request["suggestions_path"])) == q
    assert json.loads(Path(request["result_path"]).read_text()) == result
    after = {path: path.read_bytes() for path in Path(request["config_path"]).parent.iterdir()}
    assert after == before


@pytest.mark.parametrize("defect", ["source", "q", "schema", "bounds", "observed", "objective",
                                   "duplicate", "row_id", "fidelity", "exception"])
def test_worker_rejects_invalid_samples(frozen, controlled, monkeypatch, defect):
    request = _request(frozen, controlled, "qmf_kg", 2)

    def suggest(session, *, batch_size):
        frame = _suggestions(session, batch_size, "qmf_kg")
        if defect == "source":
            frame["source"] = "random"
        elif defect == "q":
            frame = frame.iloc[:1]
        elif defect == "schema":
            frame = frame[list(reversed(frame.columns))]
        elif defect == "bounds":
            frame.loc[0, session.config.variable_names[0]] = -100.0
        elif defect == "observed":
            frame.loc[0, "status"] = "observed"
        elif defect == "objective":
            frame.loc[0, session.config.objective.name] = "not blank"
        elif defect == "duplicate":
            names = session.config.variable_names
            frame.loc[1, names] = frame.loc[0, names]
        elif defect == "row_id":
            frame.loc[0, "row_id"] = session.df.iloc[0].row_id
        elif defect == "fidelity":
            frame.loc[0, "fidelity"] = .6
        else:
            warnings.warn("warning before failure", UserWarning, stacklevel=1)
            raise RuntimeError("controlled fit failure")
        return frame

    monkeypatch.setattr(CampaignSession, "suggest_next", suggest)
    result = worker.run_request(request)
    assert result["status"] == "failed"
    assert result["error"]
    assert result["call_seconds"] is not None
    assert result["inputs_unchanged"] is True
    assert not Path(request["suggestions_path"]).exists()
    if defect == "exception":
        assert any(item["message"] == "warning before failure" for item in result["warnings"])


@pytest.mark.parametrize("raises", [False, True])
def test_worker_checks_input_bytes_even_after_call_failure(frozen, controlled, monkeypatch, raises):
    request = _request(frozen, controlled)
    log = controlled / "input.csv"
    log.write_bytes(Path(request["log_path"]).read_bytes())
    request["log_path"] = str(log)

    def suggest(session, *, batch_size):
        log.write_bytes(log.read_bytes() + b"\n")
        if raises:
            raise RuntimeError("controlled call failure")
        return _suggestions(session, batch_size, "log_ei")

    monkeypatch.setattr(CampaignSession, "suggest_next", suggest)
    result = worker.run_request(request)
    assert result["status"] == "failed"
    assert result["inputs_unchanged"] is False
    assert "Input bytes changed" in result["error"]
    assert not Path(request["suggestions_path"]).exists()


@pytest.mark.parametrize("platform,multiplier,unit", [
    ("linux", 1024, "KiB"), ("darwin", 1, "bytes"),
])
def test_rss_units(monkeypatch, platform, multiplier, unit):
    monkeypatch.setattr(worker.sys, "platform", platform)
    monkeypatch.setattr(worker, "resource", SimpleNamespace(
        RUSAGE_SELF=0, getrusage=lambda who: SimpleNamespace(ru_maxrss=123),
    ))
    result = worker._peak_rss()
    assert result["peak_rss_bytes"] == 123 * multiplier
    assert result["peak_rss_native_unit"] == unit


@pytest.mark.parametrize("condition", ["platform", "module", "missing", "error", "zero", "nan"])
def test_rss_unavailable_is_null(monkeypatch, condition):
    monkeypatch.setattr(worker.sys, "platform", "win32" if condition == "platform" else "linux")

    def usage(who):
        if condition == "error":
            raise OSError("RSS unavailable")
        if condition == "missing":
            return SimpleNamespace()
        return SimpleNamespace(ru_maxrss=float("nan") if condition == "nan" else 0)

    monkeypatch.setattr(worker, "resource", None if condition == "module" else SimpleNamespace(
        RUSAGE_SELF=0, getrusage=usage,
    ))
    result = worker._peak_rss()
    assert result["peak_rss_bytes"] is None
    assert result["peak_rss_status"] == "unavailable"


@pytest.mark.parametrize("key", ["config", "log"])
def test_request_hash_mismatch_fails_before_loading(frozen, controlled, monkeypatch, key):
    request = _request(frozen, controlled)
    request[f"{key}_sha256"] = "0" * 64

    def forbidden(*args, **kwargs):
        pytest.fail("Hash mismatch must be rejected before loading a campaign.")

    monkeypatch.setattr(CampaignSession, "from_files", forbidden)
    result = worker.run_request(request)
    assert result["status"] == "failed"
    assert f"{key} SHA-256 differs" in result["error"]
    assert result["call_seconds"] is None
    assert not Path(request["suggestions_path"]).exists()


def test_csv_write_failure_retains_failure_and_warnings(frozen, controlled, monkeypatch):
    request = _request(frozen, controlled)
    monkeypatch.setattr(CampaignSession, "suggest_next", lambda session, batch_size:
                        _suggestions(session, batch_size, "log_ei"))

    def fail_write(frame, path, **kwargs):
        Path(path).write_text("partial output")
        warnings.warn("controlled CSV warning", UserWarning, stacklevel=1)
        raise OSError("controlled disk error")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_write)
    result = worker.run_request(request)
    assert result["status"] == "failed"
    assert "controlled disk error" in result["error"]
    assert any(item["message"] == "controlled CSV warning" for item in result["warnings"])
    assert json.loads(Path(request["result_path"]).read_text()) == result
    assert not Path(request["suggestions_path"]).exists()


def test_rejects_repo_import_and_wrong_venv(tmp_path):
    with pytest.raises(ValueError, match="installation_root"):
        worker._installed_packages(tmp_path)
    with pytest.raises(ValueError, match="requested installation"):
        worker._check_bo_modules(tmp_path)


def test_worker_standalone_source_contract():
    source = Path(worker.__file__).read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] in sys.stdlib_module_names for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module.split(".")[0] in sys.stdlib_module_names
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name.startswith("benchmarks") for name in imports)
    assert "sys.path" not in source.replace("sys.path modifications", "")
    assert "CampaignSession.initialize(" not in source
    assert ".append_suggestions(" not in source
    assert ".mark_observed(" not in source
    assert "evaluate_true(" not in source
    assert "session.suggest_next(batch_size=q)" in source


def test_isolated_subprocess_checks_installation_without_fitting(frozen, tmp_path):
    if sys.prefix == sys.base_prefix:
        pytest.skip("An installed-package virtual environment is needed for isolation smoke test.")
    request = _request(frozen, tmp_path, q=0)
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))
    command = [sys.executable, "-I", str(Path(worker.__file__).resolve()), str(request_path)]
    process = subprocess.run(command,
                             cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert process.returncode == 1, process.stderr
    result = json.loads(Path(request["result_path"]).read_text())
    if "imported outside installation_root" in result["error"]:
        # Editable installs must fail closed even with isolated Python startup.
        assert "bo_forge imported outside installation_root" in result["error"]
        assert result["package_locations"] == {}
    else:
        assert "batch_size must be a positive integer" in result["error"]
        assert set(result["package_locations"]) == set(worker.PACKAGES)
        assert all(Path(path).is_relative_to(Path(sys.prefix).resolve())
                   for path in result["package_locations"].values())
    assert result["call_seconds"] is None
    assert result["inputs_unchanged"] is True
    assert not Path(request["suggestions_path"]).exists()
