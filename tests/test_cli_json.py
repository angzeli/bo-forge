"""Versioned inspection contracts use actual backend results, never parsed text."""

import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd
import pytest

from bo_forge import __version__
from bo_forge._cli.output import (
    INSPECTION_COMMANDS,
    SerializationError,
    json_value,
    render,
    table_payload,
)
from bo_forge.cli import build_parser, run
from bo_forge.session import CampaignSession

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schemas/cli-inspection-v1.json").read_text())
EXAMPLES = {
    "cost-summary": "07", "replicate-summary": "08", "stage-summary": "14",
    "context-summary": "16", "fidelity-summary": "15", "fidelity-coverage": "22",
    "qlog-nei-summary": "18", "pareto-front": "10", "pareto-summary": "10",
    "model-summary": "17", "model-compare": "17",
}


def example_args(command):
    prefix = EXAMPLES.get(command, "01")
    config = next((ROOT / "configs").glob(f"{prefix}_*.yaml"))
    log = next((ROOT / "examples").glob(f"{prefix}_*campaign_log.csv"))
    return ["--config", str(config), "--log", str(log)]


def validate_contract(payload):
    jsonschema.validate(payload, SCHEMA)
    data = payload["data"]
    if isinstance(data, dict) and "columns" in data:
        assert all(set(row) == set(data["columns"]) for row in data["records"])


def response(capsys):
    captured = capsys.readouterr()
    assert captured.out.endswith("\n") and len(captured.out.splitlines()) == 1
    payload = json.loads(captured.out, parse_constant=lambda value: pytest.fail(value))
    validate_contract(payload)
    return payload, captured.err


@pytest.mark.parametrize("command", INSPECTION_COMMANDS)
def test_every_inspection_matches_backend_once(command, monkeypatch, capsys):
    args = example_args(command)
    before = {path: Path(path).read_bytes() for path in (args[1], args[3])}
    session = CampaignSession.from_files(args[1], args[3])
    if command == "model-compare":
        frame = pd.DataFrame({"model_profile": ["rough", "default"],
                              "fit_status": ["failed", "insufficient_observed"]})
        calls = []

        def compare(self, profiles=None):
            calls.append(profiles)
            print("backend progress")
            warnings.warn("captured fit warning", UserWarning, stacklevel=2)
            return frame

        monkeypatch.setattr(CampaignSession, "model_profile_comparison", compare)
        args += ["--profile", "rough", "--profile", "default"]
        expected = table_payload(frame)
    elif command == "validate":
        expected = {"valid": True}
    elif command == "status":
        expected = {"status": session.campaign_status()}
    else:
        expected = table_payload(getattr(session, command.replace("-", "_")
                                        if command != "provenance" else "provenance_summary")())
    with warnings.catch_warnings(record=True):
        assert run([command, *args, "--format", "json"]) == 0
    payload, stderr = response(capsys)
    assert payload["data"] == expected
    assert payload["ok"] and payload["error"] is None
    if command == "model-compare":
        assert calls == [["rough", "default"]]
        assert "backend progress" in stderr
    assert all(Path(path).read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("command", INSPECTION_COMMANDS)
@pytest.mark.parametrize("flag", [["--format=json"], ["--format", "json"]])
def test_missing_arguments_are_json(command, flag, capsys):
    assert run([command, *flag]) == 2
    payload, stderr = response(capsys)
    assert payload["command"] == command
    assert payload["error"]["code"] == "argument_error"
    assert "required" in stderr


@pytest.mark.parametrize("command", ["report", "init-log", "review",
                                      "mark-observed", "doctor", "model-evaluate", "plot",
                                      "provenance-recover"])
def test_deferred_commands_reject_format_before_handler(command, monkeypatch, capsys):
    import bo_forge.cli as cli

    args = [] if command == "doctor" else example_args(command)
    args += {
        "review": ["--row-id", "1", "--decision", "accept"],
        "mark-observed": ["--row-id", "1", "--objective-value", "1"],
        "plot": ["--kind", "progress", "--output", "unused.png"],
    }.get(command, [])
    parser = build_parser()
    parsed = parser.parse_args([command, *args])
    subparser = parser._subparsers._group_actions[0].choices[command]
    assert callable(parsed.handler)
    subparser.set_defaults(handler=lambda _: pytest.fail("handler executed"))
    monkeypatch.setattr(cli, "build_parser", lambda: parser)
    assert run([command, *args, "--format=json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unrecognized arguments: --format=json" in captured.err


@pytest.mark.parametrize("extra", [["--profile", "unknown"], ["--unknown"],
                                    ["--format=json", "--format"], ["--profile"]])
def test_parser_errors_never_execute(extra, monkeypatch, capsys):
    monkeypatch.setattr(CampaignSession, "from_files", lambda *a, **kw: pytest.fail("loaded"))
    assert run(["model-compare", *example_args("model-compare"), "--format=json", *extra]) == 2
    assert response(capsys)[0]["error"]["code"] == "argument_error"


def test_golden_payloads_and_schema(capsys):
    jsonschema.Draft202012Validator.check_schema(SCHEMA)
    assert run(["validate", *example_args("validate"), "--format=json"]) == 0
    validate = response(capsys)[0]
    assert run(["summary", "--format=json"]) == 2
    argument = response(capsys)[0]
    frame = pd.DataFrame({"name": ["001", "温度"], "value": [float("nan"),
        0.12345678901234566], "flag": [True, False]})
    table = json.loads(render("summary", table_payload(frame)))
    for name, payload in (("validate", validate), ("argument_error", argument), ("table", table)):
        golden = json.loads((ROOT / f"tests/fixtures/cli_json/{name}.json").read_text())
        golden["bo_forge_version"] = __version__
        assert payload == golden
        validate_contract(payload)


def test_empty_and_native_values():
    assert table_payload(pd.DataFrame(columns=["a"])) == {"columns": ["a"], "records": []}
    assert json_value([pd.NA, pd.NaT, np.nan, np.inf, -np.inf]) == [None] * 5
    assert json_value([np.bool_(True), np.int64(3), np.float64(0.25)]) == [True, 3, 0.25]
    assert json_value("001") == "001"
    for value in (object(), Path("file"), complex(1), {1: "value"}):
        with pytest.raises(SerializationError):
            json_value(value)


def test_serialization_failure_and_unexpected_exception(monkeypatch, capsys):
    monkeypatch.setattr(CampaignSession, "summary", lambda self: pd.DataFrame({"x": [object()]}))
    assert run(["summary", *example_args("summary"), "--format=json"]) == 1
    payload, _ = response(capsys)
    assert payload["error"]["code"] == "serialization_error" and payload["data"] is None

    def bug(self):
        raise RuntimeError("programming error")

    monkeypatch.setattr(CampaignSession, "summary", bug)
    with pytest.raises(RuntimeError, match="programming error"):
        run(["summary", *example_args("summary"), "--format=json"])
    assert capsys.readouterr().out == ""


def test_module_json_and_lightweight_import():
    script = """
import sys
from bo_forge.cli import run
assert run(sys.argv[1:]) == 0
heavy = {'torch', 'botorch', 'gpytorch', 'matplotlib', 'streamlit', 'fastapi'}
assert not heavy & sys.modules.keys()
"""
    completed = subprocess.run([sys.executable, "-c", script, "validate",
        *example_args("validate"), "--format=json"], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    jsonschema.validate(json.loads(completed.stdout), SCHEMA)


def test_bounded_real_model_comparison(capsys):
    assert run(["model-compare", *example_args("model-compare"),
                "--profile", "default", "--format=json"]) == 0
    payload, _ = response(capsys)
    assert payload["data"]["records"][0]["fit_status"] == "ok"


@pytest.mark.parametrize("command", INSPECTION_COMMANDS)
def test_managed_inspections_preserve_campaign_and_archives(command, tmp_path, monkeypatch, capsys):
    from bo_forge import adopt_provenance

    args = example_args(command)
    config, log = tmp_path / "config.yaml", tmp_path / "log.csv"
    shutil.copyfile(args[1], config)
    shutil.copyfile(args[3], log)
    preview = adopt_provenance(config, log)
    adopt_provenance(config, log, apply=True, reason="JSON acceptance",
                     expected_identities=preview["expected_identities"])
    (tmp_path / "archive.json").write_bytes((tmp_path / "log.csv.manifest.json").read_bytes())
    if command == "model-compare":
        monkeypatch.setattr(CampaignSession, "model_profile_comparison",
                            lambda *a, **kw: pd.DataFrame({"fit_status": ["failed"]}))
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    assert run([command, "--config", str(config), "--log", str(log),
                "--require-provenance", "--format=json"]) == 0
    assert response(capsys)[0]["ok"]
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == before
