"""Failure, provenance read-only, and legacy text-mode JSON boundary checks."""

import json
import subprocess
import sys
import warnings

import pandas as pd
import pytest

from bo_forge import errors
from bo_forge._campaign.provenance import manifest_path_for_log
from bo_forge.cli import run
from bo_forge.session import CampaignSession
from tests._session_support import write_config
from tests.test_cli_json import example_args, response
from tests.test_provenance_resume import _pending_previous, _pending_resulting


@pytest.mark.parametrize("cls,code", [
    (errors.BOForgeError, "bo_forge_error"),
    (errors.ConfigError, "config_error"),
    (errors.LogValidationError, "log_validation_error"),
    (errors.LogWriteError, "log_write_error"),
    (errors.LogBusyError, "log_busy_error"),
    (errors.LogConflictError, "log_conflict_error"),
    (errors.ProvenanceError, "provenance_error"),
    (errors.ProvenanceRecoveryRequired, "provenance_recovery_required"),
    (errors.SuggestionError, "suggestion_error"),
])
def test_error_class_mapping(cls, code, monkeypatch, capsys):
    kwargs = {"reason_code": "pending_previous_state", "recovery_action": "recover"}
    exc = cls("failure", **kwargs) if cls is errors.ProvenanceRecoveryRequired else cls("failure")

    def fail(self):
        raise exc

    monkeypatch.setattr(CampaignSession, "summary", fail)
    assert run(["summary", *example_args("summary"), "--format=json"]) == 1
    payload, stderr = response(capsys)
    assert payload["error"]["code"] == code and payload["data"] is None
    assert "failure" in stderr


@pytest.mark.parametrize("state,code,reason", [
    ("ready", None, None), ("legacy", None, None),
    ("required", "provenance_error", "manifest_required"),
    ("malformed", "provenance_error", "manifest_invalid"),
    ("formatting", "log_conflict_error", "config_bytes_changed_semantics_same"),
    ("semantic", "log_conflict_error", "config_semantics_changed"),
    ("log_changed", "log_conflict_error", "log_hash_changed"),
    ("pending_previous", "provenance_recovery_required", "pending_previous_state"),
    ("pending_resulting", "provenance_recovery_required", "pending_resulting_state"),
])
def test_provenance_retains_evidence_without_writes(tmp_path, capsys, state, code, reason):
    config = write_config(tmp_path / "config.yaml")
    log = tmp_path / "log.csv"
    CampaignSession.initialize(config, log)
    manifest = manifest_path_for_log(log)
    archive = tmp_path / "prior.manifest.json"
    archive.write_bytes(manifest.read_bytes())
    if state in {"legacy", "required"}:
        manifest.unlink()
    elif state == "malformed":
        manifest.write_text("{broken")
    elif state == "formatting":
        config.write_text(config.read_text() + "\n# comment\n")
    elif state == "semantic":
        config.write_text(config.read_text().replace("random_seed: 5", "random_seed: 8"))
    elif state == "log_changed":
        log.write_bytes(log.read_bytes() + b"\n")
    elif state == "pending_previous":
        _pending_previous(config, log)
    elif state == "pending_resulting":
        _pending_resulting(config, log)
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    args = ["--config", str(config), "--log", str(log), "--format=json"]
    if state == "required":
        args.append("--require-provenance")
    assert run(["provenance", *args]) == (1 if code else 0)
    payload, _ = response(capsys)
    if code:
        assert payload["error"]["code"] == code
        assert payload["error"]["details"]["reason_code"] == reason
    if state not in {"required", "malformed"}:
        fields = {row["field"]: row["value"] for row in payload["data"]["records"]}
        assert fields["reason_code"] == reason
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_json_warnings_and_text_compatibility(monkeypatch, capsys):
    calls = []
    frame = pd.DataFrame({"a": [1], "b": [True]})

    def summary(self):
        calls.append(1)
        print("backend chatter")
        warnings.warn("fit warning", UserWarning, stacklevel=2)
        return frame

    monkeypatch.setattr(CampaignSession, "summary", summary)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        assert run(["summary", *example_args("summary"), "--format=json"]) == 0
    payload, stderr = response(capsys)
    assert "backend chatter" in stderr
    assert payload["data"]["records"] == [{"a": 1, "b": True}]
    assert calls == [1]
    with warnings.catch_warnings(record=True):
        for flags in ([], ["--format=text"]):
            assert run(["summary", *example_args("summary"), *flags]) == 0
            expected = "backend chatter\n" + frame.to_string(index=False) + "\n"
            assert capsys.readouterr().out == expected


def test_explicit_text_and_help_remain_text(capsys):
    for flag in ([], ["--format=text"]):
        assert run(["validate", *example_args("validate"), *flag]) == 0
        assert capsys.readouterr().out == "Campaign log is valid.\n"
    assert run(["summary", "--format=json", "--help"]) == 0
    assert capsys.readouterr().out.startswith("usage:")


def test_warnings_on_stderr_in_real_process():
    script = """
import sys, warnings
import pandas as pd
from bo_forge.session import CampaignSession
from bo_forge.cli import run
def summary(self):
    warnings.warn('fit warning', UserWarning)
    return pd.DataFrame({'value': [1]})
CampaignSession.summary = summary
raise SystemExit(run(sys.argv[1:]))
"""
    completed = subprocess.run([sys.executable, "-c", script, "summary",
        *example_args("summary"), "--format=json"], text=True, capture_output=True)
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["ok"]
    assert "fit warning" in completed.stderr


@pytest.mark.parametrize("command", ["stage-summary", "context-summary", "fidelity-summary",
                                    "fidelity-coverage", "qlog-nei-summary", "pareto-front",
                                    "pareto-summary"])
def test_unsupported_campaigns_keep_capability_errors(command, capsys):
    assert run([command, *example_args("validate"), "--format=json"]) == 1
    assert response(capsys)[0]["error"]["code"] == "config_error"


def test_failed_comparison_rows_do_not_change_exit_code(monkeypatch, capsys):
    frame = pd.DataFrame({"model_profile": ["smooth"], "fit_status": ["failed"],
                          "rmse_model_space": [float("nan")]})
    monkeypatch.setattr(CampaignSession, "model_profile_comparison", lambda *a, **kw: frame)
    assert run(["model-compare", *example_args("model-compare"), "--format=json"]) == 0
    payload, _ = response(capsys)
    assert payload["ok"] is True
    assert payload["data"]["records"][0]["rmse_model_space"] is None
    assert json.dumps(payload, allow_nan=False)
