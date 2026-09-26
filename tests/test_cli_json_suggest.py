"""Suggestion previews reuse the JSON contract without changing or reserving rows."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from bo_forge import __version__
from bo_forge._cli.output import INSPECTION_COMMANDS, JSON_COMMANDS, table_payload
from bo_forge.cli import run
from bo_forge.errors import SuggestionError
from bo_forge.session import CampaignSession
from tests._cli_support import write_config
from tests.test_cli_json import ROOT, SCHEMA, example_args, response


def test_json_registration_keeps_inspection_scope():
    assert len(INSPECTION_COMMANDS) == 16 and "suggest" not in INSPECTION_COMMANDS
    assert JSON_COMMANDS == (*INSPECTION_COMMANDS, "suggest")
    assert set(SCHEMA["properties"]["command"]["enum"]) == set(JSON_COMMANDS)


@pytest.mark.parametrize("prefix", ["01", "05", "06", "07", "08", "10", "12", "14",
                                    "15", "16", "18", "19", "20", "21", "22"])
def test_preview_forwards_once_and_preserves_route_table(prefix, monkeypatch, capsys):
    config = next((ROOT / "configs").glob(f"{prefix}_*.yaml"))
    log = next((ROOT / "examples").glob(f"{prefix}_*campaign_log.csv"))
    campaign = CampaignSession.from_files(config, log)
    frame = campaign.df.iloc[:2].iloc[::-1].copy()
    frame["status"] = "suggested"
    frame["source"] = campaign.config.bo.acquisition
    frame.attrs["private_fit_evidence"] = object()
    expected = table_payload(frame)
    before = {path: path.read_bytes() for path in (config, log)}
    calls = []

    def suggest(self, **kwargs):
        calls.append(kwargs)
        return frame

    monkeypatch.setattr(CampaignSession, "suggest_next", suggest)
    flags = ["--batch-size", "4"]
    stage = "screen" if campaign.config.is_structured_campaign else None
    context = (
        {campaign.config.context_variable_names[0]: "0.4"} if campaign.config.context else None
    )
    if stage:
        flags += ["--stage", stage]
    if context:
        flags += ["--context", f"{next(iter(context))}=0.4"]
    assert run(["suggest", "--config", str(config), "--log", str(log),
                *flags, "--format=json"]) == 0
    payload, _ = response(capsys)
    assert calls == [{"batch_size": 4, "stage": stage, "context_values": context}]
    assert payload["data"] == expected
    assert payload["command"] == "suggest" and payload["error"] is None
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("format_flags", [["--format=json"], ["--format", "json"],
                                          ["--for=json"]])
@pytest.mark.parametrize("write_flag", ["append", "output", "both"])
@pytest.mark.parametrize("format_first", [True, False])
def test_write_flags_fail_before_loading(tmp_path, monkeypatch, capsys,
                                        format_flags, write_flag, format_first):
    import bo_forge.cli as cli

    def forbidden(*args, **kwargs):
        pytest.fail("JSON write flags reached campaign or filesystem work")

    monkeypatch.setattr(cli, "_load_session", forbidden)
    monkeypatch.setattr(cli, "_write_csv", forbidden)
    monkeypatch.setattr(CampaignSession, "suggest_next", forbidden)
    monkeypatch.setattr(CampaignSession, "append_suggestions", forbidden)
    output = tmp_path / "not-created" / "suggestions.csv"
    flags = ["--append"] if write_flag in {"append", "both"} else []
    if write_flag in {"output", "both"}:
        flags += ["--output", str(output)]
    flags = format_flags + flags if format_first else flags + format_flags
    assert run(["suggest", *example_args("suggest"), *flags]) == 2
    payload, _ = response(capsys)
    assert payload["error"]["code"] == "argument_error" and payload["data"] is None
    assert not output.parent.exists()


@pytest.mark.parametrize("flags", [
    ["--format=json"], ["--format", "json"], ["--format=json", "--batch-size", "bad"],
    ["--format=json", "--format"], ["--format=json", "--format=invalid"],
    ["--unknown", "--format=json"], ["--format=text", "--format=json"],
    ["--format=json", "--", "--format=text"],
])
def test_parser_failures_keep_json_intent(flags, monkeypatch, capsys):
    monkeypatch.setattr(CampaignSession, "from_files", lambda *a, **kw: pytest.fail("loaded"))
    assert run(["suggest", *flags]) == 2
    assert response(capsys)[0]["error"]["code"] == "argument_error"


@pytest.mark.parametrize("flags", [["--format=json", "--format=text"],
                                    ["--", "--format=json"], ["--format=invalid"]])
def test_text_format_selection_preserved(flags, capsys):
    assert run(["suggest", *flags]) == 2
    assert capsys.readouterr().out == ""


def test_real_initial_suggestion_golden_contract(tmp_path, capsys):
    config = write_config(tmp_path / "config.yaml")
    log = tmp_path / "log.csv"
    CampaignSession.initialize(config, log)
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    assert run(["suggest", "--config", str(config), "--log", str(log), "--format=json"]) == 0
    payload, _ = response(capsys)
    row, = payload["data"]["records"]
    row["row_id"] = "preview_0"  # Only the generated UUID is nondeterministic.
    golden = json.loads((ROOT / "tests/fixtures/cli_json/suggest_initial.json").read_text())
    golden["bo_forge_version"] = __version__
    assert payload == golden
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("kind", ["model", "argument_error", "error"])
def test_suggestion_golden_contract(kind, monkeypatch, capsys):
    frame = pd.DataFrame([{
        "row_id": "preview_0", "iteration": 1, "status": "suggested",
        "source": "log_ei" if kind == "model" else "sobol",
        "x": 0.25, "score": "",
        "predicted_mean": 1.7 if kind == "model" else np.nan,
        "predicted_std": 0.2 if kind == "model" else np.nan,
        "acquisition": -0.5 if kind == "model" else np.nan,
    }])

    def suggest(self, **kwargs):
        if kind == "error":
            raise SuggestionError("controlled suggestion failure")
        return frame

    monkeypatch.setattr(CampaignSession, "suggest_next", suggest)
    flags = ["--append"] if kind == "argument_error" else []
    expected_code = {"error": 1, "argument_error": 2}.get(kind, 0)
    assert run(["suggest", *example_args("suggest"), "--format=json", *flags]) == expected_code
    payload, _ = response(capsys)
    golden = json.loads((ROOT / f"tests/fixtures/cli_json/suggest_{kind}.json").read_text())
    golden["bo_forge_version"] = __version__
    assert payload == golden


def test_progress_warnings_and_native_values(monkeypatch, capsys):
    frame = pd.DataFrame({"row_id": ["001", "温度"], "flag": [True, False],
                          "value": [np.inf, -np.inf], "score": [np.nan, 0.125]})
    frame.attrs["hidden"] = object()
    calls = []

    def suggest(self, **kwargs):
        calls.append(kwargs)
        print("optimizer progress")
        warnings.warn("optimizer warning", UserWarning, stacklevel=2)
        return frame

    monkeypatch.setattr(CampaignSession, "suggest_next", suggest)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        assert run(["suggest", *example_args("suggest"), "--format=json"]) == 0
    payload, stderr = response(capsys)
    assert "optimizer progress" in stderr
    assert any("optimizer warning" in str(warning.message) for warning in captured)
    assert "Generated" not in stderr and len(calls) == 1
    assert payload["data"] == table_payload(frame)


def test_contextual_replicate_note_stays_on_stderr(monkeypatch, capsys):
    config = next((ROOT / "configs").glob("21_*.yaml"))
    log = next((ROOT / "examples").glob("21_*campaign_log.csv"))
    frame = CampaignSession.from_files(config, log).df.iloc[:1].copy()
    frame["source"], frame["replicate_group"] = "cost_log_ei", "new_group"
    monkeypatch.setattr(CampaignSession, "suggest_next", lambda *a, **kw: frame)
    assert run(["suggest", "--config", str(config), "--log", str(log), "--format=json"]) == 0
    assert "No active repeat was selected" in response(capsys)[1]
