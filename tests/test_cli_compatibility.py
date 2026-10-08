"""Contracts captured before the v3.4.2 command-owner extraction."""

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import bo_forge.cli as cli
from bo_forge import __version__
from bo_forge._cli import campaign as campaign_commands
from bo_forge._cli.errors import _CLIOutputError
from bo_forge.session import CampaignSession
from tests._cli_support import write_config, write_cost_review_config

BASELINE = Path(__file__).parent / "fixtures" / "cli_parser_v341.json"


def parser_contract():
    """Capture declared options, defaults, ordering, and selected help text."""
    parser = cli.build_parser()
    subparsers = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    contract = {}
    for name, command in [("root", parser), *subparsers.choices.items()]:
        actions = []
        for action in command._actions:
            actions.append({
                "options": action.option_strings, "dest": action.dest,
                "action": type(action).__name__, "default": action.default,
                "required": action.required, "nargs": action.nargs,
                "choices": list(action.choices) if action.choices is not None else None,
                "type": getattr(action.type, "__name__", None),
                "metavar": action.metavar, "help": action.help,
            })
        contract[name] = {"actions": actions, "allow_abbrev": command.allow_abbrev}
        if name in {"root", "doctor", "init-log", "suggest", "plot", "model-evaluate"}:
            contract[name]["help"] = command.format_help().replace(__version__, "<version>")
    return contract


def test_parser_options_order_and_help_match_v341(monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    current = parser_contract()
    assert list(current) == list(baseline)
    assert current == baseline


def test_every_command_dispatches_to_its_owner_and_lifecycle_operation():
    from bo_forge._cli import (
        environment,
        inspection,
        model_evaluate,
        provenance,
        provenance_lifecycle,
    )

    owners = {
        environment: ["doctor", "plot"],
        campaign_commands: ["init-log", "suggest", "review", "mark-observed"],
        inspection: ["validate", "summary", "status", "next-action", "cost-summary",
                     "replicate-summary", "stage-summary", "fidelity-summary", "fidelity-coverage",
                     "context-summary", "qlog-nei-summary", "model-summary", "model-compare",
                     "pareto-front", "pareto-summary", "report"],
        model_evaluate: ["model-evaluate"],
        provenance: ["provenance", "provenance-recover"],
    }
    expected = {name: {"handler": getattr(owner, "_cmd_" + name.replace("-", "_"))}
                for owner, names in owners.items() for name in names}
    for operation in ["adopt", "migrate", "accept-config", "fork"]:
        expected["provenance-" + operation] = {
            "handler": provenance_lifecycle._run, "lifecycle_operation": operation,
        }
    parser = cli.build_parser()
    commands = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction)).choices
    assert {name: command._defaults for name, command in commands.items()} == expected


@pytest.mark.parametrize("stdout,stderr,returncode", [
    ('{"schema_version":1,"ok":true,"data":{"records":[]},"error":null}', "", 0),
    ("", "unexpected failure evidence", 1),
])
def test_documented_inspection_consumer_preserves_unexpected_failure(
    monkeypatch, stdout, stderr, returncode,
):
    import subprocess

    doc = (Path(__file__).parents[1] / "docs" / "CLI.md").read_text(encoding="utf-8")
    example = re.findall(r"```python\n(.*?)```", doc, re.DOTALL)[0]
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode))
    if returncode:
        with pytest.raises(RuntimeError, match="unexpected failure evidence"):
            exec(example, {})
    else:
        namespace = {}
        exec(example, namespace)
        assert namespace["rows"] == []


@pytest.mark.parametrize("command,expected", [
    ("validate", "Campaign log is valid.\n"),
    ("status", "controlled_status\n"),
    ("summary", "field  value\n rows      2\n"),
])
def test_representative_text_output(command, expected, tmp_path, monkeypatch, capsys):
    config = write_config(tmp_path / "campaign.yaml")
    log = tmp_path / "campaign.csv"
    CampaignSession.initialize(config, log)
    monkeypatch.setattr(CampaignSession, "campaign_status", lambda self: "controlled_status")
    monkeypatch.setattr(CampaignSession, "summary",
                        lambda self: pd.DataFrame({"field": ["rows"], "value": [2]}))
    assert cli.run([command, "--config", str(config), "--log", str(log)]) == 0
    captured = capsys.readouterr()
    assert captured.out == expected and captured.err == ""


@pytest.mark.parametrize("case,expected", [
    ("json_write", "suggest --format json cannot be combined"),
    ("provenance_before_context", "Provenance manifest path references do not match"),
    ("output_before_context", "Choose a separate artifact output path"),
    ("context_before_pending", "Malformed --context value"),
    ("objective_before_conflicting_flags", "Malformed --objective value"),
])
def test_first_error_selection_and_no_mutation(case, expected, tmp_path, monkeypatch, capsys):
    config = write_config(tmp_path / "campaign.yaml")
    log = tmp_path / "campaign.csv"
    campaign = CampaignSession.initialize(config, log)
    pending = campaign.suggest_next()
    campaign.append_suggestions(pending)
    manifest = tmp_path / "campaign.csv.manifest.json"
    (tmp_path / "previous.manifest.json").write_bytes(manifest.read_bytes())
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    monkeypatch.setattr(CampaignSession, "suggest_next",
                        lambda *a, **kw: pytest.fail("unexpected generation"))
    flags = ["--context", "malformed"]
    command = "suggest"
    if case in {"json_write", "provenance_before_context"}:
        config = tmp_path / "missing.yaml"
    if case == "json_write":
        flags += ["--append", "--format=json"]
    if case == "output_before_context":
        flags += ["--output", str(log)]
    if case == "objective_before_conflicting_flags":
        command = "mark-observed"
        flags = ["--row-id", str(pending.iloc[0].row_id),
                 "--objective", "malformed", "--objective-value", "1"]
    assert cli.run([command, "--config", str(config), "--log", str(log), *flags]) == (
        2 if case == "json_write" else 1
    )
    captured = capsys.readouterr()
    assert expected in captured.err
    if case == "json_write":
        assert json.loads(captured.out)["error"]["code"] == "argument_error"
    else:
        assert captured.out == ""
    assert {p: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("fail_export", [False, True])
def test_text_generation_export_append_order(tmp_path, monkeypatch, capsys, fail_export):
    config = write_config(tmp_path / "campaign.yaml")
    log = tmp_path / "campaign.csv"
    CampaignSession.initialize(config, log)
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    events = []
    suggest, append = CampaignSession.suggest_next, CampaignSession.append_suggestions
    write = campaign_commands._write_csv

    def generate(self, **kwargs):
        events.append("generate")
        return suggest(self, **kwargs)

    def export(*args):
        events.append("export")
        if fail_export:
            raise _CLIOutputError("controlled export failure")
        return write(*args)

    def mutate(self, frame, **kwargs):
        events.append("append")
        return append(self, frame, **kwargs)

    monkeypatch.setattr(CampaignSession, "suggest_next", generate)
    monkeypatch.setattr(CampaignSession, "append_suggestions", mutate)
    monkeypatch.setattr(campaign_commands, "_write_csv", export)
    output = tmp_path / "preview.csv"
    assert cli.run(["suggest", "--config", str(config), "--log", str(log),
                    "--output", str(output), "--append"]) == int(fail_export)
    captured = capsys.readouterr()
    assert captured.out.startswith("Generated 1 suggestion(s).\n")
    assert events == (["generate", "export"] if fail_export else ["generate", "export", "append"])
    if fail_export:
        assert captured.err == "Error: controlled export failure\n"
        assert {p: p.read_bytes() for p in tmp_path.iterdir()} == before
    else:
        assert captured.err == ""
        pd.testing.assert_frame_equal(pd.read_csv(output), pd.read_csv(log))


def test_review_and_observation_execute_once(tmp_path, monkeypatch, capsys):
    config = write_cost_review_config(tmp_path / "campaign.yaml")
    log = tmp_path / "campaign.csv"
    session = CampaignSession.initialize(config, log)
    pending = session.suggest_next()
    session.append_suggestions(pending)
    row_id = str(pending.iloc[0].row_id)
    calls = []
    review, observe = CampaignSession.review_suggestion, CampaignSession.mark_observed

    def reviewed(self, *args, **kwargs):
        calls.append(("review", args, kwargs))
        return review(self, *args, **kwargs)

    def observed(self, *args, **kwargs):
        calls.append(("observe", args, kwargs))
        return observe(self, *args, **kwargs)

    monkeypatch.setattr(CampaignSession, "review_suggestion", reviewed)
    monkeypatch.setattr(CampaignSession, "mark_observed", observed)
    common = ["--config", str(config), "--log", str(log), "--row-id", row_id]
    assert cli.run(["review", *common, "--decision", "accept", "--note", "reviewed"]) == 0
    assert cli.run(["mark-observed", *common, "--objective-value", "1.2",
                    "--actual-cost", "3.0"]) == 0
    assert calls == [("review", (row_id, "accept", "reviewed"), {}),
                     ("observe", (row_id,), {"objective_value": 1.2, "actual_cost": 3.0})]
    assert capsys.readouterr().err == ""
    assert CampaignSession.from_files(config, log).df.status.tolist() == ["observed"]
