"""Read-only suggestion JSON preserves backend guards and campaign identities."""

import pandas as pd
import pytest

from bo_forge._cli.output import table_payload
from bo_forge.cli import run
from bo_forge.session import CampaignSession
from tests._cli_support import write_config
from tests.test_cli_json import ROOT, response


def snapshot(directory):
    return {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}


def copy_example(directory, prefix):
    config, log = directory / "campaign.yaml", directory / "campaign.csv"
    config.write_bytes(next((ROOT / "configs").glob(f"{prefix}_*.yaml")).read_bytes())
    log.write_bytes(next((ROOT / "examples").glob(f"{prefix}_*campaign_log.csv")).read_bytes())
    return config, log


def arguments(config, log):
    return ["suggest", "--config", str(config), "--log", str(log), "--format=json"]


def test_real_initial_short_batch_and_no_reservation(tmp_path, capsys):
    config = write_config(tmp_path / "campaign.yaml", initial_design_size=2)
    log = tmp_path / "campaign.csv"
    campaign = CampaignSession.initialize(config, log)
    before = snapshot(tmp_path)
    assert run([*arguments(config, log), "--batch-size", "4", "--require-provenance"]) == 0
    payload, _ = response(capsys)
    assert payload["data"]["columns"] == list(campaign.df.columns)
    assert len(payload["data"]["records"]) == 2
    assert {row["source"] for row in payload["data"]["records"]} == {"sobol"}
    assert snapshot(tmp_path) == before
    assert CampaignSession.from_files(config, log).df.empty


@pytest.mark.parametrize("prefix,initial", [
    ("10", True), ("14", True), ("20", True), ("21", True), ("22", True), ("20", False),
])
def test_generated_route_metadata_survives_json(tmp_path, monkeypatch, capsys, prefix, initial):
    from bo_forge.costs import evaluate_cost

    config, log = copy_example(tmp_path, prefix)
    if initial:
        log.unlink()
        campaign = CampaignSession.initialize(config, log)
    else:
        campaign = CampaignSession.from_files(config, log)
    before = snapshot(tmp_path)
    generated = []
    original = CampaignSession.suggest_next

    def capture(self, **kwargs):
        frame = original(self, **kwargs)
        generated.append(frame)
        return frame

    monkeypatch.setattr(CampaignSession, "suggest_next", capture)
    flags = ["--batch-size", "1"]
    if campaign.config.is_structured_campaign:
        flags += ["--stage", campaign.config.stages[0].name]
    if campaign.config.context:
        flags += ["--context", "feedstock_acidity=0.4"]
    assert run([*arguments(config, log), *flags]) == 0
    payload, _ = response(capsys)
    assert len(generated) == 1
    assert payload["data"] == table_payload(generated[0])
    assert payload["data"]["columns"] == list(campaign.df.columns)
    row, = payload["data"]["records"]
    assert row["source"] == ("sobol" if initial else "cost_log_ei")
    if campaign.config.review.enabled:
        assert row["review_status"] == "pending"
    if campaign.config.context:
        assert row["feedstock_acidity"] == 0.4
    if campaign.config.cost:
        candidate = tuple(row[name] for name in campaign.config.variable_names)
        assert row["cost_estimate"] == pytest.approx(evaluate_cost(campaign.config, candidate))
        assert row["cost_actual"] == ""
    if not initial:
        assert row["utility"] == pytest.approx(
            row["acquisition"] - campaign.config.cost.weight * row["cost_estimate"],
        )
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("value", [None, "0.4"])
def test_real_context_defaults_and_overrides(tmp_path, capsys, value):
    config, log = copy_example(tmp_path, "16")
    log.unlink()
    CampaignSession.initialize(config, log)
    before = snapshot(tmp_path)
    flags = [] if value is None else ["--context", f"feedstock_acidity={value}"]
    assert run([*arguments(config, log), *flags]) == 0
    rows = response(capsys)[0]["data"]["records"]
    assert all(row["feedstock_acidity"] == (0.5 if value is None else float(value)) for row in rows)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("prefix,flags,marker", [
    ("01", ["--batch-size", "0"], "batch_size must"),
    ("01", ["--batch-size", "-1"], "batch_size must"),
    ("22", ["--batch-size", "5"], "qMFKG supports"),
    ("13", [], "explicit stage"),
    ("13", ["--stage", "unknown"], "stage"),
    ("01", ["--stage", "screen"], "--stage is only valid"),
    ("16", ["--context", "feedstock_acidity=nan"], "context"),
    ("16", ["--context", "feedstock_acidity=5"], "context"),
    ("16", ["--context", "acidity"], "--context"),
    ("16", ["--context", "acidity=0.4", "--context", "acidity=0.2"], "Duplicate"),
])
def test_backend_errors_are_json_and_nonmutating(tmp_path, capsys, prefix, flags, marker):
    config, log = copy_example(tmp_path, prefix)
    before = snapshot(tmp_path)
    assert run([*arguments(config, log), *flags]) == 1
    payload, stderr = response(capsys)
    assert payload["error"]["code"] == "suggestion_error" and payload["data"] is None
    assert marker.lower() in stderr.lower()
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("prefix,acquisition", [("18", "qlog_nei"), ("19", "qlog_nehvi")])
@pytest.mark.parametrize("review", ["pending", "accepted"])
def test_noisy_pending_semantics_survive_json(tmp_path, monkeypatch, capsys,
                                            prefix, acquisition, review):
    from bo_forge._optimization import router

    config, log = copy_example(tmp_path, prefix)
    frame = pd.read_csv(log, keep_default_na=False)
    frame.loc[frame.status == "suggested", "review_status"] = review
    frame.to_csv(log, index=False)
    before = snapshot(tmp_path)
    calls = []

    def model_based(**kwargs):
        calls.append(kwargs)
        assert kwargs["active_pending_df"].review_status.eq("accepted").all()
        assert len(kwargs["active_pending_df"]) == 1
        assert kwargs["observed_df"].status.eq("observed").all()
        return frame.iloc[:1].assign(status="suggested", source=acquisition)

    monkeypatch.setattr(router, f"_suggest_{acquisition}_model_based", model_based)
    assert run(arguments(config, log)) == (1 if review == "pending" else 0)
    payload, _ = response(capsys)
    assert len(calls) == (0 if review == "pending" else 1)
    if review == "pending":
        assert payload["error"]["code"] == "suggestion_error"
        assert "review_status='pending'" in payload["error"]["message"]
    else:
        assert payload["data"]["records"][0]["source"] == acquisition
    assert snapshot(tmp_path) == before


def test_exhausted_budget_is_nonmutating(tmp_path, capsys):
    config, log = copy_example(tmp_path, "07")
    config.write_text(config.read_text().replace("budget: 60.0", "budget: 0.1"))
    frame = pd.read_csv(log, keep_default_na=False)
    frame = frame.loc[frame.status == "observed"].iloc[:1]
    frame.to_csv(log, index=False)
    before = snapshot(tmp_path)
    assert run(arguments(config, log)) == 1
    payload, _ = response(capsys)
    assert payload["error"]["code"] == "suggestion_error"
    assert "budget" in payload["error"]["message"]
    assert snapshot(tmp_path) == before


def test_missing_context_default_fails_without_writes(tmp_path, capsys):
    config, log = copy_example(tmp_path, "16")
    config.write_text(config.read_text().replace(
        "  default_values:\n    feedstock_acidity: 0.5\n", "",
    ))
    before = snapshot(tmp_path)
    assert run(arguments(config, log)) == 1
    payload, _ = response(capsys)
    assert payload["error"]["code"] == "suggestion_error"
    assert "context" in payload["error"]["hint"].lower()
    assert snapshot(tmp_path) == before


def test_standard_pending_blocks_preview_without_writes(tmp_path, capsys):
    config = write_config(tmp_path / "campaign.yaml")
    log = tmp_path / "campaign.csv"
    campaign = CampaignSession.initialize(config, log)
    campaign.append_suggestions(campaign.suggest_next())
    before = snapshot(tmp_path)
    assert run(arguments(config, log)) == 1
    payload, _ = response(capsys)
    assert payload["error"]["code"] == "suggestion_error"
    assert "unresolved" in payload["error"]["message"]
    assert payload["data"] is None and snapshot(tmp_path) == before


def test_serialization_failure_preserves_managed_files(tmp_path, monkeypatch, capsys):
    config = write_config(tmp_path / "campaign.yaml")
    log = tmp_path / "campaign.csv"
    CampaignSession.initialize(config, log)
    manifest = tmp_path / "campaign.csv.manifest.json"
    (tmp_path / "previous.manifest.json").write_bytes(manifest.read_bytes())
    before = snapshot(tmp_path)
    calls = []

    def suggest(self, **kwargs):
        calls.append(1)
        return pd.DataFrame({"unsupported": [object()]})

    monkeypatch.setattr(CampaignSession, "suggest_next", suggest)
    assert run(arguments(config, log)) == 1
    payload, _ = response(capsys)
    assert payload["error"]["code"] == "serialization_error" and payload["data"] is None
    assert calls == [1] and snapshot(tmp_path) == before


def test_unexpected_suggestion_exception_propagates(tmp_path, monkeypatch, capsys):
    config = write_config(tmp_path / "campaign.yaml")
    log = tmp_path / "campaign.csv"
    CampaignSession.initialize(config, log)
    before = snapshot(tmp_path)

    def bug(*args, **kwargs):
        raise RuntimeError("programming error")

    monkeypatch.setattr(CampaignSession, "suggest_next", bug)
    with pytest.raises(RuntimeError, match="programming error"):
        run(arguments(config, log))
    assert capsys.readouterr().out == "" and snapshot(tmp_path) == before
