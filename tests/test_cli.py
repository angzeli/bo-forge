from pathlib import Path

import matplotlib
import pandas as pd
import pytest

from bo_forge.cli import run
from bo_forge.config import BOConfig, CampaignConfig, ObjectiveConfig, VariableConfig
from bo_forge.io import empty_campaign_log
from bo_forge.validation import canonical_columns

matplotlib.use("Agg")


def write_config(path: Path, *, initial_design_size: int = 2) -> Path:
    path.write_text(
        f"""
campaign_name: cli_test
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
bo:
  batch_size: 1
  initial_design_size: {initial_design_size}
  acquisition: log_ei
  random_seed: 7
  raw_samples: 16
  num_restarts: 2
  mc_samples: 16
""",
        encoding="utf-8",
    )
    return path


def config(initial_design_size: int = 2) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="cli_test",
        objective=ObjectiveConfig(name="score", direction="maximize"),
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=BOConfig(
            batch_size=1,
            initial_design_size=initial_design_size,
            random_seed=7,
            raw_samples=16,
            num_restarts=2,
            mc_samples=16,
        ),
    )


def observed_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "obs_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "x": 0.2,
                "score": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "obs_1",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "x": 0.8,
                "score": 1.5,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def write_log(path: Path, cfg: CampaignConfig, df: pd.DataFrame | None = None) -> Path:
    if df is None:
        df = empty_campaign_log(cfg)
    df.to_csv(path, index=False)
    return path


def base_args(config_path: Path, log_path: Path) -> list[str]:
    return ["--config", str(config_path), "--log", str(log_path)]


def test_version_outputs_clean_line(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["--version"]) == 0

    captured = capsys.readouterr()
    assert captured.out == "bo-forge 0.3.0\n"
    assert captured.err == ""


def test_validate_success_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config())

    assert run(["validate", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert captured.out == "Campaign log is valid.\n"
    assert captured.err == ""


def test_validate_failure_returns_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    cfg = config()
    invalid = observed_log(cfg)
    invalid.loc[0, "x"] = 2.0
    log_path = write_log(tmp_path / "campaign.csv", cfg, invalid)

    assert run(["validate", *base_args(config_path, log_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error:" in captured.err
    assert "outside bounds" in captured.err


def test_summary_status_next_action_and_report_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg))

    assert run(["summary", *base_args(config_path, log_path)]) == 0
    summary_out = capsys.readouterr().out
    assert "campaign_status" in summary_out
    assert "ready_for_bo" in summary_out

    assert run(["status", *base_args(config_path, log_path)]) == 0
    status_out = capsys.readouterr().out
    assert status_out == "ready_for_bo\n"

    assert run(["next-action", *base_args(config_path, log_path)]) == 0
    action_out = capsys.readouterr().out
    assert "suggest_bo" in action_out
    assert "ready_for_bo" in action_out

    assert run(["report", *base_args(config_path, log_path)]) == 0
    report_out = capsys.readouterr().out
    assert "BO Forge Campaign Report" in report_out
    assert "Best Observation" in report_out


def test_report_output_uses_export_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg))
    report_path = tmp_path / "reports" / "latest.txt"

    assert run(["report", *base_args(config_path, log_path), "--output", str(report_path)]) == 0

    captured = capsys.readouterr()
    assert captured.out == f"Wrote campaign report: {report_path}\n"
    assert report_path.exists()
    assert "BO Forge Campaign Report" in report_path.read_text(encoding="utf-8")


def test_suggest_without_append_does_not_change_csv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config())
    before_csv = log_path.read_text(encoding="utf-8")

    assert run(["suggest", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert "Generated 1 suggestion(s)." in captured.out
    assert "row_id" in captured.out
    assert log_path.read_text(encoding="utf-8") == before_csv


def test_suggest_append_changes_csv_but_does_not_mark_observed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config())

    assert run(["suggest", *base_args(config_path, log_path), "--append"]) == 0

    captured = capsys.readouterr()
    assert "Generated 1 suggestion(s)." in captured.out
    assert f"Appended suggestions to campaign log: {log_path}" in captured.out
    df = pd.read_csv(log_path)
    assert len(df) == 1
    assert df.loc[0, "status"] == "suggested"
    assert pd.isna(df.loc[0, "score"])


def test_suggest_output_and_append_writes_output_and_log(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config())
    output_path = tmp_path / "exports" / "suggestions.csv"

    assert (
        run(
            [
                "suggest",
                *base_args(config_path, log_path),
                "--output",
                str(output_path),
                "--append",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert f"Wrote suggestions CSV: {output_path}" in captured.out
    assert f"Appended suggestions to campaign log: {log_path}" in captured.out
    suggestions = pd.read_csv(output_path)
    log = pd.read_csv(log_path)
    assert len(suggestions) == 1
    assert len(log) == 1
    assert suggestions.loc[0, "row_id"] == log.loc[0, "row_id"]


def test_mark_observed_resolves_only_specified_pending_suggestion(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config(initial_design_size=3))
    assert run(["suggest", *base_args(config_path, log_path), "--batch-size", "2", "--append"]) == 0
    capsys.readouterr()
    pending = pd.read_csv(log_path)
    row_id = str(pending.loc[0, "row_id"])
    other_row_id = str(pending.loc[1, "row_id"])

    assert (
        run(
            [
                "mark-observed",
                *base_args(config_path, log_path),
                "--row-id",
                row_id,
                "--objective-value",
                "1.23",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out == f"Marked row {row_id} as observed in campaign log: {log_path}\n"
    df = pd.read_csv(log_path)
    assert df.loc[df["row_id"] == row_id, "status"].iloc[0] == "observed"
    assert float(df.loc[df["row_id"] == row_id, "score"].iloc[0]) == pytest.approx(1.23)
    assert df.loc[df["row_id"] == other_row_id, "status"].iloc[0] == "suggested"


@pytest.mark.parametrize("kind", ["progress", "diagnostics"])
def test_plot_writes_nested_output_path(
    kind: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg))
    output_path = tmp_path / "figures" / f"{kind}.png"
    before_csv = log_path.read_text(encoding="utf-8")

    assert (
        run(
            [
                "plot",
                *base_args(config_path, log_path),
                "--kind",
                kind,
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out == f"Wrote {kind} plot: {output_path}\n"
    assert output_path.exists()
    assert log_path.read_text(encoding="utf-8") == before_csv


def test_missing_required_arguments_return_argparse_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run(["validate"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage:" in captured.err
    assert "required" in captured.err
