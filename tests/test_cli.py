import subprocess
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

import bo_forge.cli as cli
from bo_forge.cli import run
from bo_forge.config import BOConfig, CampaignConfig, ObjectiveConfig, VariableConfig
from bo_forge.io import empty_campaign_log
from bo_forge.logs import load_campaign_log
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


def write_mixed_config(path: Path, *, initial_design_size: int = 3) -> Path:
    path.write_text(
        f"""
campaign_name: mixed_cli_test
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
  - name: repeats
    type: integer
    lower: 1
    upper: 3
  - name: dose
    type: discrete
    values: [0.1, 0.2, 0.5]
  - name: solvent
    type: categorical
    values: [MeCN, EtOH]
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


def mixed_config(initial_design_size: int = 3) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="mixed_cli_test",
        objective=ObjectiveConfig(name="score", direction="maximize"),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("repeats", "integer", 1.0, 3.0),
            VariableConfig("dose", "discrete", values=(0.1, 0.2, 0.5)),
            VariableConfig("solvent", "categorical", values=("MeCN", "EtOH")),
        ),
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


def mixed_observed_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "mixed_obs_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "x": 0.1,
                "repeats": 1,
                "dose": 0.1,
                "solvent": "MeCN",
                "score": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "mixed_obs_1",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "x": 0.4,
                "repeats": 2,
                "dose": 0.2,
                "solvent": "EtOH",
                "score": 1.5,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "mixed_obs_2",
                "iteration": 2,
                "status": "observed",
                "source": "manual",
                "x": 0.8,
                "repeats": 3,
                "dose": 0.5,
                "solvent": "MeCN",
                "score": 1.2,
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


def output_under_file_parent(tmp_path: Path, filename: str) -> Path:
    parent = tmp_path / "not_a_dir"
    parent.write_text("not a directory", encoding="utf-8")
    return parent / filename


def run_python_module(module: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )


def test_version_outputs_clean_line(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["--version"]) == 0

    captured = capsys.readouterr()
    assert captured.out == "bo-forge 0.4.1\n"
    assert captured.err == ""


@pytest.mark.parametrize("module", ["bo_forge", "bo_forge.cli"])
def test_python_module_entrypoint_version(module: str) -> None:
    completed = run_python_module(module, "--version")

    assert completed.returncode == 0
    assert completed.stdout == "bo-forge 0.4.1\n"
    assert completed.stderr == ""


def test_python_module_entrypoint_validate_success(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config())

    completed = run_python_module(
        "bo_forge",
        "validate",
        *base_args(config_path, log_path),
    )

    assert completed.returncode == 0
    assert completed.stdout == "Campaign log is valid.\n"
    assert completed.stderr == ""


def test_python_module_entrypoint_missing_arguments_returns_argparse_error() -> None:
    completed = run_python_module("bo_forge", "validate")

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "usage:" in completed.stderr
    assert "required" in completed.stderr


def test_doctor_success_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["doctor"]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "BO Forge doctor" in captured.out
    assert "BO Forge version" in captured.out
    assert "Python executable" in captured.out
    assert "Python version" in captured.out
    assert "torch" in captured.out
    assert "botorch" in captured.out
    assert "gpytorch" in captured.out
    assert captured.out.rstrip().endswith("Status: OK")


def test_doctor_import_failure_returns_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_import_module = cli.importlib.import_module

    def fail_torch_import(module_name: str) -> object:
        if module_name == "torch":
            raise ImportError("missing torch")
        return original_import_module(module_name)

    monkeypatch.setattr(cli.importlib, "import_module", fail_torch_import)

    assert run(["doctor"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error: Doctor check failed while importing 'torch': missing torch" in captured.err


def test_init_log_creates_empty_canonical_log(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = tmp_path / "nested" / "campaign.csv"

    assert run(["init-log", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert captured.out == f"Created empty campaign log: {log_path}\n"
    assert captured.err == ""
    cfg = CampaignConfig.from_yaml(config_path)
    df = load_campaign_log(log_path, cfg)
    assert df.empty
    assert list(df.columns) == canonical_columns(cfg)


def test_init_log_refuses_to_overwrite_existing_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = tmp_path / "campaign.csv"
    log_path.write_text("existing", encoding="utf-8")

    assert run(["init-log", *base_args(config_path, log_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error:" in captured.err
    assert "file already exists" in captured.err
    assert log_path.read_text(encoding="utf-8") == "existing"


def test_init_log_does_not_create_file_when_config_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "missing.yaml"
    log_path = tmp_path / "campaign.csv"

    assert run(["init-log", *base_args(config_path, log_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Could not read config file" in captured.err
    assert not log_path.exists()


def test_init_log_write_failure_returns_clear_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = output_under_file_parent(tmp_path, "campaign.csv")

    assert run(["init-log", *base_args(config_path, log_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"Error: Could not write empty campaign log '{log_path}'" in captured.err


def test_init_log_missing_required_arguments_return_argparse_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run(["init-log"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage:" in captured.err
    assert "required" in captured.err


def test_validate_success_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config())

    assert run(["validate", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert captured.out == "Campaign log is valid.\n"
    assert captured.err == ""


def test_mixed_validate_success_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = write_mixed_config(tmp_path / "mixed.yaml")
    cfg = mixed_config()
    log_path = write_log(tmp_path / "mixed.csv", cfg, mixed_observed_log(cfg))

    assert run(["validate", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert captured.out == "Campaign log is valid.\n"
    assert captured.err == ""


def test_config_load_failure_returns_hint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "missing.yaml"
    log_path = tmp_path / "campaign.csv"

    assert run(["validate", *base_args(config_path, log_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error:" in captured.err
    assert "Could not read config file" in captured.err
    assert "Hint: Check the YAML config path and campaign settings." in captured.err


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
    assert (
        "Hint: Check the CSV schema, statuses, objective values, and variable bounds."
        in captured.err
    )


def test_suggest_with_pending_suggestions_returns_hint_without_mutating_csv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config())
    assert run(["suggest", *base_args(config_path, log_path), "--append"]) == 0
    capsys.readouterr()
    before_csv = log_path.read_text(encoding="utf-8")

    assert run(["suggest", *base_args(config_path, log_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error:" in captured.err
    assert "unresolved status='suggested'" in captured.err
    assert (
        "Hint: Resolve pending suggestions or review the campaign state before "
        "requesting new suggestions."
        in captured.err
    )
    assert log_path.read_text(encoding="utf-8") == before_csv


def test_mark_observed_missing_row_returns_hint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config())

    assert (
        run(
            [
                "mark-observed",
                *base_args(config_path, log_path),
                "--row-id",
                "missing",
                "--objective-value",
                "1.0",
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error:" in captured.err
    assert "row_id was not found" in captured.err
    assert (
        "Hint: Check the row_id, pending status, campaign log path, and file permissions."
        in captured.err
    )


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


def test_report_output_write_failure_returns_clear_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg))
    report_path = output_under_file_parent(tmp_path, "latest.txt")

    assert run(["report", *base_args(config_path, log_path), "--output", str(report_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"Error: Could not write campaign report '{report_path}'" in captured.err


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


def test_mixed_suggest_append_writes_valid_mixed_row(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_mixed_config(tmp_path / "mixed.yaml", initial_design_size=4)
    cfg = mixed_config(initial_design_size=4)
    log_path = write_log(tmp_path / "mixed.csv", cfg)

    assert run(["suggest", *base_args(config_path, log_path), "--append"]) == 0

    captured = capsys.readouterr()
    assert "Generated 1 suggestion(s)." in captured.out
    df = load_campaign_log(log_path, cfg)
    assert len(df) == 1
    assert df.loc[0, "status"] == "suggested"
    assert df.loc[0, "solvent"] in {"MeCN", "EtOH"}


def test_suggest_output_write_failure_returns_clear_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config())
    output_path = output_under_file_parent(tmp_path, "suggestions.csv")
    before_csv = log_path.read_text(encoding="utf-8")

    assert run(["suggest", *base_args(config_path, log_path), "--output", str(output_path)]) == 1

    captured = capsys.readouterr()
    assert "Generated 1 suggestion(s)." in captured.out
    assert f"Error: Could not write suggestions CSV '{output_path}'" in captured.err
    assert log_path.read_text(encoding="utf-8") == before_csv


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


@pytest.mark.parametrize("kind", ["progress", "diagnostics"])
def test_plot_output_write_failure_returns_clear_error(
    kind: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg))
    output_path = output_under_file_parent(tmp_path, f"{kind}.png")
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
        == 1
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"Error: Could not write {kind} plot '{output_path}'" in captured.err
    assert log_path.read_text(encoding="utf-8") == before_csv


def test_missing_required_arguments_return_argparse_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run(["validate"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage:" in captured.err
    assert "required" in captured.err


def test_unexpected_errors_are_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_unexpected_error(args: object) -> int:
        raise RuntimeError("unexpected boom")

    monkeypatch.setattr(cli, "_cmd_validate", raise_unexpected_error)

    with pytest.raises(RuntimeError, match="unexpected boom"):
        cli.run(["validate", "--config", "campaign.yaml", "--log", "campaign.csv"])
