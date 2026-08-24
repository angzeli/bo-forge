"""CLI error-translation and argument-boundary tests."""

from tests._cli_support import (
    Path,
    base_args,
    cli,
    cost_review_config,
    cost_review_log,
    output_under_file_parent,
    pytest,
    replicate_config,
    replicate_log,
    run,
    write_cost_review_config,
    write_log,
    write_replicate_config,
)


def test_plot_cost_progress_output_write_failure_returns_clear_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_cost_review_config(tmp_path / "campaign.yaml")
    cfg = cost_review_config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, cost_review_log(cfg))
    output_path = output_under_file_parent(tmp_path, "cost-progress.png")
    before_csv = log_path.read_text(encoding="utf-8")

    assert (
        run(
            [
                "plot",
                *base_args(config_path, log_path),
                "--kind",
                "cost-progress",
                "--output",
                str(output_path),
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"Error: Could not write cost-progress plot '{output_path}'" in captured.err
    assert log_path.read_text(encoding="utf-8") == before_csv

def test_plot_replicates_output_write_failure_returns_clear_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_replicate_config(tmp_path / "campaign.yaml")
    cfg = replicate_config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, replicate_log(cfg))
    output_path = output_under_file_parent(tmp_path, "replicates.png")
    before_csv = log_path.read_text(encoding="utf-8")

    assert (
        run(
            [
                "plot",
                *base_args(config_path, log_path),
                "--kind",
                "replicates",
                "--output",
                str(output_path),
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"Error: Could not write replicates plot '{output_path}'" in captured.err
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
