"""CLI campaign mutation, report, plot, and structured workflow tests."""

from tests._cli_support import (
    CampaignConfig,
    LogBusyError,
    LogConflictError,
    Path,
    base_args,
    canonical_columns,
    cli,
    config,
    cost_review_config,
    cost_review_log,
    fidelity_config,
    fidelity_observed_log,
    load_campaign_log,
    mixed_config,
    multi_objective_config,
    multi_objective_log,
    observed_log,
    output_under_file_parent,
    pd,
    pytest,
    replicate_config,
    replicate_log,
    run,
    suggestions_module,
    write_config,
    write_cost_review_config,
    write_fidelity_config,
    write_log,
    write_mixed_config,
    write_multi_objective_config,
    write_multi_objective_cost_config,
    write_replicate_config,
)


def test_validate_rejects_single_objective_qlog_nehvi_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("log_ei", "qlog_nehvi"),
        encoding="utf-8",
    )
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg))

    assert run(["validate", *base_args(config_path, log_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error:" in captured.err
    assert "qlog_nehvi" in captured.err
    assert "only supported for coupled multi-objective campaigns" in captured.err

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

def test_structured_suggest_requires_stage_without_mutating_csv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/13_structured_campaign_core.yaml")
    log_path = tmp_path / "structured.csv"
    cfg = CampaignConfig.from_yaml(config_path)
    write_log(log_path, cfg)
    before = log_path.read_bytes()

    assert run(["suggest", *base_args(config_path, log_path), "--append"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Structured campaign suggestions require an explicit stage" in captured.err
    assert "Hint: Use --stage with one configured structured stage name" in captured.err
    assert log_path.read_bytes() == before

def test_structured_suggest_dry_run_accepts_stage_without_mutating_csv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/13_structured_campaign_core.yaml")
    log_path = tmp_path / "structured.csv"
    cfg = CampaignConfig.from_yaml(config_path)
    write_log(log_path, cfg)
    before = log_path.read_bytes()

    assert run(["suggest", *base_args(config_path, log_path), "--stage", "screen"]) == 0

    captured = capsys.readouterr()
    assert "Generated 1 suggestion(s)." in captured.out
    assert "screen" in captured.out
    assert log_path.read_bytes() == before

def test_structured_documented_init_log_then_suggest_flow_succeeds(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/13_structured_campaign_core.yaml")
    log_path = tmp_path / "structured_working.csv"

    assert run(["init-log", *base_args(config_path, log_path)]) == 0
    assert run(["suggest", *base_args(config_path, log_path), "--stage", "screen"]) == 0

    captured = capsys.readouterr()
    assert "Created empty campaign log" in captured.out
    assert "Generated 1 suggestion(s)." in captured.out
    assert "screen" in captured.out

def test_structured_suggest_append_writes_stage_aware_rows(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/13_structured_campaign_core.yaml")
    log_path = tmp_path / "structured.csv"
    cfg = CampaignConfig.from_yaml(config_path)
    write_log(log_path, cfg)

    assert (
        run(
            [
                "suggest",
                *base_args(config_path, log_path),
                "--stage",
                "screen",
                "--append",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert "Appended suggestions to campaign log" in captured.out
    df = load_campaign_log(log_path, cfg)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["stage"] == "screen"
    assert pd.notna(row["precursor_ratio"])
    assert pd.notna(row["electrolyte"])
    assert row["annealing_temperature"] == ""

def test_structured_suggest_unknown_stage_fails_without_mutating_csv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/13_structured_campaign_core.yaml")
    log_path = tmp_path / "structured.csv"
    cfg = CampaignConfig.from_yaml(config_path)
    write_log(log_path, cfg)
    before = log_path.read_bytes()

    assert (
        run(
            [
                "suggest",
                *base_args(config_path, log_path),
                "--stage",
                "missing",
                "--append",
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Unknown structured campaign stage 'missing'" in captured.err
    assert log_path.read_bytes() == before

def test_structured_suggest_invalid_stage_format_returns_stage_hint_without_mutating(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/13_structured_campaign_core.yaml")
    log_path = tmp_path / "structured.csv"
    cfg = CampaignConfig.from_yaml(config_path)
    write_log(log_path, cfg)
    before = log_path.read_bytes()

    assert (
        run(
            [
                "suggest",
                *base_args(config_path, log_path),
                "--stage",
                " screen",
                "--append",
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Invalid structured campaign stage" in captured.err
    assert "Hint: Use --stage with one configured structured stage name" in captured.err
    assert log_path.read_bytes() == before

def test_stage_summary_cli_prints_structured_stage_table(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/13_structured_campaign_core.yaml")
    log_path = tmp_path / "structured.csv"
    log_path.write_bytes(
        Path("examples/13_structured_campaign_core_campaign_log.csv").read_bytes()
    )

    assert run(["stage-summary", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert "stage" in captured.out
    assert "screen" in captured.out
    assert "refine" in captured.out
    assert "active_variables" in captured.out
    assert "No observed rows for stage." not in captured.err

def test_stage_summary_cli_rejects_non_structured_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg))

    assert run(["stage-summary", *base_args(config_path, log_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "stage-summary requires a structured campaign config" in captured.err

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

def test_log_coordination_errors_have_specific_cli_hints() -> None:
    assert cli._hint_for_error(LogConflictError("stale")) == (
        "Hint: Reload the campaign, inspect the latest log, and retry the mutation."
    )
    assert cli._hint_for_error(LogBusyError("busy")) == (
        "Hint: Another local writer is active; wait briefly and retry."
    )

@pytest.mark.parametrize(
    "objective_args, expected_error",
    [
        (["--objective", "yield_score"], "Malformed --objective value"),
        (
            [
                "--objective",
                "yield_score=60",
                "--objective",
                "yield_score=61",
                "--objective",
                "waste_score=12",
            ],
            "Duplicate --objective value",
        ),
        (["--objective", "yield_score=60"], "missing=['waste_score']"),
        (
            [
                "--objective",
                "yield_score=60",
                "--objective",
                "waste_score=12",
                "--objective",
                "unknown=1",
            ],
            "extra=['unknown']",
        ),
        (
            ["--objective", "yield_score=bad", "--objective", "waste_score=12"],
            "must be numeric",
        ),
        (
            [
                "--objective-value",
                "1.0",
                "--objective",
                "yield_score=60",
                "--objective",
                "waste_score=12",
            ],
            "Pass either --objective-value or --objective",
        ),
        (
            [
                "--objective",
                "yield_score=60",
                "--objective",
                "waste_score=12",
                "--actual-cost",
                "3.2",
            ],
            "--actual-cost requires a config with a cost section",
        ),
    ],
)
def test_multi_objective_cli_mark_observed_failures_are_byte_atomic(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    objective_args: list[str],
    expected_error: str,
) -> None:
    config_path = write_multi_objective_config(tmp_path / "multi.yaml")
    cfg = multi_objective_config()
    log_path = write_log(tmp_path / "multi.csv", cfg, multi_objective_log(cfg))
    before = log_path.read_bytes()

    assert (
        run(
            [
                "mark-observed",
                *base_args(config_path, log_path),
                "--row-id",
                "suggested_0",
                *objective_args,
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error:" in captured.err
    assert expected_error in captured.err
    assert log_path.read_bytes() == before

def test_multi_objective_cli_mark_observed_accepts_actual_cost_for_cost_config(
    tmp_path: Path,
) -> None:
    config_path = write_multi_objective_cost_config(tmp_path / "multi_cost.yaml")
    cfg = multi_objective_config(cost=True)
    log_path = write_log(tmp_path / "multi_cost.csv", cfg, multi_objective_log(cfg))

    assert (
        run(
            [
                "mark-observed",
                *base_args(config_path, log_path),
                "--row-id",
                "suggested_0",
                "--objective",
                "yield_score=70",
                "--objective",
                "waste_score=14",
                "--actual-cost",
                "2.5",
            ]
        )
        == 0
    )

    df = load_campaign_log(log_path, cfg)
    row = df.loc[df["row_id"] == "suggested_0"].iloc[0]
    assert row["status"] == "observed"
    assert float(row["yield_score"]) == pytest.approx(70.0)
    assert float(row["waste_score"]) == pytest.approx(14.0)
    assert float(row["cost_actual"]) == pytest.approx(2.5)

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
    assert "Best Raw Observation" in report_out

def test_cost_summary_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = write_cost_review_config(tmp_path / "campaign.yaml")
    cfg = cost_review_config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, cost_review_log(cfg))

    assert run(["cost-summary", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert "total_observed_cost" in captured.out
    assert "accepted_pending_cost" in captured.out
    assert "budget_remaining" in captured.out

def test_replicate_summary_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = write_replicate_config(tmp_path / "campaign.yaml")
    cfg = replicate_config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, replicate_log(cfg))

    assert run(["replicate-summary", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert "replicate_group" in captured.out
    assert "objective_mean" in captured.out
    assert "group_0" in captured.out

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

def test_multi_fidelity_cli_suggest_append_writes_qmfkg_row(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_fidelity_config(tmp_path / "fidelity.yaml")
    cfg = fidelity_config()
    log_path = write_log(tmp_path / "fidelity.csv", cfg, fidelity_observed_log(cfg))

    assert run(["suggest", *base_args(config_path, log_path), "--append"]) == 0

    captured = capsys.readouterr()
    assert "Generated 1 suggestion(s)." in captured.out
    assert f"Appended suggestions to campaign log: {log_path}" in captured.out
    df = load_campaign_log(log_path, cfg)
    row = df.iloc[-1]
    assert row["status"] == "suggested"
    assert row["source"] == "qmf_kg"
    assert float(row["x"]) >= 0.0
    assert float(row["x"]) <= 1.0
    assert float(row["fidelity"]) >= 0.2
    assert float(row["fidelity"]) <= 1.0
    assert row["activity"] == ""
    assert list(df.columns) == canonical_columns(cfg)

def test_multi_fidelity_cli_fidelity_summary_outputs_table(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_fidelity_config(tmp_path / "fidelity.yaml")
    cfg = fidelity_config()
    log_path = write_log(tmp_path / "fidelity.csv", cfg, fidelity_observed_log(cfg))

    assert run(["fidelity-summary", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert "fidelity_variable" in captured.out
    assert "target_fidelity_observed_rows" in captured.out
    assert "best_target_fidelity_row_id" in captured.out
    assert "mf_obs_3" in captured.out

def test_multi_fidelity_cli_fidelity_coverage_outputs_table(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_fidelity_config(tmp_path / "fidelity.yaml")
    cfg = fidelity_config()
    log_path = write_log(tmp_path / "fidelity.csv", cfg, fidelity_observed_log(cfg))

    assert run(["fidelity-coverage", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert "modeled_evaluation_cost" in captured.out
    assert "active_suggestions" in captured.out
    assert "objective_best" in captured.out
    assert "None" not in captured.out

def test_multi_fidelity_cli_translates_qmfkg_timeout_without_mutation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_fidelity_config(
        tmp_path / "fidelity.yaml",
        timeout_seconds=1.0,
    )
    cfg = CampaignConfig.from_yaml(config_path)
    log_path = write_log(tmp_path / "fidelity.csv", cfg, fidelity_observed_log(cfg))
    before = log_path.read_bytes()
    times = iter([10.0, 11.0])
    monkeypatch.setattr(
        suggestions_module,
        "fit_multi_fidelity_gp_model",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(suggestions_module.time, "monotonic", lambda: next(times))

    assert run(["suggest", *base_args(config_path, log_path)]) == 1

    captured = capsys.readouterr()
    assert "qMFKG acquisition optimization timed out" in captured.err
    assert "Increase or remove fidelity.optimizer_timeout_seconds" in captured.err
    assert "Resolve pending suggestions" not in captured.err
    assert log_path.read_bytes() == before

def test_multi_fidelity_cli_fidelity_summary_counts_review_blocking_qmfkg(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_fidelity_config(tmp_path / "fidelity.yaml", review=True)
    cfg = fidelity_config(review=True)
    rows = [
        {
            "row_id": "target_0",
            "iteration": 0,
            "status": "observed",
            "source": "manual",
            "review_status": "accepted",
            "review_note": "",
            "x": 0.5,
            "fidelity": 1.0,
            "activity": 1.2,
            "predicted_mean": "",
            "predicted_std": "",
            "acquisition": "",
        },
        {
            "row_id": "pending_qmfkg",
            "iteration": 1,
            "status": "suggested",
            "source": "qmf_kg",
            "review_status": "pending",
            "review_note": "",
            "x": 0.2,
            "fidelity": 0.8,
            "activity": "",
            "predicted_mean": 1.1,
            "predicted_std": 0.1,
            "acquisition": 0.2,
        },
        {
            "row_id": "accepted_qmfkg",
            "iteration": 2,
            "status": "suggested",
            "source": "qmf_kg",
            "review_status": "accepted",
            "review_note": "",
            "x": 0.3,
            "fidelity": 0.8,
            "activity": "",
            "predicted_mean": 1.2,
            "predicted_std": 0.1,
            "acquisition": 0.3,
        },
        {
            "row_id": "rejected_qmfkg",
            "iteration": 3,
            "status": "suggested",
            "source": "qmf_kg",
            "review_status": "rejected",
            "review_note": "",
            "x": 0.4,
            "fidelity": 0.8,
            "activity": "",
            "predicted_mean": 1.3,
            "predicted_std": 0.1,
            "acquisition": 0.4,
        },
        {
            "row_id": "deferred_qmfkg",
            "iteration": 4,
            "status": "suggested",
            "source": "qmf_kg",
            "review_status": "deferred",
            "review_note": "",
            "x": 0.6,
            "fidelity": 0.8,
            "activity": "",
            "predicted_mean": 1.4,
            "predicted_std": 0.1,
            "acquisition": 0.5,
        },
    ]
    log_path = write_log(
        tmp_path / "fidelity.csv",
        cfg,
        pd.DataFrame(rows, columns=canonical_columns(cfg)),
    )

    assert run(["fidelity-summary", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert "pending_qmfkg_suggestions" in captured.out
    pending_line = next(
        line for line in captured.out.splitlines() if "pending_qmfkg_suggestions" in line
    )
    assert pending_line.split()[-1] == "2"

def test_multi_fidelity_cli_fidelity_summary_rejects_non_fidelity_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config(), observed_log(config()))

    assert run(["fidelity-summary", *base_args(config_path, log_path)]) == 1

    captured = capsys.readouterr()
    assert "fidelity-summary requires a multi-fidelity config" in captured.err

    assert run(["fidelity-coverage", *base_args(config_path, log_path)]) == 1
    captured = capsys.readouterr()
    assert "fidelity-coverage requires a multi-fidelity config" in captured.err

    output_path = tmp_path / "fidelity_progress.png"
    assert run(
        [
            "plot",
            *base_args(config_path, log_path),
            "--kind",
            "fidelity-progress",
            "--output",
            str(output_path),
        ]
    ) == 1
    captured = capsys.readouterr()
    assert "plot --kind fidelity-progress requires a multi-fidelity config" in captured.err
    assert not output_path.exists()

def test_multi_fidelity_cli_plot_fidelity_diagnostics_writes_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_fidelity_config(tmp_path / "fidelity.yaml")
    cfg = fidelity_config()
    log_path = write_log(tmp_path / "fidelity.csv", cfg, fidelity_observed_log(cfg))
    output_path = tmp_path / "reports" / "fidelity.png"

    assert (
        run(
            [
                "plot",
                *base_args(config_path, log_path),
                "--kind",
                "fidelity-diagnostics",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert f"Wrote fidelity-diagnostics plot: {output_path}" in captured.out
    assert output_path.exists()

def test_multi_fidelity_cli_plot_fidelity_progress_writes_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_fidelity_config(tmp_path / "fidelity.yaml")
    cfg = fidelity_config()
    log_path = write_log(tmp_path / "fidelity.csv", cfg, fidelity_observed_log(cfg))
    output_path = tmp_path / "reports" / "fidelity_progress.png"

    assert run(
        [
            "plot",
            *base_args(config_path, log_path),
            "--kind",
            "fidelity-progress",
            "--output",
            str(output_path),
        ]
    ) == 0

    captured = capsys.readouterr()
    assert f"Wrote fidelity-progress plot: {output_path}" in captured.out
    assert output_path.exists()

def test_multi_fidelity_cli_rejects_batch_size_above_four_without_mutation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_fidelity_config(tmp_path / "fidelity.yaml")
    cfg = fidelity_config()
    log_path = write_log(tmp_path / "fidelity.csv", cfg, fidelity_observed_log(cfg))
    before = log_path.read_bytes()

    assert run(["suggest", *base_args(config_path, log_path), "--batch-size", "5", "--append"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "qMFKG supports batch_size from 1 through 4" in captured.err
    assert "Hint: Use --batch-size 1, 2, 3, or 4" in captured.err
    assert log_path.read_bytes() == before

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

def test_cost_review_suggest_append_writes_cost_and_review_columns(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_cost_review_config(tmp_path / "campaign.yaml", initial_design_size=3)
    cfg = cost_review_config(initial_design_size=3)
    log_path = write_log(tmp_path / "campaign.csv", cfg)

    assert run(["suggest", *base_args(config_path, log_path), "--append"]) == 0

    captured = capsys.readouterr()
    assert "Generated 1 suggestion(s)." in captured.out
    df = load_campaign_log(log_path, cfg)
    assert df.loc[0, "review_status"] == "pending"
    assert float(df.loc[0, "cost_estimate"]) > 0
    assert df.loc[0, "utility"] == ""

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

def test_review_and_mark_observed_with_actual_cost(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_cost_review_config(tmp_path / "campaign.yaml")
    cfg = cost_review_config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, cost_review_log(cfg))

    assert (
        run(
            [
                "review",
                *base_args(config_path, log_path),
                "--row-id",
                "suggested_0",
                "--decision",
                "accept",
                "--note",
                " approved ",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out == (
        f"Reviewed row suggested_0 as accept in campaign log: {log_path}\n"
    )

    assert (
        run(
            [
                "mark-observed",
                *base_args(config_path, log_path),
                "--row-id",
                "suggested_0",
                "--objective-value",
                "1.8",
                "--actual-cost",
                "1.7",
            ]
        )
        == 0
    )

    df = load_campaign_log(log_path, cfg)
    row = df.loc[df["row_id"] == "suggested_0"].iloc[0]
    assert row["status"] == "observed"
    assert row["review_note"] == "approved"
    assert float(row["score"]) == pytest.approx(1.8)
    assert float(row["cost_actual"]) == pytest.approx(1.7)

@pytest.mark.parametrize(
    "kind", ["progress", "diagnostics", "model-diagnostics", "model-comparison"]
)
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

def test_plot_cost_progress_writes_nested_output_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_cost_review_config(tmp_path / "campaign.yaml")
    cfg = cost_review_config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, cost_review_log(cfg))
    output_path = tmp_path / "figures" / "cost-progress.png"
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
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out == f"Wrote cost-progress plot: {output_path}\n"
    assert output_path.exists()
    assert log_path.read_text(encoding="utf-8") == before_csv

def test_plot_replicates_writes_nested_output_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_replicate_config(tmp_path / "campaign.yaml")
    cfg = replicate_config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, replicate_log(cfg))
    output_path = tmp_path / "figures" / "replicates.png"
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
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out == f"Wrote replicates plot: {output_path}\n"
    assert output_path.exists()
    assert log_path.read_text(encoding="utf-8") == before_csv

def test_plot_stage_diagnostics_writes_nested_output_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/13_structured_campaign_core.yaml")
    log_path = tmp_path / "structured.csv"
    log_path.write_bytes(
        Path("examples/13_structured_campaign_core_campaign_log.csv").read_bytes()
    )
    output_path = tmp_path / "figures" / "stage-diagnostics.png"
    before_csv = log_path.read_text(encoding="utf-8")

    assert (
        run(
            [
                "plot",
                *base_args(config_path, log_path),
                "--kind",
                "stage-diagnostics",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out == f"Wrote stage-diagnostics plot: {output_path}\n"
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
