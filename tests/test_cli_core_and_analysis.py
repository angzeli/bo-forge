"""CLI core, noisy, contextual, and model-analysis command tests."""

from bo_forge import __version__
from bo_forge._campaign.provenance import manifest_path_for_log
from tests._cli_support import (
    CampaignConfig,
    Path,
    base_args,
    canonical_columns,
    cli,
    config,
    cost_review_config,
    load_campaign_log,
    mixed_config,
    mixed_observed_log,
    observed_log,
    output_under_file_parent,
    pd,
    pytest,
    replicate_config,
    run,
    run_python_module,
    suggestions_module,
    torch,
    write_config,
    write_cost_review_config,
    write_log,
    write_mixed_config,
    write_replicate_config,
)


def test_version_outputs_clean_line(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["--version"]) == 0

    captured = capsys.readouterr()
    assert captured.out == f"bo-forge {__version__}\n"
    assert captured.err == ""

@pytest.mark.parametrize("module", ["bo_forge", "bo_forge.cli"])
def test_python_module_entrypoint_version(module: str) -> None:
    completed = run_python_module(module, "--version")

    assert completed.returncode == 0
    assert completed.stdout == f"bo-forge {__version__}\n"
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
    assert captured.out == (
        f"Created empty campaign log: {log_path}\n"
        f"Created provenance manifest: {manifest_path_for_log(log_path)}\n"
    )
    assert captured.err == ""
    cfg = CampaignConfig.from_yaml(config_path)
    df = load_campaign_log(log_path, cfg)
    assert df.empty
    assert list(df.columns) == canonical_columns(cfg)
    assert manifest_path_for_log(log_path).exists()


def test_provenance_command_reports_managed_and_legacy_campaigns(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    managed_log = tmp_path / "managed.csv"
    legacy_log = write_log(tmp_path / "legacy.csv", config())
    assert run(["init-log", *base_args(config_path, managed_log)]) == 0
    capsys.readouterr()

    assert run(["provenance", *base_args(config_path, managed_log)]) == 0
    managed_output = capsys.readouterr().out
    assert "provenance_status" in managed_output
    assert "managed" in managed_output
    assert "integrity_status" in managed_output
    assert "valid" in managed_output
    assert "campaign_id" in managed_output

    assert run(["provenance", *base_args(config_path, legacy_log)]) == 0
    legacy_output = capsys.readouterr().out
    assert "provenance_status" in legacy_output
    assert "legacy" in legacy_output


def test_provenance_command_rejects_missing_log(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = tmp_path / "missing.csv"

    assert run(["provenance", *base_args(config_path, log_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "provenance status is unknown" in captured.err
    assert ".manifest.json sidecar" in captured.err


def test_provenance_command_reports_managed_mismatch_as_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = tmp_path / "campaign.csv"
    assert run(["init-log", *base_args(config_path, log_path)]) == 0
    capsys.readouterr()
    log_path.write_bytes(log_path.read_bytes() + b"\n")
    before_log = log_path.read_bytes()
    before_manifest = manifest_path_for_log(log_path).read_bytes()

    assert run(["provenance", *base_args(config_path, log_path)]) == 1

    captured = capsys.readouterr()
    assert "integrity_status" in captured.out
    assert "mismatch" in captured.out
    assert "not in a finalized valid state" in captured.err
    assert log_path.read_bytes() == before_log
    assert manifest_path_for_log(log_path).read_bytes() == before_manifest

def test_init_log_creates_cost_review_schema(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_cost_review_config(tmp_path / "campaign.yaml")
    cfg = cost_review_config()
    log_path = tmp_path / "nested" / "campaign.csv"

    assert run(["init-log", *base_args(config_path, log_path)]) == 0

    capsys.readouterr()
    df = load_campaign_log(log_path, cfg)
    assert df.empty
    assert list(df.columns) == canonical_columns(cfg)

def test_init_log_creates_replicate_schema(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_replicate_config(tmp_path / "campaign.yaml")
    cfg = replicate_config()
    log_path = tmp_path / "nested" / "campaign.csv"

    assert run(["init-log", *base_args(config_path, log_path)]) == 0

    capsys.readouterr()
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
    assert "already exists" in captured.err
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
    assert "Error: Could not prepare campaign log directory" in captured.err

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

def test_validate_constrained_log_failure_returns_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/06_mixed_constrained_logei.yaml")
    cfg = CampaignConfig.from_yaml(config_path)
    df = pd.read_csv(
        "examples/06_mixed_constrained_logei_campaign_log.csv",
        keep_default_na=False,
    )
    df.loc[0, "solvent"] = "Water"
    df.loc[0, "base_equivalents"] = 1.0
    df.loc[0, "reaction_time"] = 20
    log_path = write_log(tmp_path / "constrained.csv", cfg, df)

    assert run(["validate", *base_args(config_path, log_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "violates constraint" in captured.err

def test_constrained_suggest_output_is_feasible(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/06_mixed_constrained_logei.yaml")
    log_path = tmp_path / "constrained.csv"
    output_path = tmp_path / "suggestions.csv"
    seed = pd.read_csv(
        "examples/06_mixed_constrained_logei_campaign_log.csv",
        keep_default_na=False,
    )
    seed.to_csv(log_path, index=False)

    assert run(
        [
            "suggest",
            *base_args(config_path, log_path),
            "--output",
            str(output_path),
        ]
    ) == 0

    captured = capsys.readouterr()
    assert "Generated" in captured.out
    suggestions = pd.read_csv(output_path, keep_default_na=False)
    assert not (
        (suggestions["solvent"] == "Water")
        & (suggestions["base_equivalents"].astype(float) >= 0.5)
    ).any()
    assert not (
        (suggestions["solvent"] == "Water")
        & (suggestions["reaction_time"].astype(int) < 35)
    ).any()

def test_qlog_nei_cli_suggest_works_with_accepted_pending_rows(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/18_noisy_pending_qlognei.yaml")
    log_path = tmp_path / "qlog_nei.csv"
    output_path = tmp_path / "qlog_nei_suggestions.csv"
    pd.read_csv(
        "examples/18_noisy_pending_qlognei_campaign_log.csv",
        keep_default_na=False,
    ).to_csv(log_path, index=False)

    assert run(
        [
            "suggest",
            *base_args(config_path, log_path),
            "--batch-size",
            "1",
            "--output",
            str(output_path),
        ]
    ) == 0

    captured = capsys.readouterr()
    assert "Generated 1 suggestion(s)." in captured.out
    suggestions = pd.read_csv(output_path, keep_default_na=False)
    assert suggestions.loc[0, "source"] == "qlog_nei"

def test_qlog_nei_cli_summary_outputs_pending_state_fields(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/18_noisy_pending_qlognei.yaml")
    log_path = tmp_path / "qlog_nei.csv"
    pd.read_csv(
        "examples/18_noisy_pending_qlognei_campaign_log.csv",
        keep_default_na=False,
    ).to_csv(log_path, index=False)

    assert run(["qlog-nei-summary", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert "observed_baseline_rows" in captured.out
    assert "active_pending_rows" in captured.out
    assert "ready_for_qlog_nei" in captured.out

def test_qlog_nei_cli_plot_diagnostics_writes_output(tmp_path: Path) -> None:
    output = tmp_path / "plots" / "qlog_nei_diagnostics.png"

    assert run(
        [
            "plot",
            *base_args(
                Path("configs/18_noisy_pending_qlognei.yaml"),
                Path("examples/18_noisy_pending_qlognei_campaign_log.csv"),
            ),
            "--kind",
            "qlog-nei-diagnostics",
            "--output",
            str(output),
        ]
    ) == 0

    assert output.exists()

def test_qlog_nei_cli_suggest_hint_for_review_pending_rows(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/18_noisy_pending_qlognei.yaml")
    log_path = tmp_path / "qlog_nei.csv"
    df = pd.read_csv(
        "examples/18_noisy_pending_qlognei_campaign_log.csv",
        keep_default_na=False,
    )
    df.loc[df["row_id"] == "nei_pending_0", "review_status"] = "pending"
    df.to_csv(log_path, index=False)
    before = log_path.read_bytes()

    assert run(["suggest", *base_args(config_path, log_path), "--batch-size", "1"]) == 1

    captured = capsys.readouterr()
    assert "review_status='pending'" in captured.err
    assert "accepted suggestions as X_pending" in captured.err
    assert log_path.read_bytes() == before

def test_qlog_nei_cli_suggest_hint_for_pending_initial_design_rows(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = CampaignConfig.from_yaml("configs/18_noisy_pending_qlognei.yaml")
    config_path = Path("configs/18_noisy_pending_qlognei.yaml")
    log_path = tmp_path / "qlog_nei.csv"
    df = pd.read_csv(
        "examples/18_noisy_pending_qlognei_campaign_log.csv",
        keep_default_na=False,
    ).iloc[:3].copy()
    pending_initial = {
        "row_id": "initial_pending",
        "iteration": 1,
        "status": "suggested",
        "source": "sobol",
        "review_status": "accepted",
        "review_note": "",
        "catalyst_loading": 0.58,
        "reaction_temperature": 96.0,
        "activity": "",
        "predicted_mean": "",
        "predicted_std": "",
        "acquisition": "",
    }
    df = pd.concat(
        [df, pd.DataFrame([pending_initial], columns=canonical_columns(cfg))],
        ignore_index=True,
    )
    df.to_csv(log_path, index=False)
    before = log_path.read_bytes()

    assert run(["suggest", *base_args(config_path, log_path), "--batch-size", "1"]) == 1

    captured = capsys.readouterr()
    assert "observe accepted pending initial suggestions" in captured.err
    assert "requires observed initial-design rows" in captured.err
    assert log_path.read_bytes() == before

def test_qlog_nehvi_cli_suggest_works_with_accepted_pending_rows(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/19_multi_objective_qlognehvi.yaml")
    log_path = tmp_path / "qlog_nehvi.csv"
    output_path = tmp_path / "qlog_nehvi_suggestions.csv"
    pd.read_csv(
        "examples/19_multi_objective_qlognehvi_campaign_log.csv",
        keep_default_na=False,
    ).to_csv(log_path, index=False)

    assert run(
        [
            "suggest",
            *base_args(config_path, log_path),
            "--batch-size",
            "1",
            "--output",
            str(output_path),
        ]
    ) == 0

    captured = capsys.readouterr()
    assert "Generated 1 suggestion(s)." in captured.out
    suggestions = pd.read_csv(output_path, keep_default_na=False)
    assert suggestions.loc[0, "source"] == "qlog_nehvi"

def test_qlog_nehvi_cli_suggest_hint_for_review_pending_rows(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/19_multi_objective_qlognehvi.yaml")
    log_path = tmp_path / "qlog_nehvi.csv"
    df = pd.read_csv(
        "examples/19_multi_objective_qlognehvi_campaign_log.csv",
        keep_default_na=False,
    )
    df.loc[df["row_id"] == "accepted_pending_000", "review_status"] = "pending"
    df.to_csv(log_path, index=False)
    before = log_path.read_bytes()

    assert run(["suggest", *base_args(config_path, log_path), "--batch-size", "1"]) == 1

    captured = capsys.readouterr()
    assert "review_status='pending'" in captured.err
    assert "Accepted suggestions are allowed as X_pending" in captured.err
    assert log_path.read_bytes() == before

def test_contextual_suggest_accepts_context_value(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/16_contextual_logei.yaml")
    log_path = tmp_path / "contextual.csv"
    output_path = tmp_path / "contextual_suggestions.csv"
    seed = pd.read_csv(
        "examples/16_contextual_logei_campaign_log.csv",
        keep_default_na=False,
    )
    seed.to_csv(log_path, index=False)

    assert run(
        [
            "suggest",
            *base_args(config_path, log_path),
            "--context",
            "feedstock_acidity=0.25",
            "--output",
            str(output_path),
        ]
    ) == 0

    captured = capsys.readouterr()
    assert "Generated 1 suggestion(s)." in captured.out
    suggestions = pd.read_csv(output_path, keep_default_na=False)
    assert suggestions["feedstock_acidity"].astype(float).tolist() == [pytest.approx(0.25)]

def test_contextual_cost_review_cli_round_trip_with_actual_cost(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/20_contextual_cost_review_logei.yaml")
    cfg = CampaignConfig.from_yaml(config_path)
    log_path = tmp_path / "contextual_cost_review.csv"
    pd.read_csv(
        "examples/20_contextual_cost_review_campaign_log.csv",
        keep_default_na=False,
    ).to_csv(log_path, index=False)

    assert (
        run(
            [
                "suggest",
                *base_args(config_path, log_path),
                "--context",
                "feedstock_acidity=0.5",
                "--append",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert f"Appended suggestions to campaign log: {log_path}" in captured.out
    pending = pd.read_csv(log_path, keep_default_na=False)
    suggested = pending.loc[pending["status"] == "suggested"].iloc[0]
    row_id = str(suggested["row_id"])
    assert suggested["source"] == "cost_log_ei"
    assert suggested["review_status"] == "pending"
    assert float(suggested["feedstock_acidity"]) == pytest.approx(0.5)
    assert float(suggested["cost_estimate"]) > 0

    assert (
        run(
            [
                "review",
                *base_args(config_path, log_path),
                "--row-id",
                row_id,
                "--decision",
                "accept",
                "--note",
                "approved",
            ]
        )
        == 0
    )
    assert (
        run(
            [
                "mark-observed",
                *base_args(config_path, log_path),
                "--row-id",
                row_id,
                "--objective-value",
                "0.84",
                "--actual-cost",
                "4.2",
            ]
        )
        == 0
    )

    df = load_campaign_log(log_path, cfg)
    observed = df.loc[df["row_id"] == row_id].iloc[0]
    assert observed["status"] == "observed"
    assert observed["review_note"] == "approved"
    assert float(observed["yield_score"]) == pytest.approx(0.84)
    assert float(observed["cost_actual"]) == pytest.approx(4.2)

    assert run(["context-summary", *base_args(config_path, log_path)]) == 0
    assert "feedstock_acidity=0.5" in capsys.readouterr().out
    assert run(["cost-summary", *base_args(config_path, log_path)]) == 0
    assert "budget_remaining" in capsys.readouterr().out

    report_path = tmp_path / "contextual_cost_review_report.md"
    context_plot_path = tmp_path / "context_diagnostics.png"
    cost_plot_path = tmp_path / "cost_progress.png"
    assert (
        run(
            [
                "report",
                *base_args(config_path, log_path),
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    assert (
        run(
            [
                "plot",
                *base_args(config_path, log_path),
                "--kind",
                "context-diagnostics",
                "--output",
                str(context_plot_path),
            ]
        )
        == 0
    )
    assert (
        run(
            [
                "plot",
                *base_args(config_path, log_path),
                "--kind",
                "cost-progress",
                "--output",
                str(cost_plot_path),
            ]
        )
        == 0
    )
    assert report_path.exists()
    assert context_plot_path.exists()
    assert cost_plot_path.exists()

def test_contextual_replicate_cli_round_trip_with_actual_cost(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = Path("configs/21_contextual_replicate_logei.yaml")
    cfg = CampaignConfig.from_yaml(config_path)
    log_path = tmp_path / "contextual_replicates.csv"
    pd.read_csv(
        "examples/21_contextual_replicate_campaign_log.csv",
        keep_default_na=False,
    ).to_csv(log_path, index=False)

    class FakePosterior:
        mean = torch.tensor([[2.0], [1.0], [10.0], [0.0]], dtype=torch.double)
        variance = torch.full((4, 1), 0.04, dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    assert (
        run(
            [
                "suggest",
                *base_args(config_path, log_path),
                "--context",
                "feedstock_acidity=0.25",
                "--batch-size",
                "1",
                "--append",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "No active repeat was selected" not in captured.err
    pending = load_campaign_log(log_path, cfg)
    suggestion = pending.loc[pending["status"] == "suggested"].iloc[0]
    row_id = str(suggestion["row_id"])
    assert suggestion["replicate_group"] == "group_acid25_best"
    assert int(suggestion["replicate_index"]) == 2
    assert float(suggestion["feedstock_acidity"]) == pytest.approx(0.25)
    assert float(suggestion["cost_estimate"]) == pytest.approx(3.9)

    assert run(
        [
            "review",
            *base_args(config_path, log_path),
            "--row-id",
            row_id,
            "--decision",
            "accept",
            "--note",
            "approved",
        ]
    ) == 0
    assert run(
        [
            "mark-observed",
            *base_args(config_path, log_path),
            "--row-id",
            row_id,
            "--objective-value",
            "0.91",
            "--actual-cost",
            "4.0",
        ]
    ) == 0
    assert run(["context-summary", *base_args(config_path, log_path)]) == 0
    assert run(["replicate-summary", *base_args(config_path, log_path)]) == 0
    assert run(["cost-summary", *base_args(config_path, log_path)]) == 0

    report_path = tmp_path / "contextual_replicate_report.md"
    context_plot_path = tmp_path / "context_diagnostics.png"
    replicate_plot_path = tmp_path / "replicate_diagnostics.png"
    assert run(
        ["report", *base_args(config_path, log_path), "--output", str(report_path)]
    ) == 0
    assert run(
        [
            "plot",
            *base_args(config_path, log_path),
            "--kind",
            "context-diagnostics",
            "--output",
            str(context_plot_path),
        ]
    ) == 0
    assert run(
        [
            "plot",
            *base_args(config_path, log_path),
            "--kind",
            "replicates",
            "--output",
            str(replicate_plot_path),
        ]
    ) == 0
    assert report_path.exists()
    assert context_plot_path.exists()
    assert replicate_plot_path.exists()

def test_contextual_replicate_cli_exploration_fallback_is_explained(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/21_contextual_replicate_logei.yaml")
    log_path = tmp_path / "contextual_replicates.csv"
    pd.read_csv(
        "examples/21_contextual_replicate_campaign_log.csv",
        keep_default_na=False,
    ).to_csv(log_path, index=False)

    assert run(
        [
            "suggest",
            *base_args(config_path, log_path),
            "--context",
            "feedstock_acidity=0.5",
            "--batch-size",
            "1",
        ]
    ) == 0

    captured = capsys.readouterr()
    assert "No active repeat was selected for the requested context" in captured.err
    assert "Generated 1 suggestion(s)." in captured.out

def test_contextual_suggest_missing_context_does_not_append(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "contextual.yaml"
    config_path.write_text(
        """
campaign_name: contextual_cli
objective: {name: score, direction: maximize}
variables:
  - {name: x, type: continuous, lower: 0, upper: 1}
  - {name: feedstock_acidity, type: continuous, lower: 0, upper: 1}
context:
  variables: [feedstock_acidity]
bo:
  batch_size: 1
  initial_design_size: 2
  acquisition: log_ei
""",
        encoding="utf-8",
    )
    cfg = CampaignConfig.from_yaml(config_path)
    log_path = write_log(tmp_path / "contextual.csv", cfg)
    before = log_path.read_bytes()

    assert run(["suggest", *base_args(config_path, log_path), "--append"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "missing=['feedstock_acidity']" in captured.err
    assert "Hint: Use --context NAME=VALUE" in captured.err
    assert log_path.read_bytes() == before

def test_contextual_suggest_rejects_malformed_context(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/16_contextual_logei.yaml")
    log_path = tmp_path / "contextual.csv"
    pd.read_csv(
        "examples/16_contextual_logei_campaign_log.csv",
        keep_default_na=False,
    ).to_csv(log_path, index=False)

    assert run(["suggest", *base_args(config_path, log_path), "--context", "bad"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Malformed --context value" in captured.err

def test_contextual_cli_context_summary_outputs_table(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/16_contextual_logei.yaml")
    log_path = tmp_path / "contextual.csv"
    pd.read_csv(
        "examples/16_contextual_logei_campaign_log.csv",
        keep_default_na=False,
    ).to_csv(log_path, index=False)

    assert run(["context-summary", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert "context_key" in captured.out
    assert "feedstock_acidity=0.3" in captured.out
    assert "ctx_seed_1" in captured.out

def test_contextual_cli_context_summary_handles_pending_only_log(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = CampaignConfig.from_yaml("configs/16_contextual_logei.yaml")
    pending = {
        "row_id": "pending_0",
        "iteration": 0,
        "status": "suggested",
        "source": "sobol",
        "catalyst_loading": 0.5,
        "reaction_temperature": 80,
        "solvent": "MeCN",
        "feedstock_acidity": 0.25,
        "yield_score": "",
        "predicted_mean": "",
        "predicted_std": "",
        "acquisition": "",
    }
    log_path = tmp_path / "contextual_pending.csv"
    pd.DataFrame(
        [[pending[column] for column in canonical_columns(cfg)]],
        columns=canonical_columns(cfg),
    ).to_csv(log_path, index=False)

    assert (
        run(
            [
                "context-summary",
                *base_args(Path("configs/16_contextual_logei.yaml"), log_path),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert "context_key" in captured.out
    assert "feedstock_acidity=0.25" in captured.out
    assert "pending_suggestions" in captured.out

def test_contextual_cli_context_summary_rejects_non_context_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config(), observed_log(config()))

    assert run(["context-summary", *base_args(config_path, log_path)]) == 1

    captured = capsys.readouterr()
    assert "context-summary requires a contextual config" in captured.err

def test_cli_model_summary_outputs_table(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/17_model_profile_logei.yaml")
    log_path = tmp_path / "model_profile.csv"
    pd.read_csv(
        "examples/17_model_profile_campaign_log.csv",
        keep_default_na=False,
    ).to_csv(log_path, index=False)

    assert run(["model-summary", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert "model_profile" in captured.out
    assert "smooth" in captured.out
    assert "covariance_profile" in captured.out
    assert "RBF/ARD" in captured.out
    assert "last_fit_status" in captured.out
    assert "not_recorded" in captured.out

def test_cli_model_compare_outputs_all_profiles(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/17_model_profile_logei.yaml")
    log_path = tmp_path / "model_profile.csv"
    pd.read_csv(
        "examples/17_model_profile_campaign_log.csv",
        keep_default_na=False,
    ).to_csv(log_path, index=False)

    assert run(["model-compare", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert "model_profile" in captured.out
    assert "fit_message" in captured.out
    assert "rmse_model_space" in captured.out
    assert "mean_predicted_std" in captured.out
    assert "default" in captured.out
    assert "smooth" in captured.out
    assert "rough" in captured.out
    assert "robust" in captured.out

def test_cli_model_compare_preserves_profile_arg_order(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/17_model_profile_logei.yaml")
    log_path = tmp_path / "model_profile.csv"
    pd.read_csv(
        "examples/17_model_profile_campaign_log.csv",
        keep_default_na=False,
    ).to_csv(log_path, index=False)

    assert (
        run(
            [
                "model-compare",
                *base_args(config_path, log_path),
                "--profile",
                "smooth",
                "--profile",
                "default",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out.find("smooth") < captured.out.find("default")
    assert "robust" not in captured.out

def test_cli_model_compare_rejects_duplicate_profile_args(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/17_model_profile_logei.yaml")
    log_path = tmp_path / "model_profile.csv"
    pd.read_csv(
        "examples/17_model_profile_campaign_log.csv",
        keep_default_na=False,
    ).to_csv(log_path, index=False)

    assert (
        run(
            [
                "model-compare",
                *base_args(config_path, log_path),
                "--profile",
                "smooth",
                "--profile",
                "smooth",
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert "Duplicate model profile requested: smooth" in captured.err

def test_cli_model_compare_rejects_unknown_profile_arg(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/17_model_profile_logei.yaml")
    log_path = tmp_path / "model_profile.csv"
    pd.read_csv(
        "examples/17_model_profile_campaign_log.csv",
        keep_default_na=False,
    ).to_csv(log_path, index=False)

    assert (
        run(
            [
                "model-compare",
                *base_args(config_path, log_path),
                "--profile",
                "experimental",
            ]
        )
        == 2
    )

    captured = capsys.readouterr()
    assert "invalid choice: 'experimental'" in captured.err

def test_cli_model_compare_rejects_unsupported_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_path = tmp_path / "multi_fidelity.csv"
    pd.read_csv(
        "examples/15_multi_fidelity_qmfkg_campaign_log.csv",
        keep_default_na=False,
    ).to_csv(log_path, index=False)

    assert (
        run(
            [
                "model-compare",
                *base_args(Path("configs/15_multi_fidelity_qmfkg.yaml"), log_path),
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert "does not support multi-fidelity configs" in captured.err

def test_contextual_cli_plot_context_diagnostics_handles_pending_only_log(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = CampaignConfig.from_yaml("configs/16_contextual_logei.yaml")
    pending = {
        "row_id": "pending_0",
        "iteration": 0,
        "status": "suggested",
        "source": "sobol",
        "catalyst_loading": 0.5,
        "reaction_temperature": 80,
        "solvent": "MeCN",
        "feedstock_acidity": 0.25,
        "yield_score": "",
        "predicted_mean": "",
        "predicted_std": "",
        "acquisition": "",
    }
    log_path = tmp_path / "contextual_pending.csv"
    pd.DataFrame(
        [[pending[column] for column in canonical_columns(cfg)]],
        columns=canonical_columns(cfg),
    ).to_csv(log_path, index=False)
    output_path = tmp_path / "reports" / "context_pending.png"

    assert (
        run(
            [
                "plot",
                *base_args(Path("configs/16_contextual_logei.yaml"), log_path),
                "--kind",
                "context-diagnostics",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert f"Wrote context-diagnostics plot: {output_path}" in captured.out
    assert output_path.exists()

def test_contextual_cli_plot_context_diagnostics_writes_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/16_contextual_logei.yaml")
    log_path = tmp_path / "contextual.csv"
    pd.read_csv(
        "examples/16_contextual_logei_campaign_log.csv",
        keep_default_na=False,
    ).to_csv(log_path, index=False)
    output_path = tmp_path / "reports" / "context.png"

    assert (
        run(
            [
                "plot",
                *base_args(config_path, log_path),
                "--kind",
                "context-diagnostics",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert f"Wrote context-diagnostics plot: {output_path}" in captured.out
    assert output_path.exists()

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
