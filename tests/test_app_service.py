import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from matplotlib import pyplot as plt

import bo_forge.suggestions as suggestions_module
from bo_forge.config import CampaignConfig
from bo_forge.errors import LogWriteError
from bo_forge.session import CampaignSession
from bo_forge.transforms import values_to_unit_cube
from bo_forge.validation import canonical_columns
from bo_forge_app.service import CampaignAppService, CampaignViewData
from bo_forge_app.streamlit_helpers import make_staged_suggestion_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def copy_example_log(tmp_path: Path, name: str) -> Path:
    source = PROJECT_ROOT / "examples" / name
    destination = tmp_path / name
    shutil.copyfile(source, destination)
    return destination


def copy_example_config(tmp_path: Path, name: str) -> Path:
    source = PROJECT_ROOT / "configs" / name
    destination = tmp_path / name
    shutil.copyfile(source, destination)
    return destination


def test_app_service_imports_without_streamlit() -> None:
    script = """
import builtins
real_import = builtins.__import__
def block_streamlit(name, *args, **kwargs):
    if name == "streamlit" or name.startswith("streamlit."):
        raise ModuleNotFoundError("blocked streamlit")
    return real_import(name, *args, **kwargs)
builtins.__import__ = block_streamlit
import bo_forge_app.service
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert completed.stdout == "ok\n"


@pytest.mark.parametrize(
    ("config_name", "log_name", "panel", "expected_tables"),
    [
        (
            "01_simple_2d_maximise_logei.yaml",
            "01_simple_2d_maximise_logei_campaign_log.csv",
            "Overview",
            ["summary", "next_action", "model_summary", "observed", "pending"],
        ),
        (
            "10_multi_objective_mixed_constrained_qlogehvi.yaml",
            "10_multi_objective_mixed_constrained_campaign_log.csv",
            "Data",
            [
                "summary",
                "next_action",
                "model_summary",
                "observed",
                "pending",
                "pareto_summary",
                "pareto_front",
            ],
        ),
        (
            "07_cost_aware_human_review_logei.yaml",
            "07_cost_aware_human_review_campaign_log.csv",
            "Resolve",
            ["pending", "observable", "review_queue"],
        ),
        (
            "08_replicate_aware_logei.yaml",
            "08_replicate_aware_campaign_log.csv",
            "Overview",
            ["summary", "next_action", "model_summary", "observed", "pending", "replicate_summary"],
        ),
        (
            "12_cost_aware_multi_objective_qlogehvi.yaml",
            "12_cost_aware_multi_objective_campaign_log.csv",
            "Overview",
            [
                "summary",
                "next_action",
                "model_summary",
                "observed",
                "pending",
                "pareto_summary",
                "cost_summary",
            ],
        ),
        (
            "13_structured_campaign_core.yaml",
            "13_structured_campaign_core_campaign_log.csv",
            "Data",
            ["summary", "next_action", "model_summary", "observed", "pending", "stage_summary"],
        ),
        (
            "15_multi_fidelity_qmfkg.yaml",
            "15_multi_fidelity_qmfkg_campaign_log.csv",
            "Overview",
            ["summary", "next_action", "model_summary", "observed", "pending", "fidelity_summary"],
        ),
        (
            "16_contextual_logei.yaml",
            "16_contextual_logei_campaign_log.csv",
            "Overview",
            ["summary", "next_action", "model_summary", "observed", "pending", "context_summary"],
        ),
    ],
)
def test_app_service_loads_validates_and_collects_view_data(
    tmp_path: Path,
    config_name: str,
    log_name: str,
    panel: str,
    expected_tables: list[str],
) -> None:
    log_path = copy_example_log(tmp_path, log_name)
    service = CampaignAppService.load(PROJECT_ROOT / "configs" / config_name, log_path)

    validation = service.validate()
    view_data = service.collect_view_data(panel)

    assert validation.ok
    assert validation.label == "Valid"
    assert isinstance(view_data, CampaignViewData)
    for table_name in expected_tables:
        assert getattr(view_data, table_name) is not None


def test_app_service_dry_run_is_non_mutating_and_uses_existing_bundle_shape(
    tmp_path: Path,
) -> None:
    log_path = copy_example_log(tmp_path, "01_simple_2d_maximise_logei_campaign_log.csv")
    service = CampaignAppService.load(
        PROJECT_ROOT / "configs" / "01_simple_2d_maximise_logei.yaml",
        log_path,
    )
    before_bytes = log_path.read_bytes()
    before_df = service.df.copy(deep=True)

    result = service.suggest_dry_run(batch_size=1)

    assert not result.suggestions.empty
    assert not result.quality.empty
    assert set(result.bundle) == {
        "suggestions",
        "suggestions_fingerprint",
        "config_path",
        "config_fingerprint",
        "log_path",
        "log_fingerprint",
        "appended",
    }
    assert result.bundle["appended"] is False
    assert log_path.read_bytes() == before_bytes
    pd.testing.assert_frame_equal(service.df, before_df)


def test_app_service_contextual_dry_run_records_context_without_mutation(
    tmp_path: Path,
) -> None:
    config_path = copy_example_config(tmp_path, "16_contextual_logei.yaml")
    log_path = copy_example_log(tmp_path, "16_contextual_logei_campaign_log.csv")
    service = CampaignAppService.load(config_path, log_path)
    before_bytes = log_path.read_bytes()

    result = service.suggest_dry_run(
        batch_size=1,
        context_values={"feedstock_acidity": 0.25},
    )

    assert result.bundle["context_values"] == {"feedstock_acidity": 0.25}
    assert result.suggestions["feedstock_acidity"].astype(float).tolist() == [
        pytest.approx(0.25)
    ]
    assert log_path.read_bytes() == before_bytes


def test_app_service_contextual_append_rejects_changed_context_without_mutation(
    tmp_path: Path,
) -> None:
    config_path = copy_example_config(tmp_path, "16_contextual_logei.yaml")
    log_path = copy_example_log(tmp_path, "16_contextual_logei_campaign_log.csv")
    service = CampaignAppService.load(config_path, log_path)
    result = service.suggest_dry_run(
        batch_size=1,
        context_values={"feedstock_acidity": 0.25},
    )
    before = log_path.read_bytes()

    with pytest.raises(ValueError, match="Context values changed after suggestions were staged"):
        service.append_staged(
            result.bundle,
            context_values={"feedstock_acidity": 0.75},
        )
    assert log_path.read_bytes() == before


def test_app_service_contextual_append_rejects_tampered_context_metadata(
    tmp_path: Path,
) -> None:
    config_path = copy_example_config(tmp_path, "16_contextual_logei.yaml")
    log_path = copy_example_log(tmp_path, "16_contextual_logei_campaign_log.csv")
    service = CampaignAppService.load(config_path, log_path)
    result = service.suggest_dry_run(
        batch_size=1,
        context_values={"feedstock_acidity": 0.25},
    )
    result.bundle["context_values"] = {"feedstock_acidity": 0.75}
    before = log_path.read_bytes()

    with pytest.raises(ValueError, match="Context values changed after suggestions were staged"):
        service.append_staged(result.bundle)
    assert log_path.read_bytes() == before


def test_app_service_structured_dry_run_records_stage_without_mutation(
    tmp_path: Path,
) -> None:
    log_path = copy_example_log(tmp_path, "13_structured_campaign_core_campaign_log.csv")
    pd.read_csv(log_path, keep_default_na=False).query("status == 'observed'").to_csv(
        log_path,
        index=False,
    )
    service = CampaignAppService.load(
        PROJECT_ROOT / "configs" / "13_structured_campaign_core.yaml",
        log_path,
    )
    before_bytes = log_path.read_bytes()

    result = service.suggest_dry_run(batch_size=1, stage="screen")

    assert result.bundle["stage"] == "screen"
    assert result.suggestions.loc[0, "stage"] == "screen"
    assert result.suggestions.loc[0, "annealing_temperature"] == ""
    assert not result.quality.empty
    assert log_path.read_bytes() == before_bytes


def test_app_service_structured_append_rejects_changed_stage_without_mutation(
    tmp_path: Path,
) -> None:
    log_path = copy_example_log(tmp_path, "13_structured_campaign_core_campaign_log.csv")
    pd.read_csv(log_path, keep_default_na=False).query("status == 'observed'").to_csv(
        log_path,
        index=False,
    )
    service = CampaignAppService.load(
        PROJECT_ROOT / "configs" / "13_structured_campaign_core.yaml",
        log_path,
    )
    result = service.suggest_dry_run(batch_size=1, stage="screen")
    before = log_path.read_bytes()

    with pytest.raises(ValueError, match="Stage selection changed after suggestions were staged"):
        service.append_staged(result.bundle, stage="refine")

    assert log_path.read_bytes() == before


def test_app_service_structured_append_requires_matching_stage_without_mutation(
    tmp_path: Path,
) -> None:
    log_path = copy_example_log(tmp_path, "13_structured_campaign_core_campaign_log.csv")
    pd.read_csv(log_path, keep_default_na=False).query("status == 'observed'").to_csv(
        log_path,
        index=False,
    )
    service = CampaignAppService.load(
        PROJECT_ROOT / "configs" / "13_structured_campaign_core.yaml",
        log_path,
    )
    result = service.suggest_dry_run(batch_size=1, stage="screen")
    before = log_path.read_bytes()

    with pytest.raises(ValueError, match="Stage selection changed after suggestions were staged"):
        service.append_staged(result.bundle)

    assert log_path.read_bytes() == before


def test_app_service_append_staged_refreshes_session(tmp_path: Path) -> None:
    log_path = copy_example_log(tmp_path, "01_simple_2d_maximise_logei_campaign_log.csv")
    service = CampaignAppService.load(
        PROJECT_ROOT / "configs" / "01_simple_2d_maximise_logei.yaml",
        log_path,
    )
    result = service.suggest_dry_run(batch_size=1)

    append_result = service.append_staged(result.bundle)

    assert append_result.service is service
    assert append_result.validation.ok
    assert append_result.appended_fingerprint == result.bundle["suggestions_fingerprint"]
    assert len(service.df) == 3
    assert len(pd.read_csv(log_path, keep_default_na=False)) == 3


def test_app_service_append_staged_rejects_changed_config_without_mutation(
    tmp_path: Path,
) -> None:
    config_path = copy_example_config(tmp_path, "01_simple_2d_maximise_logei.yaml")
    log_path = copy_example_log(tmp_path, "01_simple_2d_maximise_logei_campaign_log.csv")
    service = CampaignAppService.load(config_path, log_path)
    result = service.suggest_dry_run(batch_size=1)
    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    before = log_path.read_bytes()

    with pytest.raises(ValueError, match="Config file changed after suggestions were staged"):
        service.append_staged(result.bundle)

    assert log_path.read_bytes() == before


def test_app_service_append_staged_rejects_changed_log_without_mutation(
    tmp_path: Path,
) -> None:
    config_path = copy_example_config(tmp_path, "01_simple_2d_maximise_logei.yaml")
    log_path = copy_example_log(tmp_path, "01_simple_2d_maximise_logei_campaign_log.csv")
    service = CampaignAppService.load(config_path, log_path)
    result = service.suggest_dry_run(batch_size=1)
    log_path.write_text(log_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    before = log_path.read_bytes()

    with pytest.raises(ValueError, match="Log file changed after suggestions were staged"):
        service.append_staged(result.bundle)

    assert log_path.read_bytes() == before


def test_app_service_append_staged_rejects_appended_fingerprint_without_mutation(
    tmp_path: Path,
) -> None:
    config_path = copy_example_config(tmp_path, "01_simple_2d_maximise_logei.yaml")
    log_path = copy_example_log(tmp_path, "01_simple_2d_maximise_logei_campaign_log.csv")
    service = CampaignAppService.load(config_path, log_path)
    result = service.suggest_dry_run(batch_size=1)
    before = log_path.read_bytes()

    with pytest.raises(ValueError, match="Staged suggestions were already appended"):
        service.append_staged(
            result.bundle,
            last_appended_fingerprint=str(result.bundle["suggestions_fingerprint"]),
        )

    assert log_path.read_bytes() == before


def test_app_service_append_staged_rejects_mutated_payload_without_mutation(
    tmp_path: Path,
) -> None:
    config_path = copy_example_config(tmp_path, "01_simple_2d_maximise_logei.yaml")
    log_path = copy_example_log(tmp_path, "01_simple_2d_maximise_logei_campaign_log.csv")
    service = CampaignAppService.load(config_path, log_path)
    result = service.suggest_dry_run(batch_size=1)
    result.bundle["suggestions"].loc[0, "row_id"] = "tampered_row"
    before = log_path.read_bytes()

    with pytest.raises(ValueError, match="Staged suggestions changed after they were staged"):
        service.append_staged(result.bundle)

    assert log_path.read_bytes() == before


def test_app_service_review_and_single_objective_mark_observed(tmp_path: Path) -> None:
    config_path = PROJECT_ROOT / "configs" / "07_cost_aware_human_review_logei.yaml"
    log_path = copy_example_log(tmp_path, "07_cost_aware_human_review_campaign_log.csv")
    service = CampaignAppService.load(config_path, log_path)
    suggestions = pd.read_csv(
        PROJECT_ROOT / "examples" / "07_cost_aware_human_review_latest_suggestions.csv",
        keep_default_na=False,
    ).head(1)
    bundle = make_staged_suggestion_bundle(suggestions, config_path, log_path)

    service.append_staged(bundle)
    row_id = str(suggestions.loc[0, "row_id"])
    review_result = service.review(row_id, "accept", "ready")
    mark_result = service.mark_observed(row_id, objective_value=70.0, actual_cost=2.2)

    assert review_result.validation.ok
    assert mark_result.validation.ok
    row = service.df.loc[service.df["row_id"] == row_id].iloc[0]
    assert row["status"] == "observed"
    assert float(row["yield_score"]) == pytest.approx(70.0)
    assert float(row["cost_actual"]) == pytest.approx(2.2)


def test_app_service_multi_objective_mark_observed_with_actual_cost(
    tmp_path: Path,
) -> None:
    config_path = PROJECT_ROOT / "configs" / "12_cost_aware_multi_objective_qlogehvi.yaml"
    log_path = copy_example_log(tmp_path, "12_cost_aware_multi_objective_campaign_log.csv")
    service = CampaignAppService.load(config_path, log_path)
    suggestions = pd.read_csv(
        PROJECT_ROOT / "examples" / "12_cost_aware_multi_objective_latest_suggestions.csv",
        keep_default_na=False,
    ).head(1)
    bundle = make_staged_suggestion_bundle(suggestions, config_path, log_path)

    service.append_staged(bundle)
    row_id = str(suggestions.loc[0, "row_id"])
    service.review(row_id, "accept", "ready")
    result = service.mark_observed(
        row_id,
        objective_values={"yield": 0.7, "selectivity": 0.6, "waste": 0.4},
        actual_cost=1.9,
    )

    assert result.validation.ok
    row = service.df.loc[service.df["row_id"] == row_id].iloc[0]
    assert row["status"] == "observed"
    assert float(row["yield"]) == pytest.approx(0.7)
    assert float(row["selectivity"]) == pytest.approx(0.6)
    assert float(row["waste"]) == pytest.approx(0.4)
    assert float(row["cost_actual"]) == pytest.approx(1.9)


def test_app_service_multi_objective_partial_values_do_not_mutate(
    tmp_path: Path,
) -> None:
    config_path = PROJECT_ROOT / "configs" / "10_multi_objective_mixed_constrained_qlogehvi.yaml"
    log_path = copy_example_log(tmp_path, "10_multi_objective_mixed_constrained_campaign_log.csv")
    service = CampaignAppService.load(config_path, log_path)
    suggestions = pd.read_csv(
        PROJECT_ROOT / "examples" / "10_multi_objective_mixed_constrained_latest_suggestions.csv",
        keep_default_na=False,
    ).head(1)
    service.append_staged(make_staged_suggestion_bundle(suggestions, config_path, log_path))
    row_id = str(suggestions.loc[0, "row_id"])
    before = log_path.read_bytes()

    with pytest.raises(LogWriteError, match="objective_values keys must exactly match"):
        service.mark_observed(row_id, objective_values={"yield_score": 70.0})

    assert log_path.read_bytes() == before


def test_app_service_nonfinite_actual_cost_does_not_mutate(tmp_path: Path) -> None:
    config_path = PROJECT_ROOT / "configs" / "07_cost_aware_human_review_logei.yaml"
    log_path = copy_example_log(tmp_path, "07_cost_aware_human_review_campaign_log.csv")
    service = CampaignAppService.load(config_path, log_path)
    suggestions = pd.read_csv(
        PROJECT_ROOT / "examples" / "07_cost_aware_human_review_latest_suggestions.csv",
        keep_default_na=False,
    ).head(1)
    service.append_staged(make_staged_suggestion_bundle(suggestions, config_path, log_path))
    row_id = str(suggestions.loc[0, "row_id"])
    service.review(row_id, "accept", "ready")
    before = log_path.read_bytes()

    with pytest.raises(LogWriteError, match="actual_cost.*finite and >= 0"):
        service.mark_observed(row_id, objective_value=70.0, actual_cost=float("inf"))

    assert log_path.read_bytes() == before


def test_app_service_invalid_review_decision_does_not_mutate(tmp_path: Path) -> None:
    config_path = PROJECT_ROOT / "configs" / "07_cost_aware_human_review_logei.yaml"
    log_path = copy_example_log(tmp_path, "07_cost_aware_human_review_campaign_log.csv")
    service = CampaignAppService.load(config_path, log_path)
    suggestions = pd.read_csv(
        PROJECT_ROOT / "examples" / "07_cost_aware_human_review_latest_suggestions.csv",
        keep_default_na=False,
    ).head(1)
    service.append_staged(make_staged_suggestion_bundle(suggestions, config_path, log_path))
    row_id = str(suggestions.loc[0, "row_id"])
    before = log_path.read_bytes()

    with pytest.raises(LogWriteError, match="Invalid review decision"):
        service.review(row_id, "maybe", "not ready")

    assert log_path.read_bytes() == before


def test_app_service_report_export_and_plot_routing(tmp_path: Path) -> None:
    log_path = copy_example_log(tmp_path, "12_cost_aware_multi_objective_campaign_log.csv")
    service = CampaignAppService.load(
        PROJECT_ROOT / "configs" / "12_cost_aware_multi_objective_qlogehvi.yaml",
        log_path,
    )

    report_text = service.report_text()
    report_path = service.export_report(tmp_path / "reports" / "campaign.txt")

    assert "BO Forge Campaign Report" in report_text
    assert report_path.exists()
    assert "Cost Summary" in report_path.read_text(encoding="utf-8")

    assert service.available_plot_kinds() == [
        "pareto",
        "hypervolume",
        "pareto_parallel",
        "cost_progress",
    ]
    for kind in service.available_plot_kinds():
        plot_path = tmp_path / "plots" / f"{kind}.png"
        result = service.plot(kind, save_path=plot_path)
        assert plot_path.exists()
        assert result.written_path == plot_path
        plt.close(result.figure)


def test_app_service_structured_stage_diagnostics_plot_routing(tmp_path: Path) -> None:
    log_path = copy_example_log(tmp_path, "13_structured_campaign_core_campaign_log.csv")
    service = CampaignAppService.load(
        PROJECT_ROOT / "configs" / "13_structured_campaign_core.yaml",
        log_path,
    )

    assert "stage_diagnostics" in service.available_plot_kinds()
    plot_path = tmp_path / "plots" / "stage_diagnostics.png"
    result = service.plot("stage_diagnostics", save_path=plot_path)

    assert plot_path.exists()
    assert result.written_path == plot_path
    plt.close(result.figure)


def test_app_service_fidelity_summary_and_diagnostics_plot_routing(tmp_path: Path) -> None:
    log_path = copy_example_log(tmp_path, "15_multi_fidelity_qmfkg_campaign_log.csv")
    service = CampaignAppService.load(
        PROJECT_ROOT / "configs" / "15_multi_fidelity_qmfkg.yaml",
        log_path,
    )

    assert callable(service.fidelity_summary)
    assert "fidelity_diagnostics" in service.available_plot_kinds()
    plot_path = tmp_path / "plots" / "fidelity_diagnostics.png"
    result = service.plot("fidelity_diagnostics", save_path=plot_path)

    assert plot_path.exists()
    assert result.written_path == plot_path
    plt.close(result.figure)


def test_app_service_context_summary_and_diagnostics_plot_routing(tmp_path: Path) -> None:
    log_path = copy_example_log(tmp_path, "16_contextual_logei_campaign_log.csv")
    service = CampaignAppService.load(
        PROJECT_ROOT / "configs" / "16_contextual_logei.yaml",
        log_path,
    )

    assert callable(service.context_summary)
    assert "context_diagnostics" in service.available_plot_kinds()
    plot_path = tmp_path / "plots" / "context_diagnostics.png"
    result = service.plot("context_diagnostics", save_path=plot_path)

    assert plot_path.exists()
    assert result.written_path == plot_path
    plt.close(result.figure)


def test_app_service_contextual_cost_review_round_trip(tmp_path: Path) -> None:
    log_path = copy_example_log(tmp_path, "20_contextual_cost_review_campaign_log.csv")
    service = CampaignAppService.load(
        PROJECT_ROOT / "configs" / "20_contextual_cost_review_logei.yaml",
        log_path,
    )
    context_values = {"feedstock_acidity": 0.5}
    before = log_path.read_bytes()

    result = service.suggest_dry_run(batch_size=1, context_values=context_values)

    assert log_path.read_bytes() == before
    assert result.suggestions.loc[0, "source"] == "cost_log_ei"
    assert result.suggestions.loc[0, "review_status"] == "pending"
    assert float(result.suggestions.loc[0, "feedstock_acidity"]) == pytest.approx(0.5)
    assert float(result.suggestions.loc[0, "cost_estimate"]) > 0
    assert result.bundle["context_values"] == context_values

    append = service.append_staged(result.bundle, context_values=context_values)
    row_id = str(result.suggestions.loc[0, "row_id"])
    append.service.review(row_id, "accept", "approved")
    append.service.mark_observed(row_id, objective_value=0.84, actual_cost=4.2)
    refreshed = CampaignAppService.load(
        PROJECT_ROOT / "configs" / "20_contextual_cost_review_logei.yaml",
        log_path,
    )
    row = refreshed.df.loc[refreshed.df["row_id"] == row_id].iloc[0]

    assert row["status"] == "observed"
    assert row["review_note"] == "approved"
    assert float(row["yield_score"]) == pytest.approx(0.84)
    assert float(row["cost_actual"]) == pytest.approx(4.2)
    assert "context_diagnostics" in refreshed.available_plot_kinds()
    assert "cost_progress" in refreshed.available_plot_kinds()
    assert refreshed.collect_view_data("Overview").context_summary is not None
    assert refreshed.collect_view_data("Overview").cost_summary is not None


def test_app_service_model_summary_and_diagnostics_plot_routing(tmp_path: Path) -> None:
    log_path = copy_example_log(tmp_path, "17_model_profile_campaign_log.csv")
    service = CampaignAppService.load(
        PROJECT_ROOT / "configs" / "17_model_profile_logei.yaml",
        log_path,
    )

    assert callable(service.model_summary)
    assert callable(service.model_profile_comparison)
    assert "model_diagnostics" in service.available_plot_kinds()
    assert "model_comparison" in service.available_plot_kinds()
    comparison = service.model_profile_comparison(profiles=["default"])
    assert comparison["model_profile"].tolist() == ["default"]
    plot_path = tmp_path / "plots" / "model_diagnostics.png"
    result = service.plot("model_diagnostics", save_path=plot_path)

    assert plot_path.exists()
    assert result.written_path == plot_path
    plt.close(result.figure)

    comparison_path = tmp_path / "plots" / "model_comparison.png"
    comparison_result = service.plot("model_comparison", save_path=comparison_path)

    assert comparison_path.exists()
    assert comparison_result.written_path == comparison_path
    plt.close(comparison_result.figure)


def test_app_service_qlog_nei_dry_run_handles_active_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = copy_example_log(tmp_path, "18_noisy_pending_qlognei_campaign_log.csv")
    service = CampaignAppService.load(
        PROJECT_ROOT / "configs" / "18_noisy_pending_qlognei.yaml",
        log_path,
    )
    candidate = values_to_unit_cube(service.config, [(0.45, 610.0)])

    def fake_optimizer(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        x_pending = kwargs["x_pending"]
        assert isinstance(x_pending, torch.Tensor)
        assert x_pending.shape == (1, 2)
        return candidate, torch.tensor(0.25, dtype=torch.double), "qlog_nei"

    monkeypatch.setattr(suggestions_module, "optimize_qlog_nei", fake_optimizer)

    result = service.suggest_dry_run(batch_size=1)

    assert result.suggestions.loc[0, "source"] == "qlog_nei"
    assert result.bundle["suggestions_fingerprint"]


def test_app_service_qlog_nehvi_dry_run_handles_active_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = copy_example_log(tmp_path, "19_multi_objective_qlognehvi_campaign_log.csv")
    service = CampaignAppService.load(
        PROJECT_ROOT / "configs" / "19_multi_objective_qlognehvi.yaml",
        log_path,
    )
    candidate = values_to_unit_cube(service.config, [(72.0, "MeCN")])
    before = log_path.read_bytes()

    def fake_optimizer(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        x_pending = kwargs["x_pending"]
        assert isinstance(x_pending, torch.Tensor)
        assert x_pending.shape == (1, candidate.shape[1])
        return candidate, torch.tensor(0.25, dtype=torch.double), "qlog_nehvi"

    monkeypatch.setattr(suggestions_module, "optimize_qlog_nehvi", fake_optimizer)

    result = service.suggest_dry_run(batch_size=1)

    assert log_path.read_bytes() == before
    assert result.suggestions.loc[0, "source"] == "qlog_nehvi"
    assert result.bundle["suggestions_fingerprint"]


def test_app_service_qlog_nei_summary_and_diagnostics_plot_routing(
    tmp_path: Path,
) -> None:
    log_path = copy_example_log(tmp_path, "18_noisy_pending_qlognei_campaign_log.csv")
    service = CampaignAppService.load(
        PROJECT_ROOT / "configs" / "18_noisy_pending_qlognei.yaml",
        log_path,
    )

    view_data = service.collect_view_data("Overview")
    summary = service.qlog_nei_summary()
    plot_path = tmp_path / "plots" / "qlog_nei_diagnostics.png"
    result = service.plot("qlog_nei_diagnostics", save_path=plot_path)

    assert view_data.qlog_nei_summary is not None
    assert summary.loc[summary["field"] == "active_pending_rows", "value"].iloc[0] == 1
    assert "qlog_nei_diagnostics" in service.available_plot_kinds()
    assert plot_path.exists()
    assert result.written_path == plot_path
    plt.close(result.figure)


def test_app_service_context_summary_handles_pending_only_log(tmp_path: Path) -> None:
    cfg = CampaignConfig.from_yaml(PROJECT_ROOT / "configs" / "16_contextual_logei.yaml")
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
    service = CampaignAppService.load(
        PROJECT_ROOT / "configs" / "16_contextual_logei.yaml",
        log_path,
    )

    view_data = service.collect_view_data("Overview")

    assert view_data.context_summary is not None
    assert view_data.context_summary["context_key"].tolist() == [
        "feedstock_acidity=0.25"
    ]
    assert int(view_data.context_summary["pending_suggestions"].iloc[0]) == 1


def test_app_service_read_helper_allowlist_exposes_only_non_mutating_helpers(
    tmp_path: Path,
) -> None:
    log_path = copy_example_log(tmp_path, "01_simple_2d_maximise_logei_campaign_log.csv")
    service = CampaignAppService.load(
        PROJECT_ROOT / "configs" / "01_simple_2d_maximise_logei.yaml",
        log_path,
    )

    assert callable(service.summary)
    assert callable(service.next_action)
    assert callable(service.suggestion_quality)
    structured_log_path = copy_example_log(
        tmp_path,
        "13_structured_campaign_core_campaign_log.csv",
    )
    structured_service = CampaignAppService.load(
        PROJECT_ROOT / "configs" / "13_structured_campaign_core.yaml",
        structured_log_path,
    )
    assert callable(structured_service.stage_summary)
    assert service.mark_observed.__func__ is CampaignAppService.mark_observed
    for mutator in ["append_suggestions", "review_suggestion", "reload"]:
        with pytest.raises(AttributeError, match=mutator):
            getattr(service, mutator)


def test_app_service_validate_failure_is_non_mutating(tmp_path: Path) -> None:
    config_path = PROJECT_ROOT / "configs" / "01_simple_2d_maximise_logei.yaml"
    log_path = copy_example_log(tmp_path, "01_simple_2d_maximise_logei_campaign_log.csv")
    valid_service = CampaignAppService.load(config_path, log_path)
    invalid_df = valid_service.df.drop(columns=["activity"])
    service = CampaignAppService.from_session(
        CampaignSession(
            config_path=config_path,
            log_path=log_path,
            config=valid_service.config,
            df=invalid_df,
        )
    )
    before = log_path.read_bytes()

    result = service.validate()

    assert not result.ok
    assert result.label == "Validation issue"
    assert "missing required columns" in result.message
    assert log_path.read_bytes() == before


def test_app_service_plot_rejects_unknown_kind(tmp_path: Path) -> None:
    log_path = copy_example_log(tmp_path, "01_simple_2d_maximise_logei_campaign_log.csv")
    service = CampaignAppService.load(
        PROJECT_ROOT / "configs" / "01_simple_2d_maximise_logei.yaml",
        log_path,
    )

    with pytest.raises(ValueError, match="Unsupported plot kind: unknown_kind"):
        service.plot("unknown_kind")
