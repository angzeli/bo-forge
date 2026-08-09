from pathlib import Path

import pytest

from bo_forge.config import CampaignConfig
from bo_forge.session import CampaignSession
from bo_forge.validation import canonical_columns
from bo_forge_app.streamlit_helpers import available_plot_kinds

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SINGLE_REPORT_KEYS = (
    "summary",
    "next_action",
    "model_summary",
    "best_observation",
    "best_replicate_group",
    "replicate_summary",
    "pending_suggestions",
    "review_queue",
    "cost_summary",
)
MULTI_REPORT_KEYS = (
    "summary",
    "next_action",
    "model_summary",
    "pareto_summary",
    "pareto_front",
    "pending_suggestions",
)


@pytest.mark.parametrize(
    ("config_name", "log_name", "expected_report_keys", "expected_plot_kinds"),
    [
        (
            "01_simple_2d_maximise_logei",
            "01_simple_2d_maximise_logei",
            SINGLE_REPORT_KEYS,
            ["progress", "diagnostics", "model_diagnostics", "model_comparison"],
        ),
        (
            "07_cost_aware_human_review_logei",
            "07_cost_aware_human_review",
            SINGLE_REPORT_KEYS,
            [
                "progress",
                "diagnostics",
                "model_diagnostics",
                "model_comparison",
                "cost_progress",
            ],
        ),
        (
            "08_replicate_aware_logei",
            "08_replicate_aware",
            SINGLE_REPORT_KEYS,
            [
                "progress",
                "diagnostics",
                "model_diagnostics",
                "model_comparison",
                "replicates",
            ],
        ),
        (
            "10_multi_objective_mixed_constrained_qlogehvi",
            "10_multi_objective_mixed_constrained",
            MULTI_REPORT_KEYS,
            ["pareto", "hypervolume"],
        ),
        (
            "13_structured_campaign_core",
            "13_structured_campaign_core",
            (*SINGLE_REPORT_KEYS, "stage_summary"),
            ["progress", "diagnostics", "stage_diagnostics"],
        ),
        (
            "15_multi_fidelity_qmfkg",
            "15_multi_fidelity_qmfkg",
            (*SINGLE_REPORT_KEYS, "fidelity_summary", "fidelity_coverage"),
            ["progress", "diagnostics", "fidelity_diagnostics", "fidelity_progress"],
        ),
        (
            "16_contextual_logei",
            "16_contextual_logei",
            (*SINGLE_REPORT_KEYS, "context_summary"),
            [
                "progress",
                "diagnostics",
                "model_diagnostics",
                "model_comparison",
                "context_diagnostics",
            ],
        ),
        (
            "18_noisy_pending_qlognei",
            "18_noisy_pending_qlognei",
            (*SINGLE_REPORT_KEYS, "qlog_nei_summary"),
            [
                "progress",
                "diagnostics",
                "model_diagnostics",
                "model_comparison",
                "qlog_nei_diagnostics",
            ],
        ),
        (
            "19_multi_objective_qlognehvi",
            "19_multi_objective_qlognehvi",
            (*MULTI_REPORT_KEYS, "review_queue"),
            ["pareto", "hypervolume"],
        ),
        (
            "21_contextual_replicate_logei",
            "21_contextual_replicate",
            (*SINGLE_REPORT_KEYS, "context_summary"),
            [
                "progress",
                "diagnostics",
                "model_diagnostics",
                "model_comparison",
                "context_diagnostics",
                "cost_progress",
                "replicates",
            ],
        ),
    ],
)
def test_representative_campaign_contracts_remain_stable(
    config_name: str,
    log_name: str,
    expected_report_keys: tuple[str, ...],
    expected_plot_kinds: list[str],
) -> None:
    config_path = PROJECT_ROOT / "configs" / f"{config_name}.yaml"
    log_path = PROJECT_ROOT / "examples" / f"{log_name}_campaign_log.csv"
    config = CampaignConfig.from_yaml(config_path)
    campaign = CampaignSession.from_files(config_path, log_path)

    campaign.validate()
    assert list(campaign.df.columns) == canonical_columns(config)
    assert list(campaign.summary().columns) == ["field", "value"]
    assert list(campaign.next_action().columns) == [
        "campaign_status",
        "action",
        "reason",
        "suggested_call",
    ]
    assert tuple(campaign.report()) == expected_report_keys
    assert available_plot_kinds(config) == expected_plot_kinds
