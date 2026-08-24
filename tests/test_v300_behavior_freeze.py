"""Representative workflow freeze for the v3 architecture reset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from bo_forge import CampaignConfig, CampaignSession, suggest_next
from bo_forge.validation import canonical_columns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVE_CAMPAIGNS = (
    (
        "standard",
        "01_simple_2d_maximise_logei.yaml",
        "01_simple_2d_maximise_logei_campaign_log.csv",
    ),
    (
        "cost",
        "07_cost_aware_human_review_logei.yaml",
        "07_cost_aware_human_review_campaign_log.csv",
    ),
    (
        "review",
        "07_cost_aware_human_review_logei.yaml",
        "07_cost_aware_human_review_campaign_log.csv",
    ),
    ("replicate", "08_replicate_aware_logei.yaml", "08_replicate_aware_campaign_log.csv"),
    (
        "multi-objective",
        "10_multi_objective_mixed_constrained_qlogehvi.yaml",
        "10_multi_objective_mixed_constrained_campaign_log.csv",
    ),
    (
        "structured",
        "13_structured_campaign_core.yaml",
        "13_structured_campaign_core_campaign_log.csv",
    ),
    ("contextual", "16_contextual_logei.yaml", "16_contextual_logei_campaign_log.csv"),
    ("qlog-nei", "18_noisy_pending_qlognei.yaml", "18_noisy_pending_qlognei_campaign_log.csv"),
    (
        "qlog-nehvi",
        "19_multi_objective_qlognehvi.yaml",
        "19_multi_objective_qlognehvi_campaign_log.csv",
    ),
    (
        "qmfkg",
        "22_discrete_multi_fidelity_qmfkg.yaml",
        "22_discrete_multi_fidelity_qmfkg_campaign_log.csv",
    ),
)

INITIAL_SUGGESTION_CASES = {
    "standard": ("01_simple_2d_maximise_logei.yaml", None, None),
    "cost_review": ("07_cost_aware_human_review_logei.yaml", None, None),
    "replicate": ("08_replicate_aware_logei.yaml", None, None),
    "multi_objective": (
        "10_multi_objective_mixed_constrained_qlogehvi.yaml",
        None,
        None,
    ),
    "structured": ("13_structured_campaign_core.yaml", "screen", None),
    "contextual": (
        "16_contextual_logei.yaml",
        None,
        {"feedstock_acidity": 0.2},
    ),
    "qlog_nei": ("18_noisy_pending_qlognei.yaml", None, None),
    "qlog_nehvi": ("19_multi_objective_qlognehvi.yaml", None, None),
    "qmfkg": ("22_discrete_multi_fidelity_qmfkg.yaml", None, None),
}
V253_INITIAL_SUGGESTION_DIGEST = (
    "910f77ebd4e1a39bf50ad01fe0fd31c6a3648a35535ff17bf5d7b6869b1f3d94"
)


@pytest.mark.parametrize(
    ("workflow", "config_name", "log_name"),
    REPRESENTATIVE_CAMPAIGNS,
    ids=[item[0] for item in REPRESENTATIVE_CAMPAIGNS],
)
def test_representative_campaign_reads_are_valid_and_non_mutating(
    workflow: str,
    config_name: str,
    log_name: str,
) -> None:
    del workflow
    config_path = PROJECT_ROOT / "configs" / config_name
    log_path = PROJECT_ROOT / "examples" / log_name
    config_before = config_path.read_bytes()
    log_before = log_path.read_bytes()

    campaign = CampaignSession.from_files(config_path, log_path)

    assert campaign.validate() is None
    assert campaign.df.columns.tolist() == canonical_columns(campaign.config)
    assert not campaign.summary().empty
    assert not campaign.next_action().empty
    assert config_path.read_bytes() == config_before
    assert log_path.read_bytes() == log_before


def test_representative_reports_are_read_only() -> None:
    for _workflow, config_name, log_name in REPRESENTATIVE_CAMPAIGNS:
        config_path = PROJECT_ROOT / "configs" / config_name
        log_path = PROJECT_ROOT / "examples" / log_name
        before = log_path.read_bytes()

        report = CampaignSession.from_files(config_path, log_path).report()

        assert "summary" in report
        assert not report["summary"].empty
        assert log_path.read_bytes() == before


def test_initial_suggestion_payloads_match_v253_contract() -> None:
    """Freeze seeded candidate values, ordering, columns, and public metadata."""
    contract: dict[str, object] = {}
    for workflow, (config_name, stage, context_values) in INITIAL_SUGGESTION_CASES.items():
        config = CampaignConfig.from_yaml(PROJECT_ROOT / "configs" / config_name)
        empty_log = pd.DataFrame(columns=canonical_columns(config))
        suggestions = suggest_next(
            config,
            empty_log,
            batch_size=2,
            stage=stage,
            context_values=context_values,
        ).drop(columns="row_id")
        records = suggestions.fillna("").to_dict(orient="records")
        for row in records:
            if row.get("replicate_group"):
                row["replicate_group"] = "<generated>"
        contract[workflow] = {
            "columns": suggestions.columns.tolist(),
            "records": records,
        }

    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(payload).hexdigest() == V253_INITIAL_SUGGESTION_DIGEST
