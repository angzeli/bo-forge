from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from bo_forge.config import CampaignConfig
from bo_forge.validation import canonical_columns

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_SUGGESTION_FIXTURES = {
    "07_cost_aware_human_review_latest_suggestions.csv": (
        "07_cost_aware_human_review_logei.yaml",
        {
            "row_id": "test_cost_review_suggestion",
            "iteration": 10,
            "status": "suggested",
            "source": "cost_log_ei",
            "review_status": "pending",
            "catalyst_loading": 0.1014052436476374,
            "reaction_time": 38,
            "base_equivalents": 0.2,
            "solvent": "EtOH",
            "cost_estimate": 2.52,
            "predicted_mean": 74.64187094254942,
            "predicted_std": 2.55080677259746,
            "acquisition": -1.165820855361833,
            "utility": -2.425820855361833,
        },
    ),
    "10_multi_objective_mixed_constrained_latest_suggestions.csv": (
        "10_multi_objective_mixed_constrained_qlogehvi.yaml",
        {
            "row_id": "test_multi_objective_suggestion",
            "iteration": 10,
            "status": "suggested",
            "source": "qlog_ehvi",
            "catalyst_loading": 0.19041780074101292,
            "reaction_time": 80,
            "base_equivalents": 0.5,
            "solvent": "MeCN",
            "predicted_mean_yield_score": 67.54020273651318,
            "predicted_std_yield_score": 7.821451917400719,
            "predicted_mean_waste_score": 16.45918120976166,
            "predicted_std_waste_score": 3.375251177576889,
            "acquisition": 2.2997515434678113,
        },
    ),
    "12_cost_aware_multi_objective_latest_suggestions.csv": (
        "12_cost_aware_multi_objective_qlogehvi.yaml",
        {
            "row_id": "test_cost_multi_objective_suggestion",
            "iteration": 9,
            "status": "suggested",
            "source": "cost_qlog_ehvi",
            "review_status": "pending",
            "catalyst_loading": 0.11302549958229066,
            "reaction_time": 31,
            "base_equivalents": 0.5,
            "solvent": "DMF",
            "cost_estimate": 1.87,
            "predicted_mean_yield": 0.6417641723864542,
            "predicted_std_yield": 0.22718860111785064,
            "predicted_mean_selectivity": 0.609158873370124,
            "predicted_std_selectivity": 0.10965381715709152,
            "predicted_mean_waste": 0.5716708357725511,
            "predicted_std_waste": 0.13488978333715257,
            "acquisition": -4.747719959642543,
            "utility": -6.354219959642543,
        },
    ),
}


@pytest.fixture
def suggestion_fixture() -> Callable[[str], pd.DataFrame]:
    """Return deterministic suggestion rows without relying on ignored artifacts."""

    def make(name: str) -> pd.DataFrame:
        config_name, values = _SUGGESTION_FIXTURES[name]
        config = CampaignConfig.from_yaml(PROJECT_ROOT / "configs" / config_name)
        columns = canonical_columns(config)
        row = {column: "" for column in columns}
        row.update(values)
        return pd.DataFrame([row], columns=columns)

    return make
