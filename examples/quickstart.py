"""Minimal BO Forge quickstart.

Run from the repository root with:

    ./.venv/bin/python examples/quickstart.py

The script copies the seed log to an ignored working CSV, asks for one
suggestion, simulates a result, records it with mark_observed(), and reloads
the campaign log.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bo_forge import (  # noqa: E402
    CampaignConfig,
    append_suggestions,
    load_campaign_log,
    mark_observed,
    suggest_next,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "simple_2d.yaml"
SEED_LOG_PATH = PROJECT_ROOT / "examples" / "simple_2d_campaign_log.csv"
WORKING_LOG_PATH = PROJECT_ROOT / "examples" / "quickstart_working_log.csv"


def simulated_activity(row) -> float:
    """Small deterministic objective used only for the quickstart example."""
    ratio = float(row["precursor_ratio"])
    temperature = float(row["annealing_temperature"])
    value = 2.2 - 4.0 * (ratio - 0.62) ** 2
    value -= ((temperature - 710.0) / 170.0) ** 2
    value += 0.04 * np.sin(10.0 * ratio)
    value += 0.03 * np.cos(temperature / 55.0)
    return float(value)


def main() -> None:
    config = CampaignConfig.from_yaml(CONFIG_PATH)
    shutil.copyfile(SEED_LOG_PATH, WORKING_LOG_PATH)

    df = load_campaign_log(WORKING_LOG_PATH, config)
    suggestion = suggest_next(config, df, batch_size=1)
    append_suggestions(WORKING_LOG_PATH, suggestion)

    row = suggestion.iloc[0]
    result = simulated_activity(row)
    mark_observed(WORKING_LOG_PATH, str(row["row_id"]), result)

    updated = load_campaign_log(WORKING_LOG_PATH, config)

    print("Suggested experiment:")
    print(suggestion.to_string(index=False))
    print()
    print(f"Recorded activity: {result:.6f}")
    print()
    print("Updated campaign tail:")
    print(updated.tail(3).to_string(index=False))
    print()
    print(f"Working log written to: {WORKING_LOG_PATH}")


if __name__ == "__main__":
    main()
