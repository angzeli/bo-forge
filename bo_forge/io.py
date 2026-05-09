"""Small IO helpers for canonical campaign files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bo_forge.config import CampaignConfig
from bo_forge.validation import canonical_columns


def empty_campaign_log(config: CampaignConfig) -> pd.DataFrame:
    """Create an empty canonical campaign log DataFrame."""
    return pd.DataFrame(columns=canonical_columns(config))


def write_campaign_log(path: str | Path, df: pd.DataFrame) -> None:
    """Write a campaign log CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.17g")
