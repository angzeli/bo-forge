"""Shared campaign arguments and lazy session loading for internal CLI commands."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bo_forge.session import CampaignSession


def _add_config_log_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_provenance_policy: bool = True,
) -> None:
    parser.add_argument("--config", required=True, type=Path, help="Campaign YAML config path.")
    parser.add_argument("--log", required=True, type=Path, help="Campaign CSV log path.")
    if include_provenance_policy:
        parser.add_argument(
            "--require-provenance",
            action="store_true",
            help="Reject legacy campaigns that do not have a provenance manifest.",
        )


def _load_session(args: argparse.Namespace) -> CampaignSession:
    from bo_forge.session import CampaignSession

    policy = "required" if args.require_provenance else "compatible"
    return CampaignSession.from_files(
        args.config,
        args.log,
        provenance_policy=policy,
    )
