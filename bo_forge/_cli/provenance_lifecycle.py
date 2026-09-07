"""Preview-first command-line provenance lifecycle operations."""

from __future__ import annotations

import json
from pathlib import Path

from bo_forge.errors import ProvenanceError


def register_lifecycle_commands(subparsers, add_config_log_arguments):
    for operation in ("adopt", "migrate", "accept-config", "fork"):
        parser = subparsers.add_parser(
            f"provenance-{operation}", help=f"Preview or explicitly apply provenance {operation}."
        )
        add_config_log_arguments(parser, include_provenance_policy=False)
        parser.add_argument(
            "--apply", action="store_true", help="Apply a previously reviewed preview."
        )
        parser.add_argument("--reason", default="", help="Nonempty reason required when applying.")
        parser.add_argument("--preview", type=Path, help="JSON preview file required with --apply.")
        if operation == "fork":
            parser.add_argument("--destination", required=True)
            parser.add_argument(
                "--config-changes",
                default="{}",
                help="JSON mapping of campaign_name, bo, model changes.",
            )
        parser.set_defaults(handler=_run, lifecycle_operation=operation)


def _run(args):
    from bo_forge._campaign.provenance_lifecycle import lifecycle

    try:
        expected = None
        if args.preview is not None:
            preview = json.loads(args.preview.read_text(encoding="utf-8"))
            if not isinstance(preview, dict) or not isinstance(
                preview.get("expected_identities"), dict
            ):
                raise ValueError("Preview must be an object containing expected_identities.")
            expected = preview["expected_identities"]
        changes = json.loads(getattr(args, "config_changes", "{}"))
        if not isinstance(changes, dict):
            raise ValueError("Config changes must be a JSON object.")
    except (OSError, ValueError, KeyError) as exc:
        raise ProvenanceError(f"Invalid lifecycle preview or config changes: {exc}") from exc
    result = lifecycle(
        args.lifecycle_operation,
        args.config,
        args.log,
        apply=args.apply,
        reason=args.reason,
        expected_identities=expected,
        destination=getattr(args, "destination", None),
        config_changes=changes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
