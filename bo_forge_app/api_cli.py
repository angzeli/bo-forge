"""Console entrypoint for the experimental BO Forge FastAPI probe."""

from __future__ import annotations

import argparse
import ipaddress
import math
import sys
from pathlib import Path

DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8765
DEFAULT_STAGE_TTL_SECONDS = 1800.0
DEFAULT_MAX_STAGED_BATCHES = 128
API_INSTALL_HINT = 'pip install "bo-forge[api]"'


class ApiLauncherError(ValueError):
    """User-facing API launcher error."""


def build_parser() -> argparse.ArgumentParser:
    """Build the experimental API launcher parser."""
    parser = argparse.ArgumentParser(
        prog="bo-forge-api",
        description="Launch the experimental BO Forge FastAPI probe.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory for relative config/log paths.",
    )
    parser.add_argument("--host", default=DEFAULT_API_HOST, help="Host address for Uvicorn.")
    parser.add_argument(
        "--port",
        type=_parse_port,
        default=DEFAULT_API_PORT,
        help="Port for Uvicorn.",
    )
    parser.add_argument(
        "--stage-ttl-seconds",
        type=_parse_positive_float,
        default=DEFAULT_STAGE_TTL_SECONDS,
        help="Lifetime of an in-memory server-managed staged batch.",
    )
    parser.add_argument(
        "--max-staged-batches",
        type=_parse_positive_int,
        default=DEFAULT_MAX_STAGED_BATCHES,
        help="Maximum active in-memory staged batches.",
    )
    return parser


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse API launcher arguments."""
    return build_parser().parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    """Run the experimental API launcher."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        root = Path(args.root).expanduser().resolve()
        if not root.is_dir():
            raise ApiLauncherError(f"API root must be an existing directory: {root}")
        try:
            import uvicorn

            from bo_forge_app.api import create_app
        except ModuleNotFoundError as exc:
            if exc.name in {"fastapi", "pydantic", "uvicorn"}:
                print(
                    f"Error: API dependencies are not installed. Install with: {API_INSTALL_HINT}",
                    file=sys.stderr,
                )
                return 1
            raise
        app = create_app(
            root,
            stage_ttl_seconds=float(args.stage_ttl_seconds),
            max_staged_batches=int(args.max_staged_batches),
        )
    except ApiLauncherError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_startup_messages(args, root)
    uvicorn.run(app, host=str(args.host), port=int(args.port))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Console-script wrapper for the experimental API launcher."""
    return run(argv)


def print_startup_messages(args: argparse.Namespace, root: Path) -> None:
    """Print API startup and safety guidance."""
    host = str(args.host)
    print(f"Starting BO Forge API probe on http://{_url_host(host)}:{args.port}")
    print(f"Root: {root}")
    print(
        "Server staging: "
        f"TTL={args.stage_ttl_seconds:g}s, max active batches={args.max_staged_batches}"
    )
    if _host_requires_network_warning(host):
        print("Network safety: BO Forge API probe has no built-in authentication.")
        print("Use only on localhost, a trusted LAN, VPN, or SSH tunnel.")
        print("Do not expose this API directly to the public internet.")
        print("The API reads and writes files under the configured root directory.")
    print("Press Ctrl+C in this terminal to stop the API probe.")


def _host_requires_network_warning(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return False
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return True
    return not address.is_loopback


def _url_host(host: str) -> str:
    if ":" in host and not (host.startswith("[") and host.endswith("]")):
        return f"[{host}]"
    return host


def _parse_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "port must be an integer between 1 and 65535"
        ) from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _parse_positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a finite positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a finite positive number")
    return parsed


def _parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
