"""Console entrypoint for launching the packaged BO Forge Streamlit app."""

from __future__ import annotations

import argparse
import ipaddress
import shlex
import shutil
import socket
import sys
from importlib import resources
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8501
CONFLICTING_STREAMLIT_OPTIONS = {
    "--server.address",
    "--server.port",
    "--server.headless",
}


def packaged_streamlit_app_path() -> Path:
    """Return the installed Streamlit app script path."""
    app_resource = resources.files("bo_forge_app").joinpath("streamlit_app.py")

    return Path(str(app_resource)).resolve()


class LauncherError(ValueError):
    """User-facing launcher configuration error."""


def build_parser() -> argparse.ArgumentParser:
    """Build the BO Forge app launcher parser."""
    parser = argparse.ArgumentParser(
        prog="bo-forge-app",
        description="Launch the local BO Forge Streamlit workbench.",
        allow_abbrev=False,
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host address for Streamlit.")
    parser.add_argument(
        "--port",
        type=_parse_port,
        default=DEFAULT_PORT,
        help="Port for Streamlit.",
    )
    browser_group = parser.add_mutually_exclusive_group()
    browser_group.add_argument(
        "--no-browser",
        action="store_true",
        help="Start Streamlit without opening a browser window.",
    )
    browser_group.add_argument(
        "--browser",
        action="store_true",
        help="Ask Streamlit to open a browser window.",
    )
    parser.add_argument(
        "--make-launcher",
        type=Path,
        metavar="PATH",
        help="Write a double-clickable macOS .command launcher and exit.",
    )
    return parser


def parse_launcher_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse BO Forge launcher arguments and return remaining Streamlit args."""
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)
    conflicts = _conflicting_passthrough_options(passthrough)
    if conflicts:
        joined = ", ".join(conflicts)
        parser.error(
            "Conflicting Streamlit server option(s): "
            f"{joined}. Use BO Forge launcher flags --host, --port, --browser, "
            "or --no-browser instead."
        )
    return args, passthrough


def build_streamlit_argv(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    """Translate BO Forge launcher arguments into Streamlit argv."""
    streamlit_args = [
        "streamlit",
        "run",
        str(packaged_streamlit_app_path()),
        "--server.address",
        str(args.host),
        "--server.port",
        str(args.port),
    ]
    if args.no_browser:
        streamlit_args.extend(["--server.headless", "true"])
    elif args.browser:
        streamlit_args.extend(["--server.headless", "false"])
    streamlit_args.extend(passthrough)
    return streamlit_args


def print_startup_messages(args: argparse.Namespace) -> None:
    """Print local/LAN startup guidance before Streamlit takes over."""
    host = str(args.host)
    print(f"Starting BO Forge Streamlit workbench on http://{_url_host(host)}:{args.port}")
    if _host_requires_network_warning(host):
        if _is_unspecified_bind(host):
            print(f"Local access: http://127.0.0.1:{args.port}")
            lan_ip = _detect_lan_ip()
            if lan_ip is None:
                print(f"LAN access: http://<host-machine-lan-ip>:{args.port}")
            else:
                print(f"LAN access: http://{lan_ip}:{args.port}")
        else:
            print(f"Network access: http://{_url_host(host)}:{args.port}")
        print("LAN safety: BO Forge has no built-in authentication.")
        print("Use only on a trusted LAN, VPN, or SSH tunnel.")
        print("Do not expose this app directly to the public internet.")
        print("The app reads and writes files on this host machine.")
    else:
        print(f"Local access: http://{_url_host(host)}:{args.port}")
    print("Press Ctrl+C in this terminal to stop the local Streamlit service.")


def _host_requires_network_warning(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return False
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return True
    return not address.is_loopback


def _is_unspecified_bind(host: str) -> bool:
    try:
        return ipaddress.ip_address(host.strip()).is_unspecified
    except ValueError:
        return False


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


def write_macos_launcher(
    path: str | Path,
    args: argparse.Namespace,
    passthrough: list[str],
) -> Path:
    """Write a double-clickable macOS .command launcher without starting Streamlit."""
    if sys.platform != "darwin":
        raise LauncherError("--make-launcher is only supported on macOS.")
    launcher_path = Path(path).expanduser()
    if not launcher_path.parent.exists():
        raise LauncherError(f"Launcher parent directory does not exist: {launcher_path.parent}")
    if launcher_path.exists():
        raise LauncherError(f"Refusing to overwrite existing launcher: {launcher_path}")
    command = shutil.which("bo-forge-app")
    if command is None:
        raise LauncherError("Could not resolve installed 'bo-forge-app' executable.")

    command_args = _launcher_command_args(args, passthrough)
    quoted_args = " ".join(shlex.quote(part) for part in command_args)
    cwd = Path.cwd().resolve()
    try:
        launcher_path.write_text(
            "#!/bin/zsh\n"
            "set -e\n"
            f"cd {shlex.quote(str(cwd))}\n"
            f"exec {shlex.quote(command)} {quoted_args}\n",
            encoding="utf-8",
        )
        launcher_path.chmod(0o755)
    except OSError as exc:
        raise LauncherError(f"Could not write launcher '{launcher_path}': {exc}") from exc
    return launcher_path.resolve()


def run(argv: list[str] | None = None) -> int:
    """Run the BO Forge app launcher."""
    args, passthrough = parse_launcher_args(sys.argv[1:] if argv is None else argv)
    if args.make_launcher is not None:
        try:
            launcher_path = write_macos_launcher(args.make_launcher, args, passthrough)
        except LauncherError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(f"Wrote BO Forge launcher: {launcher_path}")
        return 0

    try:
        from streamlit.web import cli as streamlit_cli
    except ModuleNotFoundError as exc:
        if exc.name == "streamlit":
            print(
                "Error: Streamlit is not installed. "
                'Install the app extra with: pip install "bo-forge[app]"',
                file=sys.stderr,
            )
            return 1
        raise

    print_startup_messages(args)
    sys.argv = build_streamlit_argv(args, passthrough)
    streamlit_cli.main()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Console-script wrapper for the BO Forge app launcher."""
    return run(argv)


def _conflicting_passthrough_options(passthrough: list[str]) -> list[str]:
    conflicts = []
    for arg in passthrough:
        option = arg.split("=", maxsplit=1)[0]
        if option in CONFLICTING_STREAMLIT_OPTIONS and option not in conflicts:
            conflicts.append(option)
    return conflicts


def _launcher_command_args(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    command_args = ["--host", str(args.host), "--port", str(args.port)]
    if args.no_browser:
        command_args.append("--no-browser")
    elif args.browser:
        command_args.append("--browser")
    command_args.extend(passthrough)
    return command_args


def _detect_lan_ip() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except OSError:
        return None
