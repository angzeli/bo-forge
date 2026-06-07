import builtins
import stat
import subprocess
import sys
import types
from pathlib import Path

import pytest

from bo_forge_app import cli as app_cli


def test_default_launcher_argv_maps_to_packaged_streamlit_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_argv = _install_fake_streamlit(monkeypatch)

    assert app_cli.run([]) == 0

    assert captured_argv["argv"][:3] == [
        "streamlit",
        "run",
        str(app_cli.packaged_streamlit_app_path()),
    ]
    assert captured_argv["argv"][3:] == [
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8501",
    ]
    assert "--server.headless" not in captured_argv["argv"]


def test_custom_host_port_no_browser_and_passthrough_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured_argv = _install_fake_streamlit(monkeypatch)
    monkeypatch.setattr(app_cli, "_detect_lan_ip", lambda: "192.168.1.25")

    assert (
        app_cli.run(
            [
                "--host",
                "0.0.0.0",
                "--port",
                "9001",
                "--no-browser",
                "--theme.base=light",
            ]
        )
        == 0
    )

    assert captured_argv["argv"] == [
        "streamlit",
        "run",
        str(app_cli.packaged_streamlit_app_path()),
        "--server.address",
        "0.0.0.0",
        "--server.port",
        "9001",
        "--server.headless",
        "true",
        "--theme.base=light",
    ]
    output = capsys.readouterr().out
    assert "LAN access: http://192.168.1.25:9001" in output
    assert "no built-in authentication" in output
    assert "Do not expose this app directly to the public internet." in output


def test_browser_flag_maps_to_streamlit_headless_false(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_argv = _install_fake_streamlit(monkeypatch)

    assert app_cli.run(["--browser"]) == 0

    assert captured_argv["argv"][-2:] == ["--server.headless", "false"]


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_hosts_do_not_print_network_warning(
    host: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args, _ = app_cli.parse_launcher_args(["--host", host])

    app_cli.print_startup_messages(args)

    output = capsys.readouterr().out
    assert "Local access:" in output
    assert "LAN safety" not in output
    assert "Network access:" not in output
    assert "no built-in authentication" not in output


@pytest.mark.parametrize("host", ["0.0.0.0", "::"])
def test_wildcard_hosts_print_network_warning(
    host: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args, _ = app_cli.parse_launcher_args(["--host", host])
    monkeypatch.setattr(app_cli, "_detect_lan_ip", lambda: "192.168.1.25")

    app_cli.print_startup_messages(args)

    output = capsys.readouterr().out
    assert "Local access: http://127.0.0.1:8501" in output
    assert "LAN access: http://192.168.1.25:8501" in output
    assert "no built-in authentication" in output
    if host == "::":
        assert "Starting BO Forge Streamlit workbench on http://[::]:8501" in output
        assert ":::8501" not in output


@pytest.mark.parametrize("host", ["192.168.1.25", "dev-box.local"])
def test_specific_non_loopback_hosts_print_network_warning_without_localhost_url(
    host: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args, _ = app_cli.parse_launcher_args(["--host", host])

    app_cli.print_startup_messages(args)

    output = capsys.readouterr().out
    assert f"Network access: http://{host}:8501" in output
    assert "Local access: http://127.0.0.1:8501" not in output
    assert "no built-in authentication" in output


@pytest.mark.parametrize("port", ["1", "8501", "65535"])
def test_valid_launcher_ports_are_accepted(port: str) -> None:
    args, _ = app_cli.parse_launcher_args(["--port", port])

    assert args.port == int(port)


@pytest.mark.parametrize("port", ["0", "-1", "70000", "abc"])
def test_invalid_launcher_ports_fail_clearly(
    port: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        app_cli.parse_launcher_args(["--port", port])

    assert exc_info.value.code == 2
    assert "port must" in capsys.readouterr().err


@pytest.mark.parametrize(
    "passthrough",
    [
        ["--server.address", "127.0.0.1"],
        ["--server.port=9002"],
        ["--server.headless", "true"],
    ],
)
def test_conflicting_streamlit_server_passthrough_fails_clearly(
    passthrough: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        app_cli.parse_launcher_args(passthrough)

    assert exc_info.value.code == 2
    assert "Conflicting Streamlit server option" in capsys.readouterr().err


def test_help_exits_without_importing_streamlit(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_streamlit_imports(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        app_cli.run(["--help"])

    assert exc_info.value.code == 0


def test_python_module_help_works_without_starting_streamlit() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "bo_forge_app", "--help"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--host" in completed.stdout
    assert "--make-launcher" in completed.stdout
    assert completed.stderr == ""


def test_missing_streamlit_exits_cleanly_only_when_launching(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _block_streamlit_imports(monkeypatch)

    assert app_cli.run([]) == 1

    captured = capsys.readouterr()
    assert 'pip install "bo-forge[app]"' in captured.err
    assert "Starting BO Forge Streamlit workbench" not in captured.out
    assert "Traceback" not in captured.err


def test_make_launcher_run_path_does_not_import_streamlit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher_path = tmp_path / "BO-Forge.command"
    written = {}

    def fake_write(path: Path, args: object, passthrough: list[str]) -> Path:
        written["path"] = path
        written["passthrough"] = passthrough
        return launcher_path

    _block_streamlit_imports(monkeypatch)
    monkeypatch.setattr(app_cli, "write_macos_launcher", fake_write)

    assert app_cli.run(["--make-launcher", str(launcher_path), "--theme.base=light"]) == 0
    assert written == {"path": launcher_path, "passthrough": ["--theme.base=light"]}


def test_write_macos_launcher_writes_executable_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher_path = tmp_path / "BO-Forge.command"
    args, passthrough = app_cli.parse_launcher_args(
        [
            "--host",
            "0.0.0.0",
            "--port",
            "8502",
            "--no-browser",
            "--theme.base=light",
        ]
    )
    monkeypatch.setattr(app_cli.sys, "platform", "darwin")
    monkeypatch.setattr(app_cli.shutil, "which", lambda name: "/opt/bin/bo-forge-app")

    written_path = app_cli.write_macos_launcher(launcher_path, args, passthrough)

    assert written_path == launcher_path.resolve()
    mode = launcher_path.stat().st_mode
    assert mode & stat.S_IXUSR
    text = launcher_path.read_text(encoding="utf-8")
    assert "cd " in text
    assert "/opt/bin/bo-forge-app" in text
    assert "--host 0.0.0.0" in text
    assert "--port 8502" in text
    assert "--no-browser" in text
    assert "--theme.base=light" in text
    assert "--make-launcher" not in text


def test_write_macos_launcher_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher_path = tmp_path / "BO-Forge.command"
    launcher_path.write_text("exists", encoding="utf-8")
    args, passthrough = app_cli.parse_launcher_args([])
    monkeypatch.setattr(app_cli.sys, "platform", "darwin")

    with pytest.raises(app_cli.LauncherError, match="Refusing to overwrite"):
        app_cli.write_macos_launcher(launcher_path, args, passthrough)


def test_write_macos_launcher_requires_existing_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, passthrough = app_cli.parse_launcher_args([])
    missing_path = tmp_path / "missing" / "BO-Forge.command"
    monkeypatch.setattr(app_cli.sys, "platform", "darwin")

    with pytest.raises(app_cli.LauncherError, match="parent directory does not exist"):
        app_cli.write_macos_launcher(missing_path, args, passthrough)


def test_write_macos_launcher_rejects_non_macos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, passthrough = app_cli.parse_launcher_args([])
    monkeypatch.setattr(app_cli.sys, "platform", "linux")

    with pytest.raises(app_cli.LauncherError, match="only supported on macOS"):
        app_cli.write_macos_launcher(tmp_path / "BO-Forge.command", args, passthrough)


def test_write_macos_launcher_requires_installed_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, passthrough = app_cli.parse_launcher_args([])
    monkeypatch.setattr(app_cli.sys, "platform", "darwin")
    monkeypatch.setattr(app_cli.shutil, "which", lambda name: None)

    with pytest.raises(app_cli.LauncherError, match="Could not resolve"):
        app_cli.write_macos_launcher(tmp_path / "BO-Forge.command", args, passthrough)


def test_make_launcher_write_text_failure_returns_clean_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    launcher_path = tmp_path / "BO-Forge.command"
    monkeypatch.setattr(app_cli.sys, "platform", "darwin")
    monkeypatch.setattr(app_cli.shutil, "which", lambda name: "/opt/bin/bo-forge-app")

    def fail_write_text(self: Path, *args: object, **kwargs: object) -> int:
        raise OSError("disk is full")

    monkeypatch.setattr(Path, "write_text", fail_write_text)

    assert app_cli.run(["--make-launcher", str(launcher_path)]) == 1
    error_output = capsys.readouterr().err
    assert "Could not write launcher" in error_output
    assert "disk is full" in error_output


def test_make_launcher_chmod_failure_returns_clean_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    launcher_path = tmp_path / "BO-Forge.command"
    monkeypatch.setattr(app_cli.sys, "platform", "darwin")
    monkeypatch.setattr(app_cli.shutil, "which", lambda name: "/opt/bin/bo-forge-app")

    def fail_chmod(self: Path, *args: object, **kwargs: object) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "chmod", fail_chmod)

    assert app_cli.run(["--make-launcher", str(launcher_path)]) == 1
    error_output = capsys.readouterr().err
    assert "Could not write launcher" in error_output
    assert "permission denied" in error_output


def _install_fake_streamlit(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    captured_argv: dict[str, list[str]] = {}
    streamlit_module = types.ModuleType("streamlit")
    web_module = types.ModuleType("streamlit.web")
    cli_module = types.ModuleType("streamlit.web.cli")

    def fake_streamlit_main() -> None:
        captured_argv["argv"] = list(sys.argv)

    cli_module.main = fake_streamlit_main  # type: ignore[attr-defined]
    web_module.cli = cli_module  # type: ignore[attr-defined]
    streamlit_module.web = web_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_module)
    monkeypatch.setitem(sys.modules, "streamlit.web", web_module)
    monkeypatch.setitem(sys.modules, "streamlit.web.cli", cli_module)
    return captured_argv


def _block_streamlit_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    for module_name in ["streamlit", "streamlit.web", "streamlit.web.cli"]:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "streamlit" or name.startswith("streamlit."):
            raise ModuleNotFoundError("No module named 'streamlit'", name="streamlit")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
