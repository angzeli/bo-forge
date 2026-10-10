"""Dependency-resolved installation probe tests."""

import subprocess

import pytest

from tests import _release_readiness_support as support
from tests._release_readiness_support import (
    Path,
    _pip_install_command,
    _venv_create_command,
)


def test_release_dependency_probe_commands_are_isolated_when_enabled(tmp_path: Path) -> None:
    venv_dir = tmp_path / "venv"
    pip = venv_dir / "bin" / "pip"
    wheel = tmp_path / "bo_forge.whl"

    assert "--system-site-packages" not in _venv_create_command(
        venv_dir,
        resolve_dependencies=True,
    )
    assert "--no-deps" not in _pip_install_command(
        pip,
        wheel,
        install_args=[],
        resolve_dependencies=True,
    )
    assert "--system-site-packages" in _venv_create_command(
        venv_dir,
        resolve_dependencies=False,
    )
    assert "--no-deps" in _pip_install_command(
        pip,
        wheel,
        install_args=[],
        resolve_dependencies=False,
    )


@pytest.mark.parametrize("tool_location", ["release_environment", "path"])
def test_typing_probe_resolves_uv_for_generated_hash_constraints(
    tmp_path, monkeypatch, tool_location,
):
    class ReachedTypingInstall(Exception):
        pass

    release_bin = tmp_path / "release/bin"
    path_bin = tmp_path / "path-bin"
    release_bin.mkdir(parents=True)
    path_bin.mkdir()
    tool_bin = release_bin if tool_location == "release_environment" else path_bin
    uv = tool_bin / "uv"
    uv.write_text(f"#!{support.sys.executable}\nprint('resolved uv')\n", encoding="utf-8")
    uv.chmod(0o755)
    run_subprocess = subprocess.run

    def run(command, **kwargs):
        if "mypy>=1.19,<2" in command:
            assert command[:3] == [str(uv), "pip", "install"]
            assert command[3:5] == ["--python", str(tmp_path / "venv/bin/python")]
            constraints = Path(command[command.index("-c") + 1])
            assert constraints.parent == support.PROJECT_ROOT / "requirements"
            assert "--hash=sha256:" in constraints.read_text()
            assert kwargs["check"] is True
            result = run_subprocess(command, **kwargs, capture_output=True)
            assert result.stdout == "resolved uv\n"
            raise ReachedTypingInstall
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(support.sys, "executable", str(release_bin / "python"))
    monkeypatch.setattr(support.subprocess, "run", run)
    with pytest.raises(ReachedTypingInstall):
        support._install_distribution_and_probe(
            artifact=tmp_path / "bo_forge.whl", probe_root=tmp_path,
            env={"PATH": str(path_bin)}, install_args=[], resolve_dependencies=True,
        )


def test_typing_probe_explains_missing_uv(tmp_path, monkeypatch):
    monkeypatch.setattr(support.sys, "executable", str(tmp_path / "release/bin/python"))
    monkeypatch.setattr(
        support.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    )
    with pytest.raises(RuntimeError, match="Install uv in the release environment"):
        support._install_distribution_and_probe(
            artifact=tmp_path / "bo_forge.whl", probe_root=tmp_path,
            env={"PATH": str(tmp_path)}, install_args=[], resolve_dependencies=True,
        )
