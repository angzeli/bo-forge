"""Dependency-resolved installation probe tests."""

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
