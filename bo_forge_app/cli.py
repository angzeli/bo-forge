"""Console entrypoint for launching the packaged BO Forge Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path


def packaged_streamlit_app_path() -> Path:
    """Return the installed Streamlit app script path."""
    from bo_forge_app import streamlit_app

    return Path(streamlit_app.__file__).resolve()


def main() -> None:
    """Launch the local Streamlit workbench."""
    try:
        from streamlit.web import cli as streamlit_cli
    except ModuleNotFoundError as exc:
        if exc.name == "streamlit":
            print(
                "Error: Streamlit is not installed. "
                'Install the app extra with: pip install "bo-forge[app]"',
                file=sys.stderr,
            )
            raise SystemExit(1) from None
        raise

    app_path = packaged_streamlit_app_path()
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    streamlit_cli.main()
