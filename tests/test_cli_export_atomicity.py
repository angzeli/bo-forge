"""Suggestion artifact publication must not truncate an earlier complete export."""

import stat
import zipfile
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from bo_forge._cli.campaign import _write_csv
from bo_forge._cli.errors import _CLIOutputError
from bo_forge.cli import run
from bo_forge.session import CampaignSession
from tests._cli_support import write_config


@pytest.fixture
def campaign(tmp_path):
    config = write_config(tmp_path / "campaign.yaml")
    session = CampaignSession.initialize(config, tmp_path / "campaign.csv")
    return session, session.suggest_next()


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["write", "publish"])
def test_failed_export_preserves_destination_and_all_campaign_files(
    tmp_path, monkeypatch, campaign, existing, failure,
):
    session, suggestions = campaign
    output = tmp_path / "preview.csv"
    if existing:
        output.write_bytes(b"previous complete artifact\n")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    def partial_write(self, path, **kwargs):
        Path(path).write_bytes(b"row_id,status\npartial")
        raise OSError("controlled incomplete export")

    def failed_publish(self, target):
        raise OSError("controlled publication failure")

    if failure == "write":
        monkeypatch.setattr(pd.DataFrame, "to_csv", partial_write)
    else:
        monkeypatch.setattr(Path, "replace", failed_publish)
    with pytest.raises(_CLIOutputError, match="Could not write suggestions CSV"):
        _write_csv(suggestions, output, session)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_partial_cli_export_does_not_append(tmp_path, monkeypatch, capsys, campaign):
    session, suggestions = campaign
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    calls = []

    def generate(self, **kwargs):
        calls.append("generate")
        return suggestions

    def partial_write(self, path, **kwargs):
        calls.append("export")
        Path(path).write_text("partial", encoding="utf-8")
        raise OSError("disk full")

    monkeypatch.setattr(CampaignSession, "suggest_next", generate)
    monkeypatch.setattr(CampaignSession, "append_suggestions",
                        lambda *a, **kw: pytest.fail("append after failed export"))
    monkeypatch.setattr(pd.DataFrame, "to_csv", partial_write)
    assert run(["suggest", "--config", str(session.config_path), "--log", str(session.log_path),
                "--output", str(tmp_path / "preview.csv"), "--append"]) == 1
    assert "Could not write suggestions CSV" in capsys.readouterr().err
    assert calls == ["generate", "export"]
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("suffix", [".csv", ".csv.gz", ".zip"])
@pytest.mark.parametrize("symlink", [False, True])
def test_successful_export_preserves_compression_modes_and_symlinks(
    tmp_path, campaign, suffix, symlink,
):
    session, suggestions = campaign
    target = tmp_path / ("artifact" + suffix)
    target.write_bytes(b"old export")
    target.chmod(0o640)
    output = tmp_path / ("preview" + suffix) if symlink else target
    if symlink:
        output.symlink_to(target.name)
    sources = {p: p.read_bytes() for p in tmp_path.iterdir() if p not in {target, output}}
    assert _write_csv(suggestions, output, session) == output
    assert output.is_symlink() == symlink
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    pd.testing.assert_frame_equal(pd.read_csv(output),
                                  pd.read_csv(StringIO(suggestions.to_csv(index=False))))
    if suffix == ".zip":
        with zipfile.ZipFile(output) as archive:
            assert archive.namelist() == [output.stem]
    assert all(p.read_bytes() == data for p, data in sources.items())
    assert not list(tmp_path.glob(".suggestions-*"))
