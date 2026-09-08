"""Shared temporary-campaign builders for provenance lifecycle tests."""

from bo_forge import CampaignSession
from bo_forge._campaign import provenance as io
from tests._session_support import config, write_config, write_log


def legacy(tmp_path):
    cfg = write_config(tmp_path / "campaign.yaml")
    log = write_log(tmp_path / "campaign.csv", config())
    return cfg, log


def _downgrade(log):
    manifest = io.load_manifest(log)
    manifest["schema_version"] = 1
    manifest.pop("origin")
    manifest.pop("archives")
    io._write_json_atomic(io.manifest_path_for_log(log), manifest)


def v1(tmp_path):
    cfg = write_config(tmp_path / "campaign.yaml")
    session = CampaignSession.initialize(cfg, tmp_path / "campaign.csv")
    _downgrade(session.log_path)
    return cfg, session.log_path


def apply(helper, *args, **kwargs):
    preview = helper(*args, **kwargs)
    return helper(
        *args,
        **kwargs,
        apply=True,
        reason="Reviewed campaign history",
        expected_identities=preview["expected_identities"],
    )


def snapshot(cfg, log):
    path = io.manifest_path_for_log(log)
    return cfg.read_bytes(), log.read_bytes(), path.read_bytes() if path.exists() else None
