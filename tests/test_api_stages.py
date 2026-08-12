from __future__ import annotations

import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from filelock import FileLock

import bo_forge.logs as logs_module
from bo_forge.config import CampaignConfig
from bo_forge.errors import LogWriteError
from bo_forge.io import empty_campaign_log
from bo_forge_app.api import create_app
from bo_forge_app.service import CampaignAppService
from bo_forge_app.stages import InMemoryStageStore, StageStoreError

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _copy_campaign(root: Path) -> dict[str, str]:
    config_name = "01_simple_2d_maximise_logei.yaml"
    log_name = "01_simple_2d_maximise_logei_campaign_log.csv"
    (root / "configs").mkdir()
    (root / "examples").mkdir()
    shutil.copyfile(PROJECT_ROOT / "configs" / config_name, root / "configs" / config_name)
    shutil.copyfile(PROJECT_ROOT / "examples" / log_name, root / "examples" / log_name)
    return {
        "config_path": f"configs/{config_name}",
        "log_path": f"examples/{log_name}",
    }


def _dummy_store(
    *,
    clock=None,
    ttl_seconds: float = 1800,
    max_active_stages: int = 128,
) -> InMemoryStageStore:
    return InMemoryStageStore(
        ttl_seconds=ttl_seconds,
        max_active_stages=max_active_stages,
        clock=clock,
    )


def _create_dummy_stage(
    store: InMemoryStageStore,
    tmp_path: Path,
    *,
    reservation_token: str | None = None,
):
    suggestions = pd.DataFrame([{"row_id": "row_1", "x": 0.4}])
    quality = pd.DataFrame([{"metric": "distance", "value": 0.2}])
    bundle = {"suggestions": suggestions.copy(deep=True), "fingerprint": "abc"}
    return store.create(
        suggestions=suggestions,
        quality=quality,
        bundle=bundle,
        config_path=tmp_path / "campaign.yaml",
        log_path=tmp_path / "campaign.csv",
        stage_selection="screen",
        context_values={"temperature": 25.0},
        reservation_token=reservation_token,
    )


def test_stage_store_deep_copies_payload_and_metadata(tmp_path: Path) -> None:
    store = _dummy_store()
    staged = _create_dummy_stage(store, tmp_path)
    staged.suggestions.loc[0, "x"] = 0.9
    staged.quality.loc[0, "value"] = 9.0
    staged.bundle["fingerprint"] = "changed"
    assert staged.context_values is not None
    staged.context_values["temperature"] = 100.0

    recovered = store.get(staged.stage_id)

    assert float(recovered.suggestions.loc[0, "x"]) == pytest.approx(0.4)
    assert float(recovered.quality.loc[0, "value"]) == pytest.approx(0.2)
    assert recovered.bundle["fingerprint"] == "abc"
    assert recovered.stage_selection == "screen"
    assert recovered.context_values == {"temperature": 25.0}


def test_stage_store_expiration_capacity_discard_and_terminal_states(
    tmp_path: Path,
) -> None:
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    store = _dummy_store(clock=lambda: now[0], ttl_seconds=10, max_active_stages=1)
    first = _create_dummy_stage(store, tmp_path)

    with pytest.raises(StageStoreError, match="stage store is full") as capacity:
        _create_dummy_stage(store, tmp_path)
    assert capacity.value.code == "stage_capacity"

    now[0] += timedelta(seconds=11)
    with pytest.raises(StageStoreError) as expired:
        store.get(first.stage_id)
    assert expired.value.code == "stage_expired"
    assert expired.value.status_code == 410

    second = _create_dummy_stage(store, tmp_path)
    discarded = store.discard(second.stage_id)
    assert discarded.status == "discarded"
    with pytest.raises(StageStoreError) as terminal:
        store.get(second.stage_id)
    assert terminal.value.code == "stage_discarded"


def test_stage_store_claim_is_atomic(tmp_path: Path) -> None:
    store = _dummy_store()
    staged = _create_dummy_stage(store, tmp_path)
    barrier = threading.Barrier(2)

    def claim() -> str:
        barrier.wait(timeout=5)
        try:
            return store.claim(staged.stage_id).status
        except StageStoreError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = sorted(executor.map(lambda _index: claim(), range(2)))

    assert outcomes == ["appending", "stage_in_use"]


def test_stage_store_bounds_terminal_tombstones(tmp_path: Path) -> None:
    store = _dummy_store(max_active_stages=1)

    for _index in range(40):
        staged = _create_dummy_stage(store, tmp_path)
        store.discard(staged.stage_id)

    assert store.stats()["terminal_tombstones"] == 32


def test_stage_store_capacity_reservation_is_atomic_and_consumed(tmp_path: Path) -> None:
    store = _dummy_store(max_active_stages=1)
    reservation = store.reserve_capacity()
    assert store.stats()["reserved_stages"] == 1
    with pytest.raises(StageStoreError) as full:
        store.reserve_capacity()
    assert full.value.code == "stage_capacity"

    staged = _create_dummy_stage(store, tmp_path, reservation_token=reservation)

    assert staged.status == "active"
    assert store.stats()["reserved_stages"] == 0


@pytest.mark.parametrize("ttl", [float("nan"), float("inf"), 0, -1])
def test_stage_store_rejects_invalid_ttl(ttl: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        InMemoryStageStore(ttl_seconds=ttl)


@pytest.mark.parametrize("capacity", [True, 0, -1, 1.5])
def test_stage_store_rejects_invalid_capacity(capacity: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        InMemoryStageStore(max_active_stages=capacity)  # type: ignore[arg-type]


def test_api_server_stage_create_recover_discard_and_health(tmp_path: Path) -> None:
    ref = _copy_campaign(tmp_path)
    api_client = TestClient(create_app(tmp_path, stage_ttl_seconds=60, max_staged_batches=2))

    dry_run = api_client.post("/campaign/suggestions/dry-run", json={**ref, "batch_size": 1})
    assert dry_run.status_code == 200, dry_run.text
    payload = dry_run.json()
    stage_id = payload["stage"]["stage_id"]
    assert payload["stage"]["status"] == "active"
    assert payload["staged_bundle"]

    recovered = api_client.get(f"/campaign/stages/{stage_id}")
    assert recovered.status_code == 200
    assert recovered.json()["suggestions"] == payload["suggestions"]
    assert recovered.json()["quality"] == payload["quality"]

    health = api_client.get("/health").json()["staging"]
    assert health["active_stages"] == 1
    assert health["max_staged_batches"] == 2
    assert health["stage_ttl_seconds"] == pytest.approx(60)

    discarded = api_client.delete(f"/campaign/stages/{stage_id}")
    assert discarded.status_code == 200
    assert discarded.json()["stage"]["status"] == "discarded"
    terminal = api_client.get(f"/campaign/stages/{stage_id}")
    assert terminal.status_code == 409
    assert terminal.json()["error"]["code"] == "stage_discarded"


@pytest.mark.parametrize(
    ("config_name", "request_metadata"),
    [
        ("13_structured_campaign_core.yaml", {"stage": "refine"}),
        ("10_multi_objective_mixed_constrained_qlogehvi.yaml", {}),
        ("15_multi_fidelity_qmfkg.yaml", {}),
    ],
)
def test_server_stage_round_trip_across_campaign_kinds(
    tmp_path: Path,
    config_name: str,
    request_metadata: dict[str, object],
) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "examples").mkdir()
    config_path = tmp_path / "configs" / config_name
    shutil.copyfile(PROJECT_ROOT / "configs" / config_name, config_path)
    config = CampaignConfig.from_yaml(config_path)
    log_name = f"{config_path.stem}_campaign.csv"
    log_path = tmp_path / "examples" / log_name
    empty_campaign_log(config).to_csv(log_path, index=False)
    ref = {
        "config_path": f"configs/{config_name}",
        "log_path": f"examples/{log_name}",
    }
    api_client = TestClient(create_app(tmp_path))

    dry_run = api_client.post(
        "/campaign/suggestions/dry-run",
        json={**ref, "batch_size": 1, **request_metadata},
    )
    assert dry_run.status_code == 200, dry_run.text
    stage_id = dry_run.json()["stage"]["stage_id"]
    recovered = api_client.get(f"/campaign/stages/{stage_id}")
    assert recovered.status_code == 200
    assert recovered.json()["stage"]["stage_selection"] == request_metadata.get("stage")

    appended = api_client.post(f"/campaign/stages/{stage_id}/append")
    assert appended.status_code == 200, appended.text
    assert appended.json()["stage"]["status"] == "consumed"
    assert len(pd.read_csv(log_path, keep_default_na=False)) == 1


def test_api_stage_capacity_and_restart_are_process_local(tmp_path: Path) -> None:
    ref = _copy_campaign(tmp_path)
    first_app = create_app(tmp_path, max_staged_batches=1)
    api_client = TestClient(first_app)
    created = api_client.post("/campaign/suggestions/dry-run", json=ref)
    stage_id = created.json()["stage"]["stage_id"]

    full = api_client.post("/campaign/suggestions/dry-run", json=ref)
    assert full.status_code == 503
    assert full.json()["error"]["code"] == "stage_capacity"

    restarted = TestClient(create_app(tmp_path, max_staged_batches=1))
    missing = restarted.get(f"/campaign/stages/{stage_id}")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "stage_not_found"


def test_api_rejects_full_stage_store_before_suggestion_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = _copy_campaign(tmp_path)
    api_client = TestClient(create_app(tmp_path, max_staged_batches=1))
    assert api_client.post("/campaign/suggestions/dry-run", json=ref).status_code == 200
    monkeypatch.setattr(
        CampaignAppService,
        "suggest_dry_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("optimizer should not run")
        ),
    )

    full = api_client.post("/campaign/suggestions/dry-run", json=ref)

    assert full.status_code == 503
    assert full.json()["error"]["code"] == "stage_capacity"


def test_api_failed_dry_run_releases_capacity_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = _copy_campaign(tmp_path)
    api_client = TestClient(create_app(tmp_path, max_staged_batches=1))
    original_suggest = CampaignAppService.suggest_dry_run
    monkeypatch.setattr(
        CampaignAppService,
        "suggest_dry_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(LogWriteError("failed dry-run")),
    )

    failed = api_client.post("/campaign/suggestions/dry-run", json=ref)
    assert failed.status_code == 400
    assert api_client.get("/health").json()["staging"]["reserved_stages"] == 0

    monkeypatch.setattr(CampaignAppService, "suggest_dry_run", original_suggest)
    assert api_client.post("/campaign/suggestions/dry-run", json=ref).status_code == 200


def test_api_stage_expiration_returns_gone(tmp_path: Path) -> None:
    ref = _copy_campaign(tmp_path)
    api_client = TestClient(create_app(tmp_path, stage_ttl_seconds=0.01))
    stage_id = api_client.post("/campaign/suggestions/dry-run", json=ref).json()["stage"][
        "stage_id"
    ]
    time.sleep(0.02)

    expired = api_client.get(f"/campaign/stages/{stage_id}")

    assert expired.status_code == 410
    assert expired.json()["error"]["code"] == "stage_expired"


def test_two_server_stages_from_same_log_make_second_stale_without_mutation(
    tmp_path: Path,
) -> None:
    ref = _copy_campaign(tmp_path)
    log_path = tmp_path / ref["log_path"]
    api_client = TestClient(create_app(tmp_path))
    first = api_client.post("/campaign/suggestions/dry-run", json=ref).json()["stage"]
    second = api_client.post("/campaign/suggestions/dry-run", json=ref).json()["stage"]

    appended = api_client.post(f"/campaign/stages/{first['stage_id']}/append")
    assert appended.status_code == 200, appended.text
    assert appended.json()["stage"]["status"] == "consumed"
    after_first = log_path.read_bytes()

    stale = api_client.post(f"/campaign/stages/{second['stage_id']}/append")
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "stage_stale"
    assert log_path.read_bytes() == after_first

    consumed = api_client.get(f"/campaign/stages/{first['stage_id']}")
    assert consumed.status_code == 409
    assert consumed.json()["error"]["code"] == "stage_consumed"


def test_client_carried_append_retires_server_stage_and_releases_capacity(
    tmp_path: Path,
) -> None:
    ref = _copy_campaign(tmp_path)
    api_client = TestClient(create_app(tmp_path, max_staged_batches=1))
    dry_run = api_client.post("/campaign/suggestions/dry-run", json=ref).json()
    stage_id = dry_run["stage"]["stage_id"]

    append = api_client.post(
        "/campaign/suggestions/append",
        json={**ref, "staged_bundle": dry_run["staged_bundle"]},
    )
    assert append.status_code == 200, append.text
    terminal = api_client.get(f"/campaign/stages/{stage_id}")
    assert terminal.status_code == 409
    assert terminal.json()["error"]["code"] == "stage_consumed"

    row_id = dry_run["suggestions"]["records"][0]["row_id"]
    observed = api_client.post(
        "/campaign/observations",
        json={
            **ref,
            "row_id": row_id,
            "objective_value": 1.0,
            "expected_log_fingerprint": append.json()["log_fingerprint"],
        },
    )
    assert observed.status_code == 200, observed.text
    assert api_client.post("/campaign/suggestions/dry-run", json=ref).status_code == 200


def test_concurrent_server_stage_append_writes_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = _copy_campaign(tmp_path)
    log_path = tmp_path / ref["log_path"]
    before_rows = len(pd.read_csv(log_path, keep_default_na=False))
    api_client = TestClient(create_app(tmp_path))
    stage_id = api_client.post("/campaign/suggestions/dry-run", json=ref).json()["stage"][
        "stage_id"
    ]
    original_append = CampaignAppService.append_staged
    entered = threading.Event()
    release = threading.Event()

    def delayed_append(self, *args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return original_append(self, *args, **kwargs)

    monkeypatch.setattr(CampaignAppService, "append_staged", delayed_append)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            api_client.post,
            f"/campaign/stages/{stage_id}/append",
        )
        assert entered.wait(timeout=5)
        second = api_client.post(f"/campaign/stages/{stage_id}/append")
        release.set()
        first_response = first.result(timeout=10)

    assert first_response.status_code == 200
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "stage_in_use"
    assert len(pd.read_csv(log_path, keep_default_na=False)) == before_rows + 1


def test_server_stage_log_busy_error_restores_active_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = _copy_campaign(tmp_path)
    log_path = tmp_path / ref["log_path"]
    before = log_path.read_bytes()
    api_client = TestClient(create_app(tmp_path))
    stage_id = api_client.post("/campaign/suggestions/dry-run", json=ref).json()["stage"][
        "stage_id"
    ]
    monkeypatch.setattr(logs_module, "LOG_LOCK_TIMEOUT_SECONDS", 0.01)

    with FileLock(logs_module._log_lock_path(log_path)):
        response = api_client.post(f"/campaign/stages/{stage_id}/append")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "log_busy"
    assert log_path.read_bytes() == before
    assert api_client.get(f"/campaign/stages/{stage_id}").json()["stage"]["status"] == "active"


def test_server_stage_changed_config_becomes_stale_without_log_mutation(
    tmp_path: Path,
) -> None:
    ref = _copy_campaign(tmp_path)
    config_path = tmp_path / ref["config_path"]
    log_path = tmp_path / ref["log_path"]
    api_client = TestClient(create_app(tmp_path))
    stage_id = api_client.post("/campaign/suggestions/dry-run", json=ref).json()["stage"][
        "stage_id"
    ]
    config_path.write_text("invalid: [\n", encoding="utf-8")
    before = log_path.read_bytes()

    response = api_client.post(f"/campaign/stages/{stage_id}/append")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "stage_stale"
    assert "Config file changed" in response.json()["error"]["message"]
    assert log_path.read_bytes() == before


def test_server_stage_unexpected_failure_restores_active_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = _copy_campaign(tmp_path)
    app = create_app(tmp_path)
    api_client = TestClient(app, raise_server_exceptions=False)
    stage_id = api_client.post("/campaign/suggestions/dry-run", json=ref).json()["stage"][
        "stage_id"
    ]
    monkeypatch.setattr(
        CampaignAppService,
        "append_staged",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unexpected")),
    )

    response = api_client.post(f"/campaign/stages/{stage_id}/append")

    assert response.status_code == 500
    assert api_client.get(f"/campaign/stages/{stage_id}").json()["stage"]["status"] == "active"


def test_server_stage_write_then_failure_becomes_stale_not_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = _copy_campaign(tmp_path)
    log_path = tmp_path / ref["log_path"]
    api_client = TestClient(create_app(tmp_path, max_staged_batches=1))
    stage_id = api_client.post("/campaign/suggestions/dry-run", json=ref).json()["stage"][
        "stage_id"
    ]
    before_rows = len(pd.read_csv(log_path, keep_default_na=False))
    original_append = CampaignAppService.append_staged

    def write_then_fail(self, *args, **kwargs):
        original_append(self, *args, **kwargs)
        raise LogWriteError("simulated post-write validation failure")

    monkeypatch.setattr(CampaignAppService, "append_staged", write_then_fail)

    response = api_client.post(f"/campaign/stages/{stage_id}/append")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "stage_stale"
    assert len(pd.read_csv(log_path, keep_default_na=False)) == before_rows + 1
    terminal = api_client.get(f"/campaign/stages/{stage_id}")
    assert terminal.status_code == 409
    assert terminal.json()["error"]["code"] == "stage_stale"


def test_get_stale_stage_releases_capacity_after_external_log_change(
    tmp_path: Path,
) -> None:
    ref = _copy_campaign(tmp_path)
    log_path = tmp_path / ref["log_path"]
    api_client = TestClient(create_app(tmp_path, max_staged_batches=1))
    stage_id = api_client.post("/campaign/suggestions/dry-run", json=ref).json()["stage"][
        "stage_id"
    ]
    log_path.write_bytes(log_path.read_bytes() + b"\n")

    stale = api_client.get(f"/campaign/stages/{stage_id}")

    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "stage_stale"
    replacement = api_client.post("/campaign/suggestions/dry-run", json=ref)
    assert replacement.status_code == 200, replacement.text
