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
from bo_forge_app.api import _error_recovery, create_app
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

    stats = store.stats()
    assert stats["terminal_tombstones"] == 32
    assert stats["terminal_status_counts"]["discarded"] == 32
    assert stats["lifecycle_totals"]["discarded"] == 40


def test_stage_store_terminal_history_is_metadata_only(tmp_path: Path) -> None:
    store = _dummy_store()
    staged = _create_dummy_stage(store, tmp_path)

    store.discard(staged.stage_id)

    tombstone = store._terminal[staged.stage_id]
    assert tombstone.suggestion_count == 1
    assert tombstone.context_variable_names == ("temperature",)
    for heavy_attribute in ("suggestions", "quality", "bundle", "context_values"):
        assert not hasattr(tombstone, heavy_attribute)


def test_stage_store_listing_is_filtered_bounded_and_deterministic(
    tmp_path: Path,
) -> None:
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    store = _dummy_store(clock=lambda: now[0], max_active_stages=4)
    first = _create_dummy_stage(store, tmp_path)
    now[0] += timedelta(seconds=1)
    second = _create_dummy_stage(store, tmp_path)
    now[0] += timedelta(seconds=1)
    store.discard(first.stage_id)
    now[0] += timedelta(seconds=1)
    third = _create_dummy_stage(store, tmp_path)

    listing = store.list_summaries(include_terminal=True, limit=2)

    assert [item.stage_id for item in listing.stages] == [third.stage_id, first.stage_id]
    assert listing.total == 3
    assert listing.returned == 2
    assert listing.truncated is True
    assert listing.status_counts == {"active": 2, "discarded": 1}
    filtered = store.list_summaries(
        include_terminal=True,
        statuses=["discarded"],
    )
    assert [item.stage_id for item in filtered.stages] == [first.stage_id]
    assert second.stage_id not in [item.stage_id for item in filtered.stages]


def test_stage_store_listing_uses_stage_id_to_break_timestamp_ties(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    store = _dummy_store(clock=lambda: now, max_active_stages=2)
    first = _create_dummy_stage(store, tmp_path)
    second = _create_dummy_stage(store, tmp_path)

    listing = store.list_summaries()

    assert [item.stage_id for item in listing.stages] == sorted(
        [first.stage_id, second.stage_id]
    )


def test_stage_store_listing_validation_does_not_hold_store_lock(
    tmp_path: Path,
) -> None:
    store = _dummy_store()
    _create_dummy_stage(store, tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def blocked_validator(_snapshot) -> None:
        entered.set()
        assert release.wait(timeout=5)
        return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        listing_future = executor.submit(
            store.list_summaries,
            validator=blocked_validator,
        )
        assert entered.wait(timeout=5)
        stats_future = executor.submit(store.stats)
        try:
            assert stats_future.result(timeout=1)["active_stages"] == 1
        finally:
            release.set()
        assert listing_future.result(timeout=5).total == 1


def test_stage_store_renewal_resets_expiry_without_changing_payload(
    tmp_path: Path,
) -> None:
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    store = _dummy_store(clock=lambda: now[0], ttl_seconds=10, max_active_stages=1)
    staged = _create_dummy_stage(store, tmp_path)
    original_suggestions = staged.suggestions.copy(deep=True)
    original_quality = staged.quality.copy(deep=True)
    original_bundle = staged.bundle.copy()
    now[0] += timedelta(seconds=4)

    renewed = store.renew(staged.stage_id)

    assert renewed.created_at == staged.created_at
    assert renewed.expires_at == now[0] + timedelta(seconds=10)
    assert renewed.last_transition_at == now[0]
    assert renewed.renewal_count == 1
    assert renewed.status_reason == "Stage expiry renewed."
    pd.testing.assert_frame_equal(renewed.suggestions, original_suggestions)
    pd.testing.assert_frame_equal(renewed.quality, original_quality)
    assert renewed.bundle["fingerprint"] == original_bundle["fingerprint"]
    pd.testing.assert_frame_equal(
        renewed.bundle["suggestions"],
        original_bundle["suggestions"],
    )
    assert store.stats()["capacity_remaining"] == 0

    now[0] += timedelta(seconds=2)
    renewed_again = store.renew(staged.stage_id)
    assert renewed_again.renewal_count == 2
    assert renewed_again.expires_at == now[0] + timedelta(seconds=10)


def test_stage_store_renewal_validates_without_lock_and_starts_ttl_after_validation(
    tmp_path: Path,
) -> None:
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    store = _dummy_store(clock=lambda: now[0], ttl_seconds=10)
    staged = _create_dummy_stage(store, tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def blocked_validator(_snapshot) -> None:
        entered.set()
        assert release.wait(timeout=5)
        now[0] += timedelta(seconds=4)
        return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        renewal_future = executor.submit(
            store.renew,
            staged.stage_id,
            validator=blocked_validator,
        )
        assert entered.wait(timeout=5)
        stats_future = executor.submit(store.stats)
        try:
            assert stats_future.result(timeout=1)["active_stages"] == 1
        finally:
            release.set()
        renewed = renewal_future.result(timeout=5)

    assert renewed.last_transition_at == now[0]
    assert renewed.expires_at == now[0] + timedelta(seconds=10)


def test_stage_store_renewal_rejects_non_active_states(tmp_path: Path) -> None:
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    store = _dummy_store(clock=lambda: now[0], ttl_seconds=1, max_active_stages=5)

    appending = _create_dummy_stage(store, tmp_path)
    store.claim(appending.stage_id)
    with pytest.raises(StageStoreError) as in_use:
        store.renew(appending.stage_id)
    assert in_use.value.code == "stage_in_use"

    consumed = _create_dummy_stage(store, tmp_path)
    store.claim(consumed.stage_id)
    store.complete(consumed.stage_id, "consumed")
    discarded = _create_dummy_stage(store, tmp_path)
    store.discard(discarded.stage_id)
    stale = _create_dummy_stage(store, tmp_path)
    store.mark_stale_if_active(stale.stage_id)
    expired = _create_dummy_stage(store, tmp_path)
    now[0] += timedelta(seconds=2)

    expected_codes = {
        consumed.stage_id: "stage_consumed",
        discarded.stage_id: "stage_discarded",
        stale.stage_id: "stage_stale",
        expired.stage_id: "stage_expired",
    }
    for stage_id, expected_code in expected_codes.items():
        with pytest.raises(StageStoreError) as error:
            store.renew(stage_id)
        assert error.value.code == expected_code


def test_stage_store_health_tracks_ages_capacity_and_lifecycle_totals(
    tmp_path: Path,
) -> None:
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    store = _dummy_store(clock=lambda: now[0], max_active_stages=2)
    active = _create_dummy_stage(store, tmp_path)
    now[0] += timedelta(seconds=2)
    appending = _create_dummy_stage(store, tmp_path)
    store.claim(appending.stage_id)
    now[0] += timedelta(seconds=3)

    stats = store.stats()

    assert stats["capacity_remaining"] == 0
    assert stats["oldest_active_age_seconds"] == pytest.approx(5)
    assert stats["oldest_appending_age_seconds"] == pytest.approx(3)
    assert stats["lifecycle_totals"]["created"] == 2
    assert stats["lifecycle_totals"]["claimed"] == 1
    store.discard(active.stage_id)
    assert store.stats()["terminal_status_counts"]["discarded"] == 1


def test_stage_store_capacity_rejections_are_counted(tmp_path: Path) -> None:
    store = _dummy_store(max_active_stages=1)
    _create_dummy_stage(store, tmp_path)

    with pytest.raises(StageStoreError):
        store.reserve_capacity()

    assert store.stats()["lifecycle_totals"]["capacity_rejected"] == 1


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


def test_api_stage_listing_exposes_metadata_only_and_terminal_filters(
    tmp_path: Path,
) -> None:
    ref = _copy_campaign(tmp_path)
    api_client = TestClient(create_app(tmp_path, max_staged_batches=3))
    first = api_client.post("/campaign/suggestions/dry-run", json=ref).json()["stage"]
    second = api_client.post("/campaign/suggestions/dry-run", json=ref).json()["stage"]
    api_client.delete(f"/campaign/stages/{first['stage_id']}")

    active_only = api_client.get("/campaign/stages")
    assert active_only.status_code == 200
    assert active_only.json()["total"] == 1
    assert active_only.json()["stages"][0]["stage_id"] == second["stage_id"]

    response = api_client.get(
        "/campaign/stages",
        params=[("include_terminal", "true"), ("limit", "1")],
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] == 2
    assert payload["returned"] == 1
    assert payload["truncated"] is True
    assert payload["status_counts"] == {"active": 1, "discarded": 1}
    forbidden = {
        "suggestions",
        "quality",
        "bundle",
        "config_fingerprint",
        "log_fingerprint",
        "context_values",
    }
    assert forbidden.isdisjoint(payload["stages"][0])
    assert set(payload["stages"][0]) == {
        "stage_id",
        "status",
        "created_at",
        "expires_at",
        "last_transition_at",
        "remaining_ttl_seconds",
        "suggestion_count",
        "config_path",
        "log_path",
        "stage_selection",
        "context_variable_names",
        "renewal_count",
        "status_reason",
    }

    discarded = api_client.get(
        "/campaign/stages",
        params=[("include_terminal", "true"), ("status", "discarded")],
    ).json()
    assert discarded["total"] == 1
    assert discarded["stages"][0]["stage_id"] == first["stage_id"]


def test_api_stage_listing_validates_status_and_limit(tmp_path: Path) -> None:
    api_client = TestClient(create_app(tmp_path))

    unknown = api_client.get("/campaign/stages", params={"status": "unknown"})
    too_large = api_client.get("/campaign/stages", params={"limit": 201})

    assert unknown.status_code == 422
    assert unknown.json()["error"]["code"] == "request_validation"
    assert too_large.status_code == 422
    assert too_large.json()["error"]["code"] == "request_validation"


def test_api_stage_renewal_is_explicit_and_preserves_payload(tmp_path: Path) -> None:
    ref = _copy_campaign(tmp_path)
    log_path = tmp_path / ref["log_path"]
    before = log_path.read_bytes()
    api_client = TestClient(create_app(tmp_path, stage_ttl_seconds=60))
    created = api_client.post("/campaign/suggestions/dry-run", json=ref).json()
    stage_id = created["stage"]["stage_id"]
    original = api_client.get(f"/campaign/stages/{stage_id}").json()

    renewed = api_client.post(f"/campaign/stages/{stage_id}/renew")

    assert renewed.status_code == 200, renewed.text
    assert renewed.json()["stage"]["renewal_count"] == 1
    assert renewed.json()["stage"]["created_at"] == created["stage"]["created_at"]
    assert renewed.json()["stage"]["expires_at"] > created["stage"]["expires_at"]
    recovered = api_client.get(f"/campaign/stages/{stage_id}").json()
    assert recovered["suggestions"] == original["suggestions"]
    assert recovered["quality"] == original["quality"]
    assert log_path.read_bytes() == before


def test_api_stage_listing_and_renewal_mark_changed_files_stale_without_mutation(
    tmp_path: Path,
) -> None:
    ref = _copy_campaign(tmp_path)
    log_path = tmp_path / ref["log_path"]
    renewal_config = tmp_path / "configs" / "renewal.yaml"
    renewal_log = tmp_path / "examples" / "renewal.csv"
    shutil.copyfile(tmp_path / ref["config_path"], renewal_config)
    shutil.copyfile(log_path, renewal_log)
    renewal_ref = {
        "config_path": "configs/renewal.yaml",
        "log_path": "examples/renewal.csv",
    }
    api_client = TestClient(create_app(tmp_path, max_staged_batches=2))
    listed_stage = api_client.post("/campaign/suggestions/dry-run", json=ref).json()[
        "stage"
    ]
    renewed_stage = api_client.post(
        "/campaign/suggestions/dry-run", json=renewal_ref
    ).json()["stage"]
    log_path.write_bytes(log_path.read_bytes() + b"\n")
    before = log_path.read_bytes()

    listing = api_client.get("/campaign/stages", params={"include_terminal": "true"})
    renewal_log.write_bytes(renewal_log.read_bytes() + b"\n")
    renewal_before = renewal_log.read_bytes()
    renew = api_client.post(f"/campaign/stages/{renewed_stage['stage_id']}/renew")

    assert listing.status_code == 200
    listed = {item["stage_id"]: item for item in listing.json()["stages"]}
    assert listed[listed_stage["stage_id"]]["status"] == "stale"
    assert listed[renewed_stage["stage_id"]]["status"] == "active"
    assert renew.status_code == 409
    assert renew.json()["error"]["code"] == "stage_stale"
    assert log_path.read_bytes() == before
    assert renewal_log.read_bytes() == renewal_before


def test_api_health_is_cheap_and_does_not_hash_campaign_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(tmp_path)
    monkeypatch.setattr(
        "bo_forge_app.api.file_fingerprint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("health must not hash files")
        ),
    )

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    staging = response.json()["staging"]
    assert staging["capacity_remaining"] == 128
    assert staging["oldest_active_age_seconds"] is None
    assert staging["oldest_appending_age_seconds"] is None
    assert staging["lifecycle_totals"]["created"] == 0


def test_api_health_lifecycle_counters_reset_with_process_store(tmp_path: Path) -> None:
    ref = _copy_campaign(tmp_path)
    first_client = TestClient(create_app(tmp_path))
    created = first_client.post("/campaign/suggestions/dry-run", json=ref).json()
    first_client.delete(f"/campaign/stages/{created['stage']['stage_id']}")
    first_health = first_client.get("/health").json()["staging"]
    restarted_health = TestClient(create_app(tmp_path)).get("/health").json()["staging"]

    assert first_health["lifecycle_totals"]["created"] == 1
    assert first_health["lifecycle_totals"]["discarded"] == 1
    assert restarted_health["lifecycle_totals"]["created"] == 0
    assert restarted_health["terminal_tombstones"] == 0


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


def test_server_stage_metadata_uses_safely_inferred_single_stage(
    tmp_path: Path,
) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "examples").mkdir()
    config_path = tmp_path / "configs" / "single_stage.yaml"
    config_path.write_text(
        """campaign_name: single_stage
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0.0
    upper: 1.0
stages:
  - name: screen
    variables: [x]
bo:
  batch_size: 1
  initial_design_size: 2
  acquisition: log_ei
""",
        encoding="utf-8",
    )
    config = CampaignConfig.from_yaml(config_path)
    log_path = tmp_path / "examples" / "single_stage.csv"
    empty_campaign_log(config).to_csv(log_path, index=False)
    api_client = TestClient(create_app(tmp_path))

    response = api_client.post(
        "/campaign/suggestions/dry-run",
        json={
            "config_path": "configs/single_stage.yaml",
            "log_path": "examples/single_stage.csv",
            "batch_size": 1,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["suggestions"]["records"][0]["stage"] == "screen"
    assert payload["stage"]["stage_selection"] == "screen"
    listing = api_client.get("/campaign/stages").json()["stages"]
    listed = next(
        item for item in listing if item["stage_id"] == payload["stage"]["stage_id"]
    )
    assert listed["stage_selection"] == "screen"


def test_server_stage_resolves_all_context_metadata_but_preserves_partial_append(
    tmp_path: Path,
) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "examples").mkdir()
    config_path = tmp_path / "configs" / "partial_context.yaml"
    config_path.write_text(
        """campaign_name: partial_context
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0.0
    upper: 1.0
  - name: temperature
    type: continuous
    lower: 0.0
    upper: 1.0
  - name: pressure
    type: continuous
    lower: 0.0
    upper: 1.0
context:
  variables: [temperature, pressure]
  default_values:
    temperature: 0.5
    pressure: 0.25
bo:
  batch_size: 1
  initial_design_size: 2
  acquisition: log_ei
""",
        encoding="utf-8",
    )
    config = CampaignConfig.from_yaml(config_path)
    log_path = tmp_path / "examples" / "partial_context.csv"
    empty_campaign_log(config).to_csv(log_path, index=False)
    api_client = TestClient(create_app(tmp_path))

    response = api_client.post(
        "/campaign/suggestions/dry-run",
        json={
            "config_path": "configs/partial_context.yaml",
            "log_path": "examples/partial_context.csv",
            "context_values": {"temperature": 0.75},
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["stage"]["context_values"] == {
        "temperature": 0.75,
        "pressure": 0.25,
    }
    listed = api_client.get("/campaign/stages").json()["stages"][0]
    assert listed["context_variable_names"] == ["pressure", "temperature"]
    appended = api_client.post(
        f"/campaign/stages/{payload['stage']['stage_id']}/append"
    )
    assert appended.status_code == 200, appended.text
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


def test_server_stages_only_rejects_client_bundle_without_mutating_log(
    tmp_path: Path,
) -> None:
    ref = _copy_campaign(tmp_path)
    log_path = tmp_path / ref["log_path"]
    api_client = TestClient(create_app(tmp_path, server_stages_only=True))
    dry_run = api_client.post("/campaign/suggestions/dry-run", json=ref).json()
    before = log_path.read_bytes()

    response = api_client.post(
        "/campaign/suggestions/append",
        json={**ref, "staged_bundle": dry_run["staged_bundle"]},
    )

    assert response.status_code == 403
    assert response.json()["error"] == {
        "code": "client_bundle_append_disabled",
        "message": (
            "Client-carried staged-bundle append is disabled for this deployment. "
            "Use a server-managed stage append instead."
        ),
        "retryable": False,
        "suggested_action": (
            "Generate a dry-run and append its server-managed stage ID instead."
        ),
    }
    assert log_path.read_bytes() == before


def test_server_stages_only_keeps_server_managed_append_operational(
    tmp_path: Path,
) -> None:
    ref = _copy_campaign(tmp_path)
    log_path = tmp_path / ref["log_path"]
    before_rows = len(pd.read_csv(log_path, keep_default_na=False))
    api_client = TestClient(create_app(tmp_path, server_stages_only=True))
    stage_id = api_client.post("/campaign/suggestions/dry-run", json=ref).json()["stage"][
        "stage_id"
    ]

    response = api_client.post(f"/campaign/stages/{stage_id}/append")

    assert response.status_code == 200, response.text
    assert response.json()["stage"]["status"] == "consumed"
    assert len(pd.read_csv(log_path, keep_default_na=False)) == before_rows + 1


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
        renew = api_client.post(f"/campaign/stages/{stage_id}/renew")
        discard = api_client.delete(f"/campaign/stages/{stage_id}")
        release.set()
        first_response = first.result(timeout=10)

    assert first_response.status_code == 200
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "stage_in_use"
    assert renew.status_code == 409
    assert renew.json()["error"]["code"] == "stage_in_use"
    assert discard.status_code == 409
    assert discard.json()["error"]["code"] == "stage_in_use"
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
    error = response.json()["error"]
    assert error["code"] == "log_busy"
    assert error["retryable"] is True
    assert "Wait briefly" in error["suggested_action"]
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


def test_server_stage_terminal_reason_does_not_retain_exception_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = _copy_campaign(tmp_path)
    api_client = TestClient(create_app(tmp_path))
    stage_id = api_client.post("/campaign/suggestions/dry-run", json=ref).json()["stage"][
        "stage_id"
    ]
    secret_detail = "/Users/private/internal/campaign.csv"
    monkeypatch.setattr(
        CampaignAppService,
        "append_staged",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError(secret_detail)),
    )

    response = api_client.post(f"/campaign/stages/{stage_id}/append")
    listing = api_client.get(
        "/campaign/stages",
        params={"include_terminal": "true"},
    ).json()["stages"]
    terminal = next(item for item in listing if item["stage_id"] == stage_id)

    assert response.status_code == 409
    assert secret_detail not in response.text
    assert terminal["status_reason"] == (
        "Staged batch failed append integrity checks and cannot be retried."
    )
    assert secret_detail not in terminal["status_reason"]


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


@pytest.mark.parametrize(
    ("code", "retryable", "action_text"),
    [
        ("stage_in_use", True, "Wait"),
        ("stage_capacity", True, "Append or discard"),
        ("log_busy", True, "Wait briefly"),
        ("stage_expired", False, "new dry-run"),
        ("stage_consumed", False, "Refresh"),
        ("stage_discarded", False, "new dry-run"),
        ("stage_stale", False, "new dry-run"),
        ("stage_not_found", False, "stage ID"),
        ("stale_log", False, "new log fingerprint"),
        ("path_outside_root", False, "inside"),
        ("client_bundle_append_disabled", False, "server-managed stage ID"),
        ("request_validation", False, "Correct"),
        ("bo_forge_error", False, "Correct"),
    ],
)
def test_api_error_recovery_guidance_is_machine_readable(
    code: str,
    retryable: bool,
    action_text: str,
) -> None:
    actual_retryable, suggested_action = _error_recovery(code)

    assert actual_retryable is retryable
    assert action_text in suggested_action


def test_api_stage_errors_include_recovery_fields(tmp_path: Path) -> None:
    response = TestClient(create_app(tmp_path)).get("/campaign/stages/missing")

    assert response.status_code == 404
    assert response.json()["error"] == {
        "code": "stage_not_found",
        "message": "Staged batch was not found.",
        "retryable": False,
        "suggested_action": (
            "Check the stage ID or generate a new dry-run in this API process."
        ),
    }
