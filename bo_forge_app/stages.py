"""Thread-safe in-memory staging for the experimental API probe."""

from __future__ import annotations

import copy
import math
import secrets
import threading
from collections import Counter, OrderedDict
from collections.abc import Callable, Collection
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

STAGE_STATUSES = (
    "active",
    "appending",
    "consumed",
    "discarded",
    "stale",
    "expired",
)


class StageStoreError(ValueError):
    """Structured stage lifecycle error exposed by the API adapter."""

    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass
class StageSnapshot:
    """Deep-copied active stage state safe to return outside the store lock."""

    stage_id: str
    status: str
    created_at: datetime
    expires_at: datetime
    last_transition_at: datetime
    renewal_count: int
    status_reason: str | None
    suggestions: pd.DataFrame
    quality: pd.DataFrame
    bundle: dict[str, object]
    config_path: Path
    log_path: Path
    stage_selection: str | None
    context_values: dict[str, object] | None


@dataclass(frozen=True)
class StageSummary:
    """Metadata-only stage lifecycle summary."""

    stage_id: str
    status: str
    created_at: datetime
    expires_at: datetime
    last_transition_at: datetime
    remaining_ttl_seconds: float
    suggestion_count: int
    config_path: Path
    log_path: Path
    stage_selection: str | None
    context_variable_names: tuple[str, ...]
    renewal_count: int
    status_reason: str | None


@dataclass(frozen=True)
class StageListing:
    """One bounded, deterministic lifecycle listing."""

    stages: tuple[StageSummary, ...]
    total: int
    returned: int
    truncated: bool
    status_counts: dict[str, int]


@dataclass(frozen=True)
class StageValidationSnapshot:
    """Lightweight immutable state used for file validation outside the lock."""

    stage_id: str
    config_path: Path
    log_path: Path
    config_fingerprint: str
    log_fingerprint: str


@dataclass
class _StageRecord:
    stage_id: str
    status: str
    created_at: datetime
    expires_at: datetime
    last_transition_at: datetime
    renewal_count: int
    status_reason: str | None
    suggestions: pd.DataFrame
    quality: pd.DataFrame
    bundle: dict[str, object]
    config_path: Path
    log_path: Path
    stage_selection: str | None
    context_values: dict[str, object] | None


@dataclass(frozen=True)
class _StageTombstone:
    stage_id: str
    status: str
    created_at: datetime
    expires_at: datetime
    last_transition_at: datetime
    renewal_count: int
    status_reason: str | None
    suggestion_count: int
    config_path: Path
    log_path: Path
    stage_selection: str | None
    context_variable_names: tuple[str, ...]


StageValidator = Callable[[StageValidationSnapshot], str | None]


class InMemoryStageStore:
    """Bounded process-local storage for exact API suggestion batches."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 1800.0,
        max_active_stages: int = 128,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        try:
            parsed_ttl = float(ttl_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("stage TTL must be a finite positive number") from exc
        if not math.isfinite(parsed_ttl) or parsed_ttl <= 0:
            raise ValueError("stage TTL must be a finite positive number")
        if (
            isinstance(max_active_stages, bool)
            or not isinstance(max_active_stages, int)
            or max_active_stages <= 0
        ):
            raise ValueError("maximum staged batches must be a positive integer")
        self.ttl_seconds = parsed_ttl
        self.max_active_stages = max_active_stages
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.RLock()
        self._active: dict[str, _StageRecord] = {}
        self._reservations: set[str] = set()
        self._terminal: OrderedDict[str, _StageTombstone] = OrderedDict()
        self._max_tombstones = max(32, self.max_active_stages * 2)
        self._lifecycle_totals = {
            name: 0
            for name in (
                "created",
                "claimed",
                "restored",
                "renewed",
                "consumed",
                "discarded",
                "stale",
                "expired",
                "capacity_rejected",
            )
        }

    def create(
        self,
        *,
        suggestions: pd.DataFrame,
        quality: pd.DataFrame,
        bundle: dict[str, object],
        config_path: str | Path,
        log_path: str | Path,
        stage_selection: str | None = None,
        context_values: dict[str, object] | None = None,
        reservation_token: str | None = None,
    ) -> StageSnapshot:
        """Store one staged batch after enforcing TTL and capacity limits."""
        with self._lock:
            now = self._now()
            self._expire_locked(now)
            if reservation_token is not None:
                if reservation_token not in self._reservations:
                    raise ValueError("Stage capacity reservation is invalid or expired.")
                self._reservations.remove(reservation_token)
            elif len(self._active) + len(self._reservations) >= self.max_active_stages:
                self._raise_capacity_locked()
            stage_id = secrets.token_urlsafe(32)
            while stage_id in self._active or stage_id in self._terminal:
                stage_id = secrets.token_urlsafe(32)
            record = _StageRecord(
                stage_id=stage_id,
                status="active",
                created_at=now,
                expires_at=now + timedelta(seconds=self.ttl_seconds),
                last_transition_at=now,
                renewal_count=0,
                status_reason=None,
                suggestions=suggestions.copy(deep=True).reset_index(drop=True),
                quality=quality.copy(deep=True).reset_index(drop=True),
                bundle=copy.deepcopy(bundle),
                config_path=Path(config_path).expanduser().resolve(),
                log_path=Path(log_path).expanduser().resolve(),
                stage_selection=stage_selection,
                context_values=None
                if context_values is None
                else copy.deepcopy(context_values),
            )
            self._active[stage_id] = record
            self._lifecycle_totals["created"] += 1
            return self._snapshot(record)

    def ensure_capacity(self) -> None:
        """Fail before expensive suggestion generation when no stage slot is available."""
        with self._lock:
            self._expire_locked(self._now())
            if len(self._active) + len(self._reservations) >= self.max_active_stages:
                self._raise_capacity_locked()

    def reserve_capacity(self) -> str:
        """Atomically reserve one stage slot before suggestion generation."""
        with self._lock:
            self._expire_locked(self._now())
            if len(self._active) + len(self._reservations) >= self.max_active_stages:
                self._raise_capacity_locked()
            token = secrets.token_urlsafe(24)
            while token in self._reservations:
                token = secrets.token_urlsafe(24)
            self._reservations.add(token)
            return token

    def release_reservation(self, token: str) -> None:
        """Release a capacity reservation after failed suggestion generation."""
        with self._lock:
            self._reservations.discard(token)

    def active_snapshots(self) -> list[StageSnapshot]:
        """Return deep copies of active records for lazy file-staleness checks."""
        with self._lock:
            self._expire_locked(self._now())
            return [self._snapshot(record) for record in self._active.values()]

    def validation_snapshots(self) -> list[StageValidationSnapshot]:
        """Return lightweight active-stage state for lock-free file validation."""
        with self._lock:
            self._expire_locked(self._now())
            return [
                self._validation_snapshot(record)
                for record in self._active.values()
                if record.status == "active"
            ]

    def get(self, stage_id: str) -> StageSnapshot:
        """Return an active or currently appending stage without extending its TTL."""
        with self._lock:
            self._expire_locked(self._now())
            record = self._active.get(stage_id)
            if record is None:
                self._raise_terminal_or_missing(stage_id)
            return self._snapshot(record)

    def list_summaries(
        self,
        *,
        include_terminal: bool = False,
        statuses: Collection[str] | None = None,
        limit: int = 50,
        validator: StageValidator | None = None,
    ) -> StageListing:
        """Return a bounded metadata-only lifecycle listing."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
            raise ValueError("stage listing limit must be an integer from 1 through 200")
        selected_statuses = set(statuses or STAGE_STATUSES)
        unknown = selected_statuses.difference(STAGE_STATUSES)
        if unknown:
            raise ValueError(f"Unknown stage status: {sorted(unknown)[0]}")

        validation_results: list[tuple[StageValidationSnapshot, str | None]] = []
        if validator is not None:
            validation_results = [
                (snapshot, validator(snapshot))
                for snapshot in self.validation_snapshots()
            ]

        with self._lock:
            now = self._now()
            self._expire_locked(now)
            for snapshot, reason in validation_results:
                if reason is None:
                    continue
                record = self._active.get(snapshot.stage_id)
                if (
                    record is not None
                    and record.status == "active"
                    and self._matches_validation_snapshot(record, snapshot)
                ):
                    self._terminalize_locked(
                        snapshot.stage_id,
                        "stale",
                        now=now,
                        reason=reason,
                    )
            summaries = [
                self._summary_from_active(record, now)
                for record in self._active.values()
                if record.status in selected_statuses
            ]
            if include_terminal:
                summaries.extend(
                    self._summary_from_tombstone(record)
                    for record in self._terminal.values()
                    if record.status in selected_statuses
                )
            summaries.sort(
                key=lambda item: (-item.last_transition_at.timestamp(), item.stage_id)
            )
            counts = Counter(item.status for item in summaries)
            total = len(summaries)
            returned_stages = tuple(summaries[:limit])
            return StageListing(
                stages=returned_stages,
                total=total,
                returned=len(returned_stages),
                truncated=total > limit,
                status_counts={
                    status: counts[status] for status in STAGE_STATUSES if counts[status]
                },
            )

    def claim(self, stage_id: str) -> StageSnapshot:
        """Atomically claim an active stage for exactly one append attempt."""
        with self._lock:
            now = self._now()
            self._expire_locked(now)
            record = self._active.get(stage_id)
            if record is None:
                self._raise_terminal_or_missing(stage_id)
            if record.status == "appending":
                raise StageStoreError(
                    "stage_in_use",
                    "This staged batch is already being appended.",
                    409,
                )
            self._transition_locked(
                record,
                "appending",
                now=now,
                reason="Append in progress.",
                counter="claimed",
            )
            return self._snapshot(record)

    def restore(self, stage_id: str) -> StageSnapshot:
        """Restore an appending stage after a retry-safe failure."""
        with self._lock:
            record = self._active.get(stage_id)
            if record is None:
                self._raise_terminal_or_missing(stage_id)
            if record.status == "appending":
                self._transition_locked(
                    record,
                    "active",
                    now=self._now(),
                    reason="Append failed safely; retry is allowed.",
                    counter="restored",
                )
            return self._snapshot(record)

    def renew(
        self,
        stage_id: str,
        *,
        validator: StageValidator | None = None,
    ) -> StageSnapshot:
        """Explicitly reset an active, file-valid stage's expiry."""
        with self._lock:
            self._expire_locked(self._now())
            record = self._active.get(stage_id)
            if record is None:
                self._raise_terminal_or_missing(stage_id)
            if record.status == "appending":
                raise StageStoreError(
                    "stage_in_use",
                    "This staged batch is currently being appended.",
                    409,
                )
            validation_snapshot = self._validation_snapshot(record)

        stale_reason = (
            None if validator is None else validator(validation_snapshot)
        )

        with self._lock:
            now = self._now()
            self._expire_locked(now)
            record = self._active.get(stage_id)
            if record is None:
                self._raise_terminal_or_missing(stage_id)
            if record.status == "appending":
                raise StageStoreError(
                    "stage_in_use",
                    "This staged batch is currently being appended.",
                    409,
                )
            if not self._matches_validation_snapshot(record, validation_snapshot):
                raise StageStoreError(
                    "stage_stale",
                    "Staged batch metadata changed during renewal.",
                    409,
                )
            if stale_reason is not None:
                self._terminalize_locked(
                    stage_id,
                    "stale",
                    now=now,
                    reason=stale_reason,
                )
                raise StageStoreError("stage_stale", stale_reason, 409)
            record.expires_at = now + timedelta(seconds=self.ttl_seconds)
            record.renewal_count += 1
            self._transition_locked(
                record,
                "active",
                now=now,
                reason="Stage expiry renewed.",
                counter="renewed",
            )
            return self._snapshot(record)

    def mark_stale_if_active(self, stage_id: str, reason: str | None = None) -> bool:
        """Retire an active stage without interfering with an append claim."""
        with self._lock:
            record = self._active.get(stage_id)
            if record is None or record.status != "active":
                return False
            self._terminalize_locked(
                stage_id,
                "stale",
                now=self._now(),
                reason=reason or "Campaign files changed after staging.",
            )
            return True

    def retire_for_log_change(
        self,
        *,
        log_path: str | Path,
        previous_log_fingerprint: str,
        consumed_suggestions_fingerprint: str | None = None,
    ) -> None:
        """Release active stages tied to a log snapshot that was just mutated."""
        canonical_log_path = Path(log_path).expanduser().resolve()
        with self._lock:
            now = self._now()
            for stage_id, record in list(self._active.items()):
                if record.status != "active" or record.log_path != canonical_log_path:
                    continue
                if record.bundle.get("log_fingerprint") != previous_log_fingerprint:
                    continue
                is_consumed = (
                    consumed_suggestions_fingerprint is not None
                    and record.bundle.get("suggestions_fingerprint")
                    == consumed_suggestions_fingerprint
                )
                self._terminalize_locked(
                    stage_id,
                    "consumed" if is_consumed else "stale",
                    now=now,
                    reason=(
                        "Staged batch appended through the compatibility endpoint."
                        if is_consumed
                        else "Campaign log changed after staging."
                    ),
                )

    def complete(
        self,
        stage_id: str,
        status: str,
        *,
        reason: str | None = None,
    ) -> StageSnapshot:
        """Move an appending stage to a bounded metadata-only tombstone."""
        if status not in {"consumed", "stale"}:
            raise ValueError(f"Unsupported terminal stage status: {status}")
        with self._lock:
            record = self._active.get(stage_id)
            if record is None:
                self._raise_terminal_or_missing(stage_id)
            self._transition_locked(
                record,
                status,
                now=self._now(),
                reason=reason or self._default_terminal_reason(status),
                counter=status,
            )
            result = self._snapshot(record)
            self._active.pop(stage_id, None)
            self._add_tombstone_locked(self._tombstone(record))
            return result

    def discard(self, stage_id: str) -> StageSnapshot:
        """Discard an active stage and retain a bounded metadata-only tombstone."""
        with self._lock:
            now = self._now()
            self._expire_locked(now)
            record = self._active.get(stage_id)
            if record is None:
                self._raise_terminal_or_missing(stage_id)
            if record.status == "appending":
                raise StageStoreError(
                    "stage_in_use",
                    "This staged batch is currently being appended.",
                    409,
                )
            self._transition_locked(
                record,
                "discarded",
                now=now,
                reason="Stage explicitly discarded.",
                counter="discarded",
            )
            result = self._snapshot(record)
            self._active.pop(stage_id, None)
            self._add_tombstone_locked(self._tombstone(record))
            return result

    def stats(self) -> dict[str, object]:
        """Return cheap additive health metadata without reading campaign files."""
        with self._lock:
            now = self._now()
            self._expire_locked(now)
            active_records = [
                record for record in self._active.values() if record.status == "active"
            ]
            appending_records = [
                record for record in self._active.values() if record.status == "appending"
            ]
            terminal_counts = Counter(record.status for record in self._terminal.values())
            used_capacity = len(self._active) + len(self._reservations)
            return {
                "active_stages": len(active_records),
                "appending_stages": len(appending_records),
                "reserved_stages": len(self._reservations),
                "max_staged_batches": self.max_active_stages,
                "capacity_remaining": max(0, self.max_active_stages - used_capacity),
                "stage_ttl_seconds": self.ttl_seconds,
                "oldest_active_age_seconds": self._oldest_age(active_records, now),
                "oldest_appending_age_seconds": self._oldest_age(
                    appending_records,
                    now,
                    since_transition=True,
                ),
                "terminal_tombstones": len(self._terminal),
                "terminal_status_counts": {
                    status: terminal_counts[status]
                    for status in ("consumed", "discarded", "stale", "expired")
                },
                "lifecycle_totals": dict(self._lifecycle_totals),
            }

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None:
            return now.replace(tzinfo=UTC)
        return now.astimezone(UTC)

    def _expire_locked(self, now: datetime) -> None:
        expired_ids = [
            stage_id
            for stage_id, record in self._active.items()
            if record.status == "active" and now >= record.expires_at
        ]
        for stage_id in expired_ids:
            self._terminalize_locked(
                stage_id,
                "expired",
                now=now,
                reason="Stage TTL expired.",
            )

    def _terminalize_locked(
        self,
        stage_id: str,
        status: str,
        *,
        now: datetime,
        reason: str,
    ) -> None:
        record = self._active[stage_id]
        self._transition_locked(
            record,
            status,
            now=now,
            reason=reason,
            counter=status,
        )
        self._active.pop(stage_id, None)
        self._add_tombstone_locked(self._tombstone(record))

    def _transition_locked(
        self,
        record: _StageRecord,
        status: str,
        *,
        now: datetime,
        reason: str | None,
        counter: str,
    ) -> None:
        record.status = status
        record.last_transition_at = now
        record.status_reason = reason
        self._lifecycle_totals[counter] += 1

    def _raise_capacity_locked(self) -> None:
        self._lifecycle_totals["capacity_rejected"] += 1
        raise StageStoreError(
            "stage_capacity",
            "The API stage store is full. Discard or append an active stage and retry.",
            503,
        )

    def _add_tombstone_locked(self, record: _StageTombstone) -> None:
        self._terminal[record.stage_id] = record
        self._terminal.move_to_end(record.stage_id)
        while len(self._terminal) > self._max_tombstones:
            self._terminal.popitem(last=False)

    def _raise_terminal_or_missing(self, stage_id: str) -> None:
        terminal = self._terminal.get(stage_id)
        if terminal is None:
            raise StageStoreError("stage_not_found", "Staged batch was not found.", 404)
        if terminal.status == "expired":
            raise StageStoreError("stage_expired", "Staged batch has expired.", 410)
        raise StageStoreError(
            f"stage_{terminal.status}",
            f"Staged batch is {terminal.status}.",
            409,
        )

    @staticmethod
    def _validation_snapshot(record: _StageRecord) -> StageValidationSnapshot:
        return StageValidationSnapshot(
            stage_id=record.stage_id,
            config_path=record.config_path,
            log_path=record.log_path,
            config_fingerprint=str(record.bundle.get("config_fingerprint", "")),
            log_fingerprint=str(record.bundle.get("log_fingerprint", "")),
        )

    @staticmethod
    def _matches_validation_snapshot(
        record: _StageRecord,
        snapshot: StageValidationSnapshot,
    ) -> bool:
        return (
            record.config_path == snapshot.config_path
            and record.log_path == snapshot.log_path
            and str(record.bundle.get("config_fingerprint", ""))
            == snapshot.config_fingerprint
            and str(record.bundle.get("log_fingerprint", ""))
            == snapshot.log_fingerprint
        )

    @staticmethod
    def _snapshot(record: _StageRecord) -> StageSnapshot:
        return StageSnapshot(
            stage_id=record.stage_id,
            status=record.status,
            created_at=record.created_at,
            expires_at=record.expires_at,
            last_transition_at=record.last_transition_at,
            renewal_count=record.renewal_count,
            status_reason=record.status_reason,
            suggestions=record.suggestions.copy(deep=True),
            quality=record.quality.copy(deep=True),
            bundle=copy.deepcopy(record.bundle),
            config_path=record.config_path,
            log_path=record.log_path,
            stage_selection=record.stage_selection,
            context_values=None
            if record.context_values is None
            else copy.deepcopy(record.context_values),
        )

    @staticmethod
    def _tombstone(record: _StageRecord) -> _StageTombstone:
        return _StageTombstone(
            stage_id=record.stage_id,
            status=record.status,
            created_at=record.created_at,
            expires_at=record.expires_at,
            last_transition_at=record.last_transition_at,
            renewal_count=record.renewal_count,
            status_reason=record.status_reason,
            suggestion_count=len(record.suggestions),
            config_path=record.config_path,
            log_path=record.log_path,
            stage_selection=record.stage_selection,
            context_variable_names=tuple(sorted((record.context_values or {}).keys())),
        )

    @staticmethod
    def _summary_from_active(record: _StageRecord, now: datetime) -> StageSummary:
        return StageSummary(
            stage_id=record.stage_id,
            status=record.status,
            created_at=record.created_at,
            expires_at=record.expires_at,
            last_transition_at=record.last_transition_at,
            remaining_ttl_seconds=max(0.0, (record.expires_at - now).total_seconds()),
            suggestion_count=len(record.suggestions),
            config_path=record.config_path,
            log_path=record.log_path,
            stage_selection=record.stage_selection,
            context_variable_names=tuple(sorted((record.context_values or {}).keys())),
            renewal_count=record.renewal_count,
            status_reason=record.status_reason,
        )

    @staticmethod
    def _summary_from_tombstone(record: _StageTombstone) -> StageSummary:
        return StageSummary(
            stage_id=record.stage_id,
            status=record.status,
            created_at=record.created_at,
            expires_at=record.expires_at,
            last_transition_at=record.last_transition_at,
            remaining_ttl_seconds=0.0,
            suggestion_count=record.suggestion_count,
            config_path=record.config_path,
            log_path=record.log_path,
            stage_selection=record.stage_selection,
            context_variable_names=record.context_variable_names,
            renewal_count=record.renewal_count,
            status_reason=record.status_reason,
        )

    @staticmethod
    def _oldest_age(
        records: list[_StageRecord],
        now: datetime,
        *,
        since_transition: bool = False,
    ) -> float | None:
        if not records:
            return None
        return max(
            0.0,
            max(
                (
                    now
                    - (
                        record.last_transition_at
                        if since_transition
                        else record.created_at
                    )
                ).total_seconds()
                for record in records
            ),
        )

    @staticmethod
    def _default_terminal_reason(status: str) -> str:
        return {
            "consumed": "Staged batch appended.",
            "stale": "Campaign state changed after staging.",
        }[status]
