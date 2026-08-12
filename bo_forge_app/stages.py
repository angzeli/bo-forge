"""Thread-safe in-memory staging for the experimental API probe."""

from __future__ import annotations

import copy
import math
import secrets
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd


class StageStoreError(ValueError):
    """Structured stage lifecycle error exposed by the API adapter."""

    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass
class StageSnapshot:
    """Deep-copied API stage state safe to return outside the store lock."""

    stage_id: str
    status: str
    created_at: datetime
    expires_at: datetime
    suggestions: pd.DataFrame
    quality: pd.DataFrame
    bundle: dict[str, object]
    config_path: Path
    log_path: Path
    stage_selection: str | None
    context_values: dict[str, object] | None


@dataclass
class _StageRecord:
    stage_id: str
    status: str
    created_at: datetime
    expires_at: datetime
    suggestions: pd.DataFrame
    quality: pd.DataFrame
    bundle: dict[str, object]
    config_path: Path
    log_path: Path
    stage_selection: str | None
    context_values: dict[str, object] | None


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
        self._terminal: OrderedDict[str, _StageRecord] = OrderedDict()
        self._max_tombstones = max(32, self.max_active_stages * 2)

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
                raise StageStoreError(
                    "stage_capacity",
                    "The API stage store is full. Discard or append an active stage and retry.",
                    503,
                )
            stage_id = secrets.token_urlsafe(32)
            while stage_id in self._active or stage_id in self._terminal:
                stage_id = secrets.token_urlsafe(32)
            record = _StageRecord(
                stage_id=stage_id,
                status="active",
                created_at=now,
                expires_at=now + timedelta(seconds=self.ttl_seconds),
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
            return self._snapshot(record)

    def ensure_capacity(self) -> None:
        """Fail before expensive suggestion generation when no stage slot is available."""
        with self._lock:
            self._expire_locked(self._now())
            if len(self._active) + len(self._reservations) >= self.max_active_stages:
                raise StageStoreError(
                    "stage_capacity",
                    "The API stage store is full. Discard or append an active stage and retry.",
                    503,
                )

    def reserve_capacity(self) -> str:
        """Atomically reserve one stage slot before suggestion generation."""
        with self._lock:
            self._expire_locked(self._now())
            if len(self._active) + len(self._reservations) >= self.max_active_stages:
                raise StageStoreError(
                    "stage_capacity",
                    "The API stage store is full. Discard or append an active stage and retry.",
                    503,
                )
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

    def get(self, stage_id: str) -> StageSnapshot:
        """Return an active or currently appending stage without exposing store state."""
        with self._lock:
            self._expire_locked(self._now())
            record = self._active.get(stage_id)
            if record is None:
                self._raise_terminal_or_missing(stage_id)
            return self._snapshot(record)

    def claim(self, stage_id: str) -> StageSnapshot:
        """Atomically claim an active stage for exactly one append attempt."""
        with self._lock:
            self._expire_locked(self._now())
            record = self._active.get(stage_id)
            if record is None:
                self._raise_terminal_or_missing(stage_id)
            if record.status == "appending":
                raise StageStoreError(
                    "stage_in_use",
                    "This staged batch is already being appended.",
                    409,
                )
            record.status = "appending"
            return self._snapshot(record)

    def restore(self, stage_id: str) -> StageSnapshot:
        """Restore an appending stage after a retry-safe failure."""
        with self._lock:
            record = self._active.get(stage_id)
            if record is None:
                self._raise_terminal_or_missing(stage_id)
            if record.status == "appending":
                record.status = "active"
            return self._snapshot(record)

    def mark_stale_if_active(self, stage_id: str) -> bool:
        """Retire an active stage without interfering with an append claim."""
        with self._lock:
            record = self._active.get(stage_id)
            if record is None or record.status != "active":
                return False
            record.status = "stale"
            self._active.pop(stage_id, None)
            self._add_tombstone_locked(record)
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
            for stage_id, record in list(self._active.items()):
                if record.status != "active" or record.log_path != canonical_log_path:
                    continue
                if record.bundle.get("log_fingerprint") != previous_log_fingerprint:
                    continue
                record.status = (
                    "consumed"
                    if consumed_suggestions_fingerprint is not None
                    and record.bundle.get("suggestions_fingerprint")
                    == consumed_suggestions_fingerprint
                    else "stale"
                )
                self._active.pop(stage_id, None)
                self._add_tombstone_locked(record)

    def complete(self, stage_id: str, status: str) -> StageSnapshot:
        """Move an appending stage to a bounded terminal tombstone."""
        if status not in {"consumed", "stale"}:
            raise ValueError(f"Unsupported terminal stage status: {status}")
        with self._lock:
            record = self._active.get(stage_id)
            if record is None:
                self._raise_terminal_or_missing(stage_id)
            record.status = status
            self._active.pop(stage_id, None)
            self._add_tombstone_locked(record)
            return self._snapshot(record)

    def discard(self, stage_id: str) -> StageSnapshot:
        """Discard an active stage and retain a bounded terminal tombstone."""
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
            record.status = "discarded"
            self._active.pop(stage_id, None)
            self._add_tombstone_locked(record)
            return self._snapshot(record)

    def stats(self) -> dict[str, object]:
        """Return additive health metadata after lazily expiring old stages."""
        with self._lock:
            self._expire_locked(self._now())
            return {
                "active_stages": sum(
                    record.status == "active" for record in self._active.values()
                ),
                "appending_stages": sum(
                    record.status == "appending" for record in self._active.values()
                ),
                "reserved_stages": len(self._reservations),
                "max_staged_batches": self.max_active_stages,
                "stage_ttl_seconds": self.ttl_seconds,
                "terminal_tombstones": len(self._terminal),
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
            record = self._active.pop(stage_id)
            record.status = "expired"
            self._add_tombstone_locked(record)

    def _add_tombstone_locked(self, record: _StageRecord) -> None:
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
    def _snapshot(record: _StageRecord) -> StageSnapshot:
        return StageSnapshot(
            stage_id=record.stage_id,
            status=record.status,
            created_at=record.created_at,
            expires_at=record.expires_at,
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
