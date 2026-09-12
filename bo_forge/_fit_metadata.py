"""Immutable, caller-owned evidence for a particular model fit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from bo_forge.config import CampaignConfig
from bo_forge.errors import BOForgeError


@dataclass(frozen=True)
class FitMetadata:
    """Scalar fit evidence; no model, tensor, or process-global state is retained."""

    fields: tuple[tuple[str, str | int | bool | None], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fields, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], (str, int, bool, type(None)))
            for item in self.fields
        ):
            raise TypeError("FitMetadata requires immutable scalar field pairs.")

    def as_dict(self) -> dict[str, Any]:
        return dict(self.fields)


@dataclass(frozen=True)
class FitResult:
    """Internal fit outcome keeping the BoTorch model and its evidence together."""

    model: Any
    metadata: FitMetadata


def config_identity(config: CampaignConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def attach_fit_metadata(frame, model):
    """Carry fit evidence through internal candidate assembly, never into CSV."""
    metadata = getattr(model, "_bo_forge_fit_metadata", None)
    if isinstance(metadata, FitMetadata):
        frame.attrs["_bo_forge_fit_metadata"] = metadata
    return frame


def failed_fit_metadata(error: BaseException) -> FitMetadata | None:
    """Recover evidence through suggestion error translation without shared state."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        evidence = getattr(error, "_bo_forge_fit_metadata", None)
        if isinstance(evidence, FitMetadata):
            return evidence
        error = error.__cause__ or error.__context__
    return None


def session_suggestions(session, batch_size, stage, context_values):
    """Transfer evidence to this caller without changing public suggestion results."""
    from bo_forge.suggestions import suggest_next

    session._fit_metadata = None
    try:
        suggestions = suggest_next(
            session.config,
            session.df.copy(deep=True),
            batch_size=batch_size,
            stage=stage,
            context_values=context_values,
        )
    except (BOForgeError, RuntimeError, ValueError) as exc:
        session._fit_metadata = failed_fit_metadata(exc)
        raise
    session._fit_metadata = suggestions.attrs.pop("_bo_forge_fit_metadata", None)
    return suggestions
