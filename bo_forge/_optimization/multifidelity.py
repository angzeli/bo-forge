"""Internal candidate suggestion generation."""

from __future__ import annotations

import time
import uuid

import pandas as pd
import torch
from botorch.exceptions.errors import BotorchError

from bo_forge._optimization.common import (
    _candidate_batch_rejection_message,
    _CandidateGenerationExhausted,
    _empty_row,
    _populate_replicate_fields,
    _populate_review_fields,
)
from bo_forge.acquisition import (
    optimize_posterior_mean_at_target_fidelity,
    optimize_qmf_kg,
)
from bo_forge.config import CampaignConfig
from bo_forge.errors import SuggestionError
from bo_forge.models import (
    fit_multi_fidelity_gp_model,
)
from bo_forge.multifidelity import (
    fidelity_level_fixed_features,
)
from bo_forge.transforms import (
    categorical_feature_assignments,
    encoded_dimension,
    objective_from_model_space,
    unit_cube_to_user_values,
    values_to_unit_cube,
)
from bo_forge.validation import (
    canonical_columns,
    next_iteration,
)

MAX_CATEGORICAL_COMBINATIONS = 64
MAX_DECODE_RETRIES = 8
MAX_INITIAL_DESIGN_BATCHES = 1000
_GENERATION_FAILURE_HINT = (
    "The feasible design space may be exhausted, constraints may be too restrictive, "
    "or bo.min_normalized_distance may be too large."
)
SUGGESTION_QUALITY_COLUMNS = [
    "row_id",
    "is_feasible",
    "violated_constraints",
    "is_exact_duplicate",
    "duplicate_allowed_by_replicates",
    "nearest_existing_distance",
    "nearest_batch_distance",
    "passes_distance_threshold",
]




def _suggest_multi_fidelity_model_based(
    config: CampaignConfig,
    df: pd.DataFrame,
    observed_df: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    try:
        torch.manual_seed(config.bo.random_seed)
        model = fit_multi_fidelity_gp_model(config, observed_df)
        fixed_features_list = categorical_feature_assignments(config)
        candidate_fixed_features_list = fidelity_level_fixed_features(config)
        model_dim = encoded_dimension(config)
        deadline = _qmfkg_optimizer_deadline(config)
        try:
            current_value = optimize_posterior_mean_at_target_fidelity(
                config=config,
                model=model,
                model_dim=model_dim,
                fixed_features_list=fixed_features_list,
                timeout_seconds=_remaining_qmfkg_optimizer_time(config, deadline),
            )
        except TimeoutError as exc:
            raise SuggestionError(_qmfkg_timeout_message(config)) from exc
        user_candidates: list[tuple[object, ...]] | None = None
        acquisition_value: torch.Tensor | None = None
        rejection_message = "no candidate was decoded"

        for attempt in range(MAX_DECODE_RETRIES):
            remaining_timeout = _remaining_qmfkg_optimizer_time(
                config,
                deadline,
                last_rejection=rejection_message if attempt > 0 else None,
            )
            torch.manual_seed(config.bo.random_seed + attempt)
            try:
                x_unit_raw, acquisition_value, _ = optimize_qmf_kg(
                    config=config,
                    model=model,
                    current_value=current_value,
                    batch_size=batch_size,
                    model_dim=model_dim,
                    fixed_features_list=candidate_fixed_features_list,
                    timeout_seconds=remaining_timeout,
                )
            except TimeoutError as exc:
                raise SuggestionError(
                    _qmfkg_timeout_message(
                        config,
                        rejection_message if attempt > 0 else None,
                    )
                ) from exc
            _check_qmfkg_optimizer_deadline(
                config,
                deadline,
                last_rejection=rejection_message if attempt > 0 else None,
            )
            decoded_candidates = unit_cube_to_user_values(config, x_unit_raw)
            rejection_message = _candidate_batch_rejection_message(
                config,
                df,
                decoded_candidates,
            )
            if rejection_message is None:
                user_candidates = decoded_candidates
                break

        if user_candidates is None or acquisition_value is None:
            raise _CandidateGenerationExhausted(
                "Could not generate a feasible, non-duplicate qMFKG suggestion after "
                f"{MAX_DECODE_RETRIES} retries. {_GENERATION_FAILURE_HINT} "
                f"Last rejection: {rejection_message}"
            )

        x_unit_repaired = values_to_unit_cube(config, user_candidates)
        with torch.no_grad():
            posterior = model.posterior(x_unit_repaired)
            mean_model = posterior.mean.squeeze(-1)
            std = posterior.variance.clamp_min(0.0).sqrt().squeeze(-1)
            mean_user = objective_from_model_space(config, mean_model)
    except SuggestionError:
        raise
    except TimeoutError as exc:
        raise SuggestionError(f"Could not generate qMFKG suggestion: {exc}") from exc
    except (BotorchError, RuntimeError, ValueError) as exc:
        raise SuggestionError(f"Could not generate qMFKG suggestion: {exc}") from exc

    rows: list[dict[str, object]] = []
    iteration = next_iteration(df)
    acquisition_scalar = float(acquisition_value.reshape(-1)[0])
    for index, candidate in enumerate(user_candidates):
        row = _empty_row(config)
        row["row_id"] = uuid.uuid4().hex
        row["iteration"] = iteration
        row["status"] = "suggested"
        row["source"] = "qmf_kg"
        _populate_replicate_fields(config, row)
        _populate_review_fields(config, row)
        for name, value in zip(config.variable_names, candidate, strict=True):
            row[name] = value
        row["predicted_mean"] = float(mean_user.reshape(-1)[index])
        row["predicted_std"] = float(std.reshape(-1)[index])
        row["acquisition"] = acquisition_scalar
        rows.append(row)
    return pd.DataFrame(rows, columns=canonical_columns(config))


def _qmfkg_optimizer_deadline(config: CampaignConfig) -> float | None:
    if config.fidelity is None or config.fidelity.optimizer_timeout_seconds is None:
        return None
    return time.monotonic() + config.fidelity.optimizer_timeout_seconds


def _remaining_qmfkg_optimizer_time(
    config: CampaignConfig,
    deadline: float | None,
    *,
    last_rejection: str | None = None,
) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining > 0:
        return remaining
    raise SuggestionError(_qmfkg_timeout_message(config, last_rejection))


def _check_qmfkg_optimizer_deadline(
    config: CampaignConfig,
    deadline: float | None,
    *,
    last_rejection: str | None = None,
) -> None:
    if deadline is not None and time.monotonic() > deadline:
        raise SuggestionError(_qmfkg_timeout_message(config, last_rejection))


def _qmfkg_timeout_message(
    config: CampaignConfig,
    last_rejection: str | None = None,
) -> str:
    assert config.fidelity is not None
    timeout = config.fidelity.optimizer_timeout_seconds
    timeout_label = "unset" if timeout is None else f"{timeout:g}"
    message = (
        "qMFKG acquisition optimization timed out before a valid candidate batch "
        f"was available: optimizer_timeout_seconds={timeout_label}."
    )
    if last_rejection is not None:
        message += f" Last rejection: {last_rejection}"
    return message
