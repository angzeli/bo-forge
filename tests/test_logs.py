import multiprocessing
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from filelock import FileLock

import bo_forge.logs as logs_module
from bo_forge.config import (
    BOConfig,
    CampaignConfig,
    CostConfig,
    FidelityConfig,
    ObjectiveConfig,
    ReplicateConfig,
    ReviewConfig,
    StageConfig,
    VariableConfig,
)
from bo_forge.errors import LogBusyError, LogConflictError, LogValidationError, LogWriteError
from bo_forge.logs import append_suggestions, load_campaign_log, mark_observed, review_suggestion
from bo_forge.validation import canonical_columns


def config() -> CampaignConfig:
    return CampaignConfig(
        campaign_name="test",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=BOConfig(batch_size=1, initial_design_size=1),
    )


def cost_review_config() -> CampaignConfig:
    cfg = config()
    return CampaignConfig(
        campaign_name=cfg.campaign_name,
        objective=cfg.objective,
        variables=cfg.variables,
        bo=cfg.bo,
        cost=CostConfig(expression="1.0 + x", budget=10.0),
        review=ReviewConfig(enabled=True),
    )


def replicate_config() -> CampaignConfig:
    cfg = config()
    return CampaignConfig(
        campaign_name=cfg.campaign_name,
        objective=cfg.objective,
        variables=cfg.variables,
        bo=cfg.bo,
        replicates=ReplicateConfig(enabled=True),
    )


def fidelity_config() -> CampaignConfig:
    return CampaignConfig(
        campaign_name="fidelity_test",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("fidelity", "continuous", 0.2, 1.0),
        ),
        bo=BOConfig(batch_size=1, initial_design_size=1, acquisition="qmf_kg"),
        fidelity=FidelityConfig(variable="fidelity", target=1.0),
    )


def discrete_fidelity_config() -> CampaignConfig:
    cfg = fidelity_config()
    return CampaignConfig(
        campaign_name=cfg.campaign_name,
        objective=cfg.objective,
        variables=cfg.variables,
        bo=cfg.bo,
        fidelity=FidelityConfig(
            variable="fidelity",
            target=1.0,
            levels=(0.25, 0.5, 0.75, 1.0),
        ),
    )


def structured_config(*, review: bool = False) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="structured_test",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("temperature", "continuous", 300.0, 900.0),
        ),
        bo=BOConfig(batch_size=1, initial_design_size=1),
        review=ReviewConfig(enabled=review),
        stages=(
            StageConfig("screen", ("x",)),
            StageConfig("refine", ("x", "temperature")),
        ),
    )


def suggestion(row_id: str = "suggested_1", *, x: float = 0.4) -> pd.DataFrame:
    cfg = config()
    return pd.DataFrame(
        [
            {
                "row_id": row_id,
                "iteration": 0,
                "status": "suggested",
                "source": "sobol",
                "x": x,
                "activity": "",
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )


def _append_in_process(
    log_path: str,
    row_id: str,
    barrier: Any,
    results: Any,
) -> None:
    try:
        barrier.wait(timeout=10)
        x = 0.2 if row_id == "process_1" else 0.8
        append_suggestions(log_path, suggestion(row_id, x=x), config=config())
    except Exception as exc:  # pragma: no cover - child-process failure reporting
        results.put((row_id, type(exc).__name__, str(exc)))
    else:
        results.put((row_id, "ok", ""))


def cost_review_suggestion(row_id: str = "suggested_1") -> pd.DataFrame:
    cfg = cost_review_config()
    return pd.DataFrame(
        [
            {
                "row_id": row_id,
                "iteration": 0,
                "status": "suggested",
                "source": "sobol",
                "review_status": "pending",
                "review_note": "",
                "x": 0.4,
                "activity": "",
                "cost_estimate": 1.4,
                "cost_actual": "",
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
                "utility": "",
            }
        ],
        columns=canonical_columns(cfg),
    )


def structured_suggestion(*, review: bool = False) -> pd.DataFrame:
    cfg = structured_config(review=review)
    row = {
        "row_id": "structured_1",
        "iteration": 0,
        "status": "suggested",
        "source": "manual",
        "stage": "screen",
        "x": 0.4,
        "temperature": "",
        "activity": "",
        "predicted_mean": "",
        "predicted_std": "",
        "acquisition": "",
    }
    if review:
        row["review_status"] = "pending"
        row["review_note"] = ""
    return pd.DataFrame([row], columns=canonical_columns(cfg))


def qmfkg_suggestion(row_id: str = "qmfkg_1") -> pd.DataFrame:
    cfg = fidelity_config()
    return pd.DataFrame(
        [
            {
                "row_id": row_id,
                "iteration": 1,
                "status": "suggested",
                "source": "qmf_kg",
                "x": 0.4,
                "fidelity": 0.8,
                "activity": "",
                "predicted_mean": 1.2,
                "predicted_std": 0.1,
                "acquisition": 0.01,
            }
        ],
        columns=canonical_columns(cfg),
    )


def qlog_nehvi_config() -> CampaignConfig:
    return CampaignConfig(
        campaign_name="qlog_nehvi_test",
        objective=ObjectiveConfig("yield_score", "maximize", 40.0),
        objectives=(
            ObjectiveConfig("yield_score", "maximize", 40.0),
            ObjectiveConfig("waste_score", "minimize", 25.0),
        ),
        variables=(
            VariableConfig("temperature", "continuous", 20.0, 100.0),
            VariableConfig("solvent", "categorical", values=("MeCN", "Water")),
        ),
        bo=BOConfig(batch_size=1, initial_design_size=1, acquisition="qlog_nehvi"),
    )


def qlog_nehvi_suggestion(
    row_id: str = "qlog_nehvi_1",
    *,
    source: str = "qlog_nehvi",
) -> pd.DataFrame:
    cfg = qlog_nehvi_config()
    return pd.DataFrame(
        [
            {
                "row_id": row_id,
                "iteration": 1,
                "status": "suggested",
                "source": source,
                "temperature": 72.0,
                "solvent": "MeCN",
                "yield_score": "",
                "waste_score": "",
                "predicted_mean_yield_score": 65.0,
                "predicted_std_yield_score": 1.5,
                "predicted_mean_waste_score": 16.0,
                "predicted_std_waste_score": 0.8,
                "acquisition": 0.01,
            }
        ],
        columns=canonical_columns(cfg),
    )


def test_append_suggestions_and_mark_observed_round_trip(tmp_path: Path) -> None:
    cfg = config()
    log_path = tmp_path / "campaign.csv"

    append_suggestions(log_path, suggestion())
    mark_observed(log_path, "suggested_1", 1.7)

    df = load_campaign_log(log_path, cfg)
    assert len(df) == 1
    assert df.loc[0, "row_id"] == "suggested_1"
    assert df.loc[0, "status"] == "observed"
    assert float(df.loc[0, "activity"]) == pytest.approx(1.7)
    assert float(df.loc[0, "x"]) == pytest.approx(0.4)


def test_append_suggestions_without_config_still_supports_non_replicate_logs(
    tmp_path: Path,
) -> None:
    cfg = config()
    log_path = tmp_path / "campaign.csv"

    append_suggestions(log_path, suggestion("non_replicate"))

    df = load_campaign_log(log_path, cfg)
    assert df.loc[0, "row_id"] == "non_replicate"


def test_review_suggestion_and_mark_observed_with_actual_cost(tmp_path: Path) -> None:
    cfg = cost_review_config()
    log_path = tmp_path / "campaign.csv"

    append_suggestions(log_path, cost_review_suggestion())
    review_suggestion(log_path, "suggested_1", "accept", " approved ")
    mark_observed(log_path, "suggested_1", 1.7, actual_cost=1.25)

    df = load_campaign_log(log_path, cfg)
    assert df.loc[0, "status"] == "observed"
    assert df.loc[0, "review_status"] == "accepted"
    assert df.loc[0, "review_note"] == "approved"
    assert float(df.loc[0, "cost_actual"]) == pytest.approx(1.25)


def test_mark_observed_rejects_unaccepted_review_row(tmp_path: Path) -> None:
    log_path = tmp_path / "campaign.csv"
    append_suggestions(log_path, cost_review_suggestion())

    with pytest.raises(LogWriteError, match="review_status is 'pending', not 'accepted'"):
        mark_observed(log_path, "suggested_1", 1.7)


def test_review_suggestion_rejects_newline_note(tmp_path: Path) -> None:
    log_path = tmp_path / "campaign.csv"
    append_suggestions(log_path, cost_review_suggestion())

    with pytest.raises(LogWriteError, match="review_note cannot contain newline"):
        review_suggestion(log_path, "suggested_1", "accept", "first\nsecond")


def test_review_suggestion_rejects_non_review_log(tmp_path: Path) -> None:
    log_path = tmp_path / "campaign.csv"
    append_suggestions(log_path, suggestion())

    with pytest.raises(LogWriteError, match="review is not enabled"):
        review_suggestion(log_path, "suggested_1", "accept")


def test_mark_observed_rejects_actual_cost_without_cost_columns(tmp_path: Path) -> None:
    log_path = tmp_path / "campaign.csv"
    append_suggestions(log_path, suggestion())

    with pytest.raises(LogWriteError, match="no cost columns"):
        mark_observed(log_path, "suggested_1", 1.7, actual_cost=1.2)


def test_mark_observed_rejects_negative_actual_cost(tmp_path: Path) -> None:
    log_path = tmp_path / "campaign.csv"
    append_suggestions(log_path, cost_review_suggestion())
    review_suggestion(log_path, "suggested_1", "accept")

    with pytest.raises(LogWriteError, match="finite and >= 0"):
        mark_observed(log_path, "suggested_1", 1.7, actual_cost=-1.0)


def test_append_suggestions_rejects_observed_rows(tmp_path: Path) -> None:
    log_path = tmp_path / "campaign.csv"
    rows = suggestion()
    rows.loc[0, "status"] = "observed"
    rows.loc[0, "activity"] = "1.0"

    with pytest.raises(LogWriteError, match="expected status='suggested'"):
        append_suggestions(log_path, rows)


def test_append_suggestions_rejects_duplicate_row_id(tmp_path: Path) -> None:
    log_path = tmp_path / "campaign.csv"
    append_suggestions(log_path, suggestion("same"))

    with pytest.raises(LogWriteError, match="duplicate row_id 'same'"):
        append_suggestions(log_path, suggestion("same"))


def test_append_suggestions_rejects_duplicate_replicate_pair_structurally(
    tmp_path: Path,
) -> None:
    cfg = replicate_config()
    rows = pd.DataFrame(
        [
            {
                "row_id": "repeat_0",
                "iteration": 0,
                "status": "suggested",
                "source": "log_ei",
                "replicate_group": "group_0",
                "replicate_index": 1,
                "x": 0.4,
                "activity": "",
                "predicted_mean": 1.0,
                "predicted_std": 0.1,
                "acquisition": 0.0,
            },
            {
                "row_id": "repeat_1",
                "iteration": 0,
                "status": "suggested",
                "source": "log_ei",
                "replicate_group": "group_0",
                "replicate_index": 1,
                "x": 0.4,
                "activity": "",
                "predicted_mean": 1.0,
                "predicted_std": 0.1,
                "acquisition": 0.0,
            },
        ],
        columns=canonical_columns(cfg),
    )
    log_path = tmp_path / "campaign.csv"

    with pytest.raises(LogValidationError, match="Duplicate replicate row"):
        append_suggestions(log_path, rows)

    assert not log_path.exists()


def test_append_suggestions_with_config_rejects_typed_equivalent_replicate_group_without_mutation(
    tmp_path: Path,
) -> None:
    cfg = replicate_config()
    existing = pd.DataFrame(
        [
            {
                "row_id": "observed_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "replicate_group": "group_0",
                "replicate_index": 0,
                "x": 0.4,
                "activity": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )
    log_path = tmp_path / "campaign.csv"
    existing.to_csv(log_path, index=False)
    suggestions = pd.DataFrame(
        [
            {
                "row_id": "bad_repeat",
                "iteration": 1,
                "status": "suggested",
                "source": "log_ei",
                "replicate_group": "group_1",
                "replicate_index": 0,
                "x": "0.4000",
                "activity": "",
                "predicted_mean": 1.0,
                "predicted_std": 0.1,
                "acquisition": 0.0,
            }
        ],
        columns=canonical_columns(cfg),
    )
    before = log_path.read_bytes()

    with pytest.raises(LogValidationError, match="same design must share"):
        append_suggestions(log_path, suggestions, config=cfg)

    assert log_path.read_bytes() == before


def test_append_suggestions_requires_config_for_replicate_logs_without_mutation(
    tmp_path: Path,
) -> None:
    cfg = replicate_config()
    existing = pd.DataFrame(
        [
            {
                "row_id": "observed_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "replicate_group": "group_0",
                "replicate_index": 0,
                "x": 0.4,
                "activity": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )
    log_path = tmp_path / "campaign.csv"
    existing.to_csv(log_path, index=False)
    suggestions = pd.DataFrame(
        [
            {
                "row_id": "bad_repeat",
                "iteration": 1,
                "status": "suggested",
                "source": "log_ei",
                "replicate_group": "group_1",
                "replicate_index": 0,
                "x": 0.4,
                "activity": "",
                "predicted_mean": 1.0,
                "predicted_std": 0.1,
                "acquisition": 0.0,
            }
        ],
        columns=canonical_columns(cfg),
    )
    before = log_path.read_bytes()

    with pytest.raises(LogWriteError, match="Replicate append requires config-aware validation"):
        append_suggestions(log_path, suggestions)

    assert log_path.read_bytes() == before


def test_append_suggestions_requires_config_for_qmfkg_logs_without_mutation(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "campaign.csv"
    qmfkg_suggestion("existing").to_csv(log_path, index=False)
    before = log_path.read_bytes()

    with pytest.raises(LogWriteError, match="qMFKG append requires config-aware validation"):
        append_suggestions(log_path, qmfkg_suggestion())

    assert log_path.read_bytes() == before


def test_append_suggestions_requires_config_for_qlog_nehvi_rows_without_mutation(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "campaign.csv"
    qlog_nehvi_suggestion("existing", source="sobol").to_csv(log_path, index=False)
    before = log_path.read_bytes()

    with pytest.raises(LogWriteError, match="qLogNEHVI append requires config-aware validation"):
        append_suggestions(log_path, qlog_nehvi_suggestion())

    assert log_path.read_bytes() == before


def test_append_suggestions_with_config_accepts_qmfkg_logs(tmp_path: Path) -> None:
    cfg = fidelity_config()
    log_path = tmp_path / "campaign.csv"

    append_suggestions(log_path, qmfkg_suggestion(), config=cfg)

    df = load_campaign_log(log_path, cfg)
    assert df.loc[0, "source"] == "qmf_kg"


def test_append_suggestions_rejects_off_grid_fidelity_without_mutation(
    tmp_path: Path,
) -> None:
    cfg = discrete_fidelity_config()
    log_path = tmp_path / "campaign.csv"
    pd.DataFrame(columns=canonical_columns(cfg)).to_csv(log_path, index=False)
    suggestions = qmfkg_suggestion()
    before = log_path.read_bytes()

    with pytest.raises(LogValidationError, match="off-grid fidelity"):
        append_suggestions(log_path, suggestions, config=cfg)

    assert log_path.read_bytes() == before


def test_mark_observed_requires_config_for_qmfkg_logs_without_mutation(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "campaign.csv"
    qmfkg_suggestion().to_csv(log_path, index=False)
    before = log_path.read_bytes()

    with pytest.raises(LogWriteError, match="qMFKG mark_observed requires"):
        mark_observed(log_path, "qmfkg_1", 1.7)

    assert log_path.read_bytes() == before


def test_mark_observed_rejects_off_grid_fidelity_without_mutation(
    tmp_path: Path,
) -> None:
    cfg = discrete_fidelity_config()
    log_path = tmp_path / "campaign.csv"
    qmfkg_suggestion().to_csv(log_path, index=False)
    before = log_path.read_bytes()

    with pytest.raises(LogValidationError, match="off-grid fidelity"):
        mark_observed(log_path, "qmfkg_1", 1.7, config=cfg)

    assert log_path.read_bytes() == before


def test_mark_observed_requires_config_for_structured_logs_without_mutation(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "campaign.csv"
    structured_suggestion().to_csv(log_path, index=False)
    before = log_path.read_bytes()

    with pytest.raises(LogWriteError, match="Structured campaign mark_observed requires"):
        mark_observed(log_path, "structured_1", 1.7)

    assert log_path.read_bytes() == before


def test_mark_observed_with_config_supports_structured_logs(tmp_path: Path) -> None:
    cfg = structured_config()
    log_path = tmp_path / "campaign.csv"
    structured_suggestion().to_csv(log_path, index=False)

    mark_observed(log_path, "structured_1", 1.7, config=cfg)

    df = load_campaign_log(log_path, cfg)
    assert df.loc[0, "status"] == "observed"
    assert float(df.loc[0, "activity"]) == pytest.approx(1.7)


def test_review_suggestion_requires_config_for_structured_logs_without_mutation(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "campaign.csv"
    structured_suggestion(review=True).to_csv(log_path, index=False)
    before = log_path.read_bytes()

    with pytest.raises(LogWriteError, match="Structured campaign review_suggestion requires"):
        review_suggestion(log_path, "structured_1", "accept")

    assert log_path.read_bytes() == before


def test_review_suggestion_with_config_supports_structured_logs(tmp_path: Path) -> None:
    cfg = structured_config(review=True)
    log_path = tmp_path / "campaign.csv"
    structured_suggestion(review=True).to_csv(log_path, index=False)

    review_suggestion(log_path, "structured_1", "accept", config=cfg)

    df = load_campaign_log(log_path, cfg)
    assert df.loc[0, "review_status"] == "accepted"


def test_mark_observed_rejects_missing_row_id(tmp_path: Path) -> None:
    log_path = tmp_path / "campaign.csv"
    append_suggestions(log_path, suggestion("present"))

    with pytest.raises(LogWriteError, match="row_id was not found"):
        mark_observed(log_path, "missing", 1.0)


def test_mark_observed_rejects_already_observed_row(tmp_path: Path) -> None:
    log_path = tmp_path / "campaign.csv"
    append_suggestions(log_path, suggestion("row_1"))
    mark_observed(log_path, "row_1", 1.0)

    with pytest.raises(LogWriteError, match="status is 'observed', not 'suggested'"):
        mark_observed(log_path, "row_1", 1.2)


def test_append_rejects_stale_expected_fingerprint_without_mutation(
    tmp_path: Path,
) -> None:
    cfg = config()
    log_path = tmp_path / "campaign.csv"
    suggestion("existing").to_csv(log_path, index=False)
    before = log_path.read_bytes()

    with pytest.raises(LogConflictError, match="changed after it was loaded"):
        append_suggestions(
            log_path,
            suggestion("new"),
            config=cfg,
            expected_log_fingerprint="stale",
        )

    assert log_path.read_bytes() == before


def test_review_rejects_stale_expected_fingerprint_without_mutation(
    tmp_path: Path,
) -> None:
    cfg = cost_review_config()
    log_path = tmp_path / "campaign.csv"
    cost_review_suggestion().to_csv(log_path, index=False)
    before = log_path.read_bytes()

    with pytest.raises(LogConflictError, match="changed after it was loaded"):
        review_suggestion(
            log_path,
            "suggested_1",
            "accept",
            config=cfg,
            expected_log_fingerprint="stale",
        )

    assert log_path.read_bytes() == before


def test_mark_observed_rejects_stale_expected_fingerprint_without_mutation(
    tmp_path: Path,
) -> None:
    cfg = cost_review_config()
    log_path = tmp_path / "campaign.csv"
    cost_review_suggestion().to_csv(log_path, index=False)
    review_suggestion(log_path, "suggested_1", "accept", config=cfg)
    before = log_path.read_bytes()

    with pytest.raises(LogConflictError, match="changed after it was loaded"):
        mark_observed(
            log_path,
            "suggested_1",
            1.2,
            config=cfg,
            expected_log_fingerprint="stale",
        )

    assert log_path.read_bytes() == before


def test_log_lock_timeout_leaves_csv_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "campaign.csv"
    suggestion("existing").to_csv(log_path, index=False)
    before = log_path.read_bytes()
    monkeypatch.setattr(logs_module, "LOG_LOCK_TIMEOUT_SECONDS", 0.01)

    with FileLock(logs_module._log_lock_path(log_path)):
        with pytest.raises(LogBusyError, match="is busy"):
            append_suggestions(log_path, suggestion("new"), config=config())

    assert log_path.read_bytes() == before
    append_suggestions(log_path, suggestion("after_release", x=0.8), config=config())
    assert len(pd.read_csv(log_path, keep_default_na=False)) == 2


def test_symlinked_and_canonical_logs_share_one_lock(tmp_path: Path) -> None:
    log_path = tmp_path / "campaign.csv"
    suggestion("existing").to_csv(log_path, index=False)
    linked = tmp_path / "linked.csv"
    linked.symlink_to(log_path)

    assert logs_module._log_lock_path(linked) == logs_module._log_lock_path(log_path)


def test_separate_process_appends_without_expected_fingerprint_preserve_both(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "campaign.csv"
    pd.DataFrame(columns=canonical_columns(config())).to_csv(log_path, index=False)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_append_in_process,
            args=(str(log_path), row_id, barrier, results),
        )
        for row_id in ("process_1", "process_2")
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    outcomes = sorted(results.get(timeout=2) for _ in processes)
    assert outcomes == [("process_1", "ok", ""), ("process_2", "ok", "")]
    written = pd.read_csv(log_path, keep_default_na=False)
    assert sorted(written["row_id"].tolist()) == ["process_1", "process_2"]


def test_same_process_threads_serialize_appends_without_lost_rows(tmp_path: Path) -> None:
    log_path = tmp_path / "campaign.csv"
    pd.DataFrame(columns=canonical_columns(config())).to_csv(log_path, index=False)
    barrier = threading.Barrier(2)

    def append(row_id: str, x: float) -> None:
        barrier.wait(timeout=5)
        append_suggestions(log_path, suggestion(row_id, x=x), config=config())

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(append, "thread_1", 0.2),
            executor.submit(append, "thread_2", 0.8),
        ]
        for future in futures:
            future.result(timeout=10)

    written = pd.read_csv(log_path, keep_default_na=False)
    assert sorted(written["row_id"].tolist()) == ["thread_1", "thread_2"]


def test_log_lock_directory_is_stable_across_tmpdir_environments(tmp_path: Path) -> None:
    script = (
        "from bo_forge.logs import _log_lock_path; "
        "print(_log_lock_path('/private/tmp/shared-campaign.csv'))"
    )
    outputs = []
    for temporary_directory in (tmp_path / "one", tmp_path / "two"):
        temporary_directory.mkdir()
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "TMPDIR": str(temporary_directory)},
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(completed.stdout.strip())

    assert outputs[0] == outputs[1]


def test_log_lock_releases_after_validation_and_write_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config()
    log_path = tmp_path / "campaign.csv"
    suggestion("existing", x=0.2).to_csv(log_path, index=False)

    with pytest.raises(LogValidationError, match="same design"):
        append_suggestions(log_path, suggestion("duplicate", x=0.2), config=cfg)

    real_atomic_write = logs_module._atomic_write_and_validate
    monkeypatch.setattr(
        logs_module,
        "_atomic_write_and_validate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(LogWriteError("write failed")),
    )
    with pytest.raises(LogWriteError, match="write failed"):
        append_suggestions(log_path, suggestion("write_failure", x=0.5), config=cfg)

    monkeypatch.setattr(logs_module, "_atomic_write_and_validate", real_atomic_write)
    append_suggestions(log_path, suggestion("after_failures", x=0.8), config=cfg)
    assert pd.read_csv(log_path, keep_default_na=False)["row_id"].tolist() == [
        "existing",
        "after_failures",
    ]


def test_log_lock_releases_after_post_write_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config()
    log_path = tmp_path / "campaign.csv"
    suggestion("existing", x=0.2).to_csv(log_path, index=False)
    real_read_csv = logs_module._read_csv
    canonical_reads = 0

    def fail_post_write_read(path: Path) -> pd.DataFrame:
        nonlocal canonical_reads
        if path == log_path.resolve():
            canonical_reads += 1
            if canonical_reads == 2:
                raise LogWriteError("post-write read failed")
        return real_read_csv(path)

    monkeypatch.setattr(logs_module, "_read_csv", fail_post_write_read)
    with pytest.raises(LogWriteError, match="Post-write validation failed"):
        append_suggestions(log_path, suggestion("written_before_failure", x=0.5), config=cfg)

    monkeypatch.setattr(logs_module, "_read_csv", real_read_csv)
    append_suggestions(log_path, suggestion("after_post_failure", x=0.8), config=cfg)
    assert pd.read_csv(log_path, keep_default_na=False)["row_id"].tolist() == [
        "existing",
        "written_before_failure",
        "after_post_failure",
    ]
