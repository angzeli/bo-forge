"""No-overwrite predictive bundles: rollback, retry, and concurrent publication."""

import errno
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import Mock

import pandas as pd
import pytest

import bo_forge.predictive as predictive
from bo_forge._campaign import provenance_fork
from bo_forge._filesystem import _DirectoryPublicationUnavailable
from bo_forge.errors import ProvenanceError


@pytest.fixture
def result():
    return predictive.PredictiveEvaluationResult(
        pd.DataFrame([{"model_profile": "default", "fit_status": "complete", "rmse": 0.5}]),
        pd.DataFrame([{"row_id": "one", "predicted_mean": 1.0}]),
        pd.DataFrame([{"fold": 1, "fit_status": "complete"}]),
        {"z": 2, "a": "original objective units"},
    )


def _bundle(path):
    return {file.name: file.read_bytes() for file in path.iterdir()}


def test_export_preserves_formats_return_path_and_missing_parents(result, tmp_path):
    destination = tmp_path / "missing" / "evaluation"
    assert result.export(destination) == destination
    assert _bundle(destination) == {
        **{f"{name}.csv": getattr(result, name).to_csv(index=False).encode()
           for name in ("summary", "predictions", "fold_outcomes")},
        "metadata.json": (json.dumps(result.metadata, sort_keys=True, indent=2) + "\n").encode(),
    }
    assert list(destination.parent.iterdir()) == [destination]


@pytest.mark.parametrize("name", ["a" * 240, "b" * 255, "\u6d4b" * 85],
                         ids=["ascii-long", "ascii-limit", "multibyte-limit"])
def test_long_destination_names_export_and_retry_without_overwrite(
    result, tmp_path, monkeypatch, name,
):
    destination = tmp_path / name
    destination.mkdir()
    destination.rmdir()
    with monkeypatch.context() as patch:
        patch.setattr(pd.DataFrame, "to_csv", Mock(side_effect=OSError("serialization failed")))
        with pytest.raises(OSError, match="serialization failed"):
            result.export(destination)
    assert list(tmp_path.iterdir()) == []
    assert result.export(destination) == destination
    before = _bundle(destination)
    assert set(before) == {"summary.csv", "predictions.csv", "fold_outcomes.csv", "metadata.json"}
    with pytest.raises(FileExistsError):
        result.export(destination)
    assert _bundle(destination) == before
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("stage", ["summary.csv", "predictions.csv", "fold_outcomes.csv",
                                  "json_serialization", "json_write", "publication"])
def test_export_failure_leaves_no_final_bundle_and_result_can_retry(
    result, tmp_path, monkeypatch, stage,
):
    destination = tmp_path / "evaluation"
    before = [table.copy(deep=True) for table in
              (result.summary, result.predictions, result.fold_outcomes)]
    error = TypeError("cannot serialize metadata") if stage == "json_serialization" else (
        OSError(f"failed at {stage}")
    )
    to_csv, write_text = pd.DataFrame.to_csv, Path.write_text

    def csv(frame, path, **kwargs):
        to_csv(frame, path, **kwargs)
        if Path(path).name == stage:
            raise error

    def text(path, data, **kwargs):
        write_text(path, data[:3], **kwargs)
        raise error

    with monkeypatch.context() as patch:
        if stage.endswith(".csv"):
            patch.setattr(pd.DataFrame, "to_csv", csv)
        elif stage == "json_serialization":
            patch.setattr(predictive.json, "dumps", Mock(side_effect=error))
        elif stage == "json_write":
            patch.setattr(Path, "write_text", text)
        else:
            patch.setattr(predictive, "rename_directory_exclusive", Mock(side_effect=error))
        with pytest.raises(type(error), match=str(error)) as raised:
            result.export(destination)
        assert raised.value is error
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []
    for original, current in zip(
        before, (result.summary, result.predictions, result.fold_outcomes), strict=True,
    ):
        pd.testing.assert_frame_equal(original, current)
    result.export(destination)
    assert len(_bundle(destination)) == 4


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_existing_destination_precedes_serialization_and_is_untouched(
    result, tmp_path, monkeypatch, kind,
):
    destination = tmp_path / "evaluation"
    if kind == "directory":
        destination.mkdir()
    elif kind == "file":
        destination.write_bytes(b"original")
    else:
        destination.symlink_to(tmp_path / "missing")
    monkeypatch.setattr(pd.DataFrame, "to_csv", Mock(side_effect=AssertionError("must not write")))
    with pytest.raises(FileExistsError):
        result.export(destination)
    assert list(tmp_path.iterdir()) == [destination]
    if kind == "file":
        assert destination.read_bytes() == b"original"
    elif kind == "symlink":
        assert destination.readlink() == tmp_path / "missing"


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_destination_created_during_preparation_is_never_removed(
    result, tmp_path, monkeypatch, kind,
):
    destination = tmp_path / "evaluation"
    rename = predictive.rename_directory_exclusive

    def competing(source, target):
        if kind == "directory":
            target.mkdir()
        elif kind == "file":
            target.write_bytes(b"another exporter")
        else:
            target.symlink_to(tmp_path / "missing")
        rename(source, target)

    monkeypatch.setattr(predictive, "rename_directory_exclusive", competing)
    with pytest.raises(FileExistsError):
        result.export(destination)
    assert list(tmp_path.iterdir()) == [destination]
    if kind == "file":
        assert destination.read_bytes() == b"another exporter"
    elif kind == "symlink":
        assert destination.is_symlink()


def test_two_exporters_publish_exactly_one_complete_bundle(result, tmp_path, monkeypatch):
    destination = tmp_path / "evaluation"
    ready = Barrier(2)
    rename = predictive.rename_directory_exclusive

    def publish(source, target):
        assert len(_bundle(source)) == 4
        ready.wait(timeout=10)
        rename(source, target)

    def export():
        try:
            return result.export(destination)
        except FileExistsError:
            return "conflict"

    monkeypatch.setattr(predictive, "rename_directory_exclusive", publish)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: export(), range(2)))
    assert outcomes.count(destination) == outcomes.count("conflict") == 1
    assert list(tmp_path.iterdir()) == [destination]
    assert len(_bundle(destination)) == 4
    pd.testing.assert_frame_equal(pd.read_csv(destination / "summary.csv"), result.summary)


def test_cleanup_failure_preserves_original_error_and_allows_retry(result, tmp_path, monkeypatch):
    destination = tmp_path / "evaluation"
    original = ValueError("invalid JSON value")
    with monkeypatch.context() as patch:
        patch.setattr(predictive.json, "dumps", Mock(side_effect=original))
        patch.setattr(predictive.shutil, "rmtree", Mock(side_effect=PermissionError("cleanup")))
        with pytest.raises(ValueError) as raised:
            result.export(destination)
        assert raised.value is original
    assert not destination.exists()
    assert len(list(tmp_path.glob(".evaluation.preparing-*"))) == 1
    result.export(destination)
    assert len(_bundle(destination)) == 4


def test_success_does_not_cleanup_a_published_or_replaced_directory(result, tmp_path, monkeypatch):
    cleanup = Mock(side_effect=OSError("must not cleanup after publication"))
    monkeypatch.setattr(predictive.shutil, "rmtree", cleanup)
    result.export(tmp_path / "evaluation")
    cleanup.assert_not_called()


@pytest.mark.parametrize("error,message", [
    (FileExistsError(errno.EEXIST, "exists"), "Fork destination already exists."),
    (_DirectoryPublicationUnavailable(
        errno.ENOTSUP, "Atomic no-overwrite directory publication requires macOS or Linux.",
    ), "Atomic no-overwrite directory publication requires macOS or Linux."),
])
def test_shared_publication_preserves_provenance_errors(monkeypatch, tmp_path, error, message):
    monkeypatch.setattr(provenance_fork, "rename_directory_exclusive", Mock(
        side_effect=error,
    ))
    with pytest.raises(ProvenanceError, match=message):
        provenance_fork._rename_directory_exclusive(tmp_path / "from", tmp_path / "to")


def test_shared_publication_preserves_underlying_filesystem_error(monkeypatch, tmp_path):
    error = OSError(errno.ENOTSUP, "Operation not supported", str(tmp_path / "destination"))
    monkeypatch.setattr(provenance_fork, "rename_directory_exclusive", Mock(side_effect=error))
    with pytest.raises(OSError) as raised:
        provenance_fork._rename_directory_exclusive(tmp_path / "from", tmp_path / "to")
    assert raised.value is error
