import builtins
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

import bo_forge.suggestions as suggestions_module
from bo_forge.config import CampaignConfig
from bo_forge.transforms import values_to_unit_cube
from bo_forge_app import api_cli
from bo_forge_app.api import create_app
from bo_forge_app.streamlit_helpers import file_fingerprint, make_staged_suggestion_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def copy_campaign(root: Path, config_name: str, log_name: str) -> dict[str, str]:
    config_dir = root / "configs"
    log_dir = root / "examples"
    config_dir.mkdir()
    log_dir.mkdir()
    shutil.copyfile(PROJECT_ROOT / "configs" / config_name, config_dir / config_name)
    shutil.copyfile(PROJECT_ROOT / "examples" / log_name, log_dir / log_name)
    return {"config_path": f"configs/{config_name}", "log_path": f"examples/{log_name}"}


def copy_suggestions(root: Path, name: str) -> pd.DataFrame:
    source = PROJECT_ROOT / "examples" / name
    destination = root / "examples" / name
    shutil.copyfile(source, destination)
    return pd.read_csv(destination, keep_default_na=False)


def client(root: Path) -> TestClient:
    return TestClient(create_app(root))


def staged_bundle_payload(root: Path, ref: dict[str, str], suggestions: pd.DataFrame) -> dict:
    bundle = make_staged_suggestion_bundle(
        suggestions,
        root / ref["config_path"],
        root / ref["log_path"],
    )
    return {
        "suggestions": {
            "columns": suggestions.columns.astype(str).tolist(),
            "records": suggestions.to_dict(orient="records"),
        },
        "suggestions_fingerprint": str(bundle["suggestions_fingerprint"]),
        "config_path": ref["config_path"],
        "config_fingerprint": str(bundle["config_fingerprint"]),
        "log_path": ref["log_path"],
        "log_fingerprint": str(bundle["log_fingerprint"]),
        "appended": False,
    }


def append_payload(
    api_client: TestClient,
    ref: dict[str, str],
    bundle: dict,
) -> dict:
    response = api_client.post(
        "/campaign/suggestions/append",
        json={**ref, "staged_bundle": bundle},
    )
    assert response.status_code == 200, response.text
    return response.json()


def current_log_fingerprint(root: Path, ref: dict[str, str]) -> str:
    return file_fingerprint(root / ref["log_path"])


def test_api_health(tmp_path: Path) -> None:
    response = client(tmp_path).get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"] == "2.2.0"
    assert payload["experimental"] is True


def test_api_validation_success_and_failure(tmp_path: Path) -> None:
    ref = copy_campaign(
        tmp_path,
        "01_simple_2d_maximise_logei.yaml",
        "01_simple_2d_maximise_logei_campaign_log.csv",
    )
    api_client = client(tmp_path)

    response = api_client.post("/campaign/validation", json=ref)
    assert response.status_code == 200
    assert response.json()["validation"]["ok"] is True
    assert response.json()["log_fingerprint"]

    (tmp_path / ref["log_path"]).write_text("not,a,campaign\n", encoding="utf-8")
    response = api_client.post("/campaign/validation", json=ref)
    assert response.status_code == 200
    payload = response.json()
    assert payload["validation"]["ok"] is False
    assert payload["validation"]["label"] == "Validation issue"


def test_api_summary_is_json_safe_and_read_only(tmp_path: Path) -> None:
    ref = copy_campaign(
        tmp_path,
        "01_simple_2d_maximise_logei.yaml",
        "01_simple_2d_maximise_logei_campaign_log.csv",
    )
    log_path = tmp_path / ref["log_path"]
    before = log_path.read_bytes()

    response = client(tmp_path).post("/campaign/summary", json=ref)

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["columns"] == ["field", "value"]
    assert "records" in payload["observed"]
    assert log_path.read_bytes() == before


def test_api_validation_and_summary_accept_multi_fidelity_example(tmp_path: Path) -> None:
    ref = copy_campaign(
        tmp_path,
        "15_multi_fidelity_qmfkg.yaml",
        "15_multi_fidelity_qmfkg_campaign_log.csv",
    )
    log_path = tmp_path / ref["log_path"]
    before = log_path.read_bytes()
    api_client = client(tmp_path)

    validation = api_client.post("/campaign/validation", json=ref)
    summary = api_client.post("/campaign/summary", json=ref)

    assert validation.status_code == 200
    assert validation.json()["validation"]["ok"] is True
    assert summary.status_code == 200
    assert summary.json()["summary"]["columns"] == ["field", "value"]
    assert log_path.read_bytes() == before


def test_api_dry_run_returns_staged_bundle_without_mutating(tmp_path: Path) -> None:
    ref = copy_campaign(
        tmp_path,
        "01_simple_2d_maximise_logei.yaml",
        "01_simple_2d_maximise_logei_campaign_log.csv",
    )
    log_path = tmp_path / ref["log_path"]
    before = log_path.read_bytes()

    response = client(tmp_path).post(
        "/campaign/suggestions/dry-run",
        json={**ref, "batch_size": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["suggestions"]["records"]
    assert payload["quality"]["records"]
    assert payload["staged_bundle"]["config_path"] == ref["config_path"]
    assert payload["staged_bundle"]["log_path"] == ref["log_path"]
    assert log_path.read_bytes() == before


def test_api_qlog_nei_dry_run_and_append_use_pending_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = copy_campaign(
        tmp_path,
        "18_noisy_pending_qlognei.yaml",
        "18_noisy_pending_qlognei_campaign_log.csv",
    )
    cfg = CampaignConfig.from_yaml(tmp_path / ref["config_path"])
    candidate = values_to_unit_cube(cfg, [(0.50, 92.0)])
    captured: dict[str, object] = {}

    def fake_optimizer(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        captured["x_pending"] = kwargs["x_pending"]
        return candidate, torch.tensor(0.25, dtype=torch.double), "qlog_nei"

    monkeypatch.setattr(suggestions_module, "optimize_qlog_nei", fake_optimizer)
    api_client = client(tmp_path)
    log_path = tmp_path / ref["log_path"]
    before = log_path.read_bytes()

    dry_run = api_client.post(
        "/campaign/suggestions/dry-run",
        json={**ref, "batch_size": 1},
    )

    assert dry_run.status_code == 200, dry_run.text
    payload = dry_run.json()
    x_pending = captured["x_pending"]
    assert isinstance(x_pending, torch.Tensor)
    assert x_pending.shape == (1, 2)
    assert payload["suggestions"]["records"][0]["source"] == "qlog_nei"
    assert payload["staged_bundle"]["suggestions"]["records"][0]["source"] == "qlog_nei"
    assert log_path.read_bytes() == before

    append = api_client.post(
        "/campaign/suggestions/append",
        json={**ref, "staged_bundle": payload["staged_bundle"]},
    )

    assert append.status_code == 200, append.text
    assert append.json()["validation"]["ok"] is True
    assert log_path.read_bytes() != before


def test_api_contextual_dry_run_accepts_context_values_without_mutating(
    tmp_path: Path,
) -> None:
    ref = copy_campaign(
        tmp_path,
        "16_contextual_logei.yaml",
        "16_contextual_logei_campaign_log.csv",
    )
    log_path = tmp_path / ref["log_path"]
    before = log_path.read_bytes()

    response = client(tmp_path).post(
        "/campaign/suggestions/dry-run",
        json={
            **ref,
            "batch_size": 1,
            "context_values": {"feedstock_acidity": 0.25},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["staged_bundle"]["context_values"] == {"feedstock_acidity": 0.25}
    assert payload["suggestions"]["records"][0]["feedstock_acidity"] == 0.25
    assert log_path.read_bytes() == before


def test_api_append_valid_bundle_mutates_through_service(tmp_path: Path) -> None:
    ref = copy_campaign(
        tmp_path,
        "01_simple_2d_maximise_logei.yaml",
        "01_simple_2d_maximise_logei_campaign_log.csv",
    )
    api_client = client(tmp_path)
    log_path = tmp_path / ref["log_path"]
    dry_run = api_client.post(
        "/campaign/suggestions/dry-run",
        json={**ref, "batch_size": 1},
    ).json()
    before = log_path.read_bytes()

    response = api_client.post(
        "/campaign/suggestions/append",
        json={**ref, "staged_bundle": dry_run["staged_bundle"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["validation"]["ok"] is True
    assert payload["appended_fingerprint"] == dry_run["staged_bundle"]["suggestions_fingerprint"]
    assert log_path.read_bytes() != before


def test_api_append_tampered_bundle_fails_without_mutation(tmp_path: Path) -> None:
    ref = copy_campaign(
        tmp_path,
        "01_simple_2d_maximise_logei.yaml",
        "01_simple_2d_maximise_logei_campaign_log.csv",
    )
    api_client = client(tmp_path)
    log_path = tmp_path / ref["log_path"]
    dry_run = api_client.post(
        "/campaign/suggestions/dry-run",
        json={**ref, "batch_size": 1},
    ).json()
    dry_run["staged_bundle"]["suggestions"]["records"][0]["row_id"] = "tampered"
    before = log_path.read_bytes()

    response = api_client.post(
        "/campaign/suggestions/append",
        json={**ref, "staged_bundle": dry_run["staged_bundle"]},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["ok"] is False
    assert "Staged suggestions changed" in payload["error"]["message"]
    assert log_path.read_bytes() == before


def test_api_append_tampered_context_metadata_fails_without_mutation(
    tmp_path: Path,
) -> None:
    ref = copy_campaign(
        tmp_path,
        "16_contextual_logei.yaml",
        "16_contextual_logei_campaign_log.csv",
    )
    api_client = client(tmp_path)
    log_path = tmp_path / ref["log_path"]
    dry_run = api_client.post(
        "/campaign/suggestions/dry-run",
        json={
            **ref,
            "batch_size": 1,
            "context_values": {"feedstock_acidity": 0.25},
        },
    ).json()
    dry_run["staged_bundle"]["context_values"] = {"feedstock_acidity": 0.75}
    before = log_path.read_bytes()

    response = api_client.post(
        "/campaign/suggestions/append",
        json={**ref, "staged_bundle": dry_run["staged_bundle"]},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["ok"] is False
    assert "Context values changed" in payload["error"]["message"]
    assert log_path.read_bytes() == before


@pytest.mark.parametrize("path_field", ["config_path", "log_path"])
def test_api_append_staged_bundle_path_escape_fails_without_mutation(
    tmp_path: Path,
    path_field: str,
) -> None:
    ref = copy_campaign(
        tmp_path,
        "01_simple_2d_maximise_logei.yaml",
        "01_simple_2d_maximise_logei_campaign_log.csv",
    )
    api_client = client(tmp_path)
    log_path = tmp_path / ref["log_path"]
    dry_run = api_client.post(
        "/campaign/suggestions/dry-run",
        json={**ref, "batch_size": 1},
    ).json()
    dry_run["staged_bundle"][path_field] = "../outside.csv"
    before = log_path.read_bytes()

    response = api_client.post(
        "/campaign/suggestions/append",
        json={**ref, "staged_bundle": dry_run["staged_bundle"]},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "path_outside_root"
    assert log_path.read_bytes() == before


def test_api_review_works_and_stale_fingerprint_fails(tmp_path: Path) -> None:
    ref = copy_campaign(
        tmp_path,
        "07_cost_aware_human_review_logei.yaml",
        "07_cost_aware_human_review_campaign_log.csv",
    )
    suggestions = copy_suggestions(tmp_path, "07_cost_aware_human_review_latest_suggestions.csv")
    bundle = staged_bundle_payload(tmp_path, ref, suggestions.head(1))
    api_client = client(tmp_path)
    append_payload(api_client, ref, bundle)
    row_id = str(suggestions.loc[0, "row_id"])
    log_path = tmp_path / ref["log_path"]
    before = log_path.read_bytes()

    stale = api_client.post(
        "/campaign/review",
        json={**ref, "row_id": row_id, "decision": "accept", "expected_log_fingerprint": "old"},
    )
    assert stale.status_code == 400
    assert stale.json()["error"]["code"] == "stale_log"
    assert log_path.read_bytes() == before

    current = file_fingerprint(log_path)
    response = api_client.post(
        "/campaign/review",
        json={
            **ref,
            "row_id": row_id,
            "decision": "accept",
            "note": "ready",
            "expected_log_fingerprint": current,
        },
    )
    assert response.status_code == 200
    assert response.json()["validation"]["ok"] is True


def test_api_mutation_endpoints_require_expected_log_fingerprint(tmp_path: Path) -> None:
    ref = copy_campaign(
        tmp_path,
        "07_cost_aware_human_review_logei.yaml",
        "07_cost_aware_human_review_campaign_log.csv",
    )
    api_client = client(tmp_path)

    review = api_client.post(
        "/campaign/review",
        json={**ref, "row_id": "suggested", "decision": "accept"},
    )
    observation = api_client.post(
        "/campaign/observations",
        json={**ref, "row_id": "suggested", "objective_value": 1.0},
    )

    for response in [review, observation]:
        assert response.status_code == 422
        payload = response.json()
        assert payload["ok"] is False
        assert payload["error"]["code"] == "request_validation"
        assert "expected_log_fingerprint" in response.text


def test_api_mark_observed_single_objective_with_actual_cost(tmp_path: Path) -> None:
    ref = copy_campaign(
        tmp_path,
        "07_cost_aware_human_review_logei.yaml",
        "07_cost_aware_human_review_campaign_log.csv",
    )
    suggestions = copy_suggestions(tmp_path, "07_cost_aware_human_review_latest_suggestions.csv")
    bundle = staged_bundle_payload(tmp_path, ref, suggestions.head(1))
    api_client = client(tmp_path)
    append_payload(api_client, ref, bundle)
    row_id = str(suggestions.loc[0, "row_id"])
    review = api_client.post(
        "/campaign/review",
        json={
            **ref,
            "row_id": row_id,
            "decision": "accept",
            "expected_log_fingerprint": current_log_fingerprint(tmp_path, ref),
        },
    )
    assert review.status_code == 200

    response = api_client.post(
        "/campaign/observations",
        json={
            **ref,
            "row_id": row_id,
            "objective_value": 71.2,
            "actual_cost": 2.4,
            "expected_log_fingerprint": review.json()["log_fingerprint"],
        },
    )

    assert response.status_code == 200
    observed = pd.read_csv(tmp_path / ref["log_path"], keep_default_na=False)
    row = observed.loc[observed["row_id"] == row_id].iloc[0]
    assert row["status"] == "observed"
    assert float(row["yield_score"]) == pytest.approx(71.2)
    assert float(row["cost_actual"]) == pytest.approx(2.4)


def test_api_mark_observed_multi_objective_and_partial_failure(
    tmp_path: Path,
) -> None:
    ref = copy_campaign(
        tmp_path,
        "10_multi_objective_mixed_constrained_qlogehvi.yaml",
        "10_multi_objective_mixed_constrained_campaign_log.csv",
    )
    suggestions = copy_suggestions(
        tmp_path,
        "10_multi_objective_mixed_constrained_latest_suggestions.csv",
    )
    bundle = staged_bundle_payload(tmp_path, ref, suggestions.head(1))
    api_client = client(tmp_path)
    append_payload(api_client, ref, bundle)
    row_id = str(suggestions.loc[0, "row_id"])
    log_path = tmp_path / ref["log_path"]
    before = log_path.read_bytes()

    failed = api_client.post(
        "/campaign/observations",
        json={
            **ref,
            "row_id": row_id,
            "objective_values": {"yield_score": 70.0},
            "expected_log_fingerprint": current_log_fingerprint(tmp_path, ref),
        },
    )
    assert failed.status_code == 400
    assert "objective_values keys must exactly match" in failed.json()["error"]["message"]
    assert log_path.read_bytes() == before

    response = api_client.post(
        "/campaign/observations",
        json={
            **ref,
            "row_id": row_id,
            "objective_values": {"yield_score": 70.0, "waste_score": 15.0},
            "expected_log_fingerprint": current_log_fingerprint(tmp_path, ref),
        },
    )
    assert response.status_code == 200
    observed = pd.read_csv(log_path, keep_default_na=False)
    row = observed.loc[observed["row_id"] == row_id].iloc[0]
    assert row["status"] == "observed"
    assert float(row["yield_score"]) == pytest.approx(70.0)
    assert float(row["waste_score"]) == pytest.approx(15.0)


def test_api_mark_observed_multi_objective_actual_cost(tmp_path: Path) -> None:
    ref = copy_campaign(
        tmp_path,
        "12_cost_aware_multi_objective_qlogehvi.yaml",
        "12_cost_aware_multi_objective_campaign_log.csv",
    )
    suggestions = copy_suggestions(tmp_path, "12_cost_aware_multi_objective_latest_suggestions.csv")
    bundle = staged_bundle_payload(tmp_path, ref, suggestions.head(1))
    api_client = client(tmp_path)
    append_payload(api_client, ref, bundle)
    row_id = str(suggestions.loc[0, "row_id"])
    review = api_client.post(
        "/campaign/review",
        json={
            **ref,
            "row_id": row_id,
            "decision": "accept",
            "expected_log_fingerprint": current_log_fingerprint(tmp_path, ref),
        },
    )
    assert review.status_code == 200

    response = api_client.post(
        "/campaign/observations",
        json={
            **ref,
            "row_id": row_id,
            "objective_values": {"yield": 0.71, "selectivity": 0.62, "waste": 0.33},
            "actual_cost": 2.1,
            "expected_log_fingerprint": review.json()["log_fingerprint"],
        },
    )

    assert response.status_code == 200
    observed = pd.read_csv(tmp_path / ref["log_path"], keep_default_na=False)
    row = observed.loc[observed["row_id"] == row_id].iloc[0]
    assert row["status"] == "observed"
    assert float(row["cost_actual"]) == pytest.approx(2.1)


def test_api_observation_stale_fingerprint_fails_without_mutation(tmp_path: Path) -> None:
    ref = copy_campaign(
        tmp_path,
        "10_multi_objective_mixed_constrained_qlogehvi.yaml",
        "10_multi_objective_mixed_constrained_campaign_log.csv",
    )
    suggestions = copy_suggestions(
        tmp_path,
        "10_multi_objective_mixed_constrained_latest_suggestions.csv",
    )
    bundle = staged_bundle_payload(tmp_path, ref, suggestions.head(1))
    api_client = client(tmp_path)
    append_payload(api_client, ref, bundle)
    row_id = str(suggestions.loc[0, "row_id"])
    log_path = tmp_path / ref["log_path"]
    before = log_path.read_bytes()

    response = api_client.post(
        "/campaign/observations",
        json={
            **ref,
            "row_id": row_id,
            "objective_values": {"yield_score": 70.0, "waste_score": 15.0},
            "expected_log_fingerprint": "stale",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "stale_log"
    assert log_path.read_bytes() == before


def test_api_rejects_absolute_and_outside_root_paths(tmp_path: Path) -> None:
    api_client = client(tmp_path)
    absolute = api_client.post(
        "/campaign/summary",
        json={
            "config_path": str(PROJECT_ROOT / "configs" / "01_simple_2d_maximise_logei.yaml"),
            "log_path": "log.csv",
        },
    )
    outside = api_client.post(
        "/campaign/summary",
        json={"config_path": "../config.yaml", "log_path": "log.csv"},
    )

    assert absolute.status_code == 400
    assert outside.status_code == 400
    assert absolute.json()["error"]["code"] == "path_outside_root"
    assert outside.json()["error"]["code"] == "path_outside_root"


def test_api_request_errors_are_structured_json(tmp_path: Path) -> None:
    response = client(tmp_path).post("/campaign/summary", json={"config_path": "only.yaml"})

    assert response.status_code == 422
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error"]["code"] == "request_validation"
    assert "Traceback" not in response.text


def test_api_cli_help_without_importing_api_dependencies() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "bo_forge_app.api_cli", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "bo-forge-api" in completed.stdout


def test_api_cli_missing_dependencies_show_install_hint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delitem(sys.modules, "bo_forge_app.api", raising=False)
    real_import = builtins.__import__

    def block_fastapi(name: str, *args: object, **kwargs: object) -> object:
        if name == "fastapi" or name.startswith("fastapi."):
            raise ModuleNotFoundError("No module named 'fastapi'", name="fastapi")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_fastapi)

    exit_code = api_cli.run(["--root", str(tmp_path)])

    assert exit_code == 1
    assert 'pip install "bo-forge[api]"' in capsys.readouterr().err


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_api_launcher_host_warning_skips_loopback(host: str) -> None:
    assert not api_cli._host_requires_network_warning(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "lab-server.local"])
def test_api_launcher_host_warning_flags_network_hosts(host: str) -> None:
    assert api_cli._host_requires_network_warning(host)


def test_api_launcher_startup_message_warns_for_network_host(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = api_cli.parse_args(["--root", str(tmp_path), "--host", "0.0.0.0"])

    api_cli.print_startup_messages(args, tmp_path)

    output = capsys.readouterr().out
    assert "no built-in authentication" in output
    assert "Do not expose this API directly to the public internet." in output
