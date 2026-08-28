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
from bo_forge.errors import SuggestionError
from bo_forge.session import CampaignSession
from bo_forge.transforms import values_to_unit_cube
from bo_forge_app import api as api_module
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
    assert payload["version"] == "3.0.1"
    assert payload["experimental"] is True
    assert payload["staging"]["active_stages"] == 0
    assert payload["staging"]["stage_ttl_seconds"] == pytest.approx(1800)
    assert payload["staging"]["max_staged_batches"] == 128
    assert payload["deployment"] == {
        "authentication": "none",
        "trusted_network_only": True,
        "stage_storage": "process_memory",
        "client_carried_bundles": True,
        "interactive_docs": True,
        "multi_worker_safe": False,
    }


def test_api_health_reports_strict_deployment_modes(tmp_path: Path) -> None:
    response = TestClient(
        create_app(tmp_path, server_stages_only=True, interactive_docs=False)
    ).get("/health")

    assert response.status_code == 200
    assert response.json()["deployment"] == {
        "authentication": "none",
        "trusted_network_only": True,
        "stage_storage": "process_memory",
        "client_carried_bundles": False,
        "interactive_docs": False,
        "multi_worker_safe": False,
    }


def test_api_no_docs_disables_all_documentation_routes_but_keeps_health(
    tmp_path: Path,
) -> None:
    api_client = TestClient(create_app(tmp_path, interactive_docs=False))

    assert api_client.get("/docs").status_code == 404
    assert api_client.get("/redoc").status_code == 404
    assert api_client.get("/openapi.json").status_code == 404
    assert api_client.get("/health").status_code == 200


def test_api_does_not_add_permissive_cors_headers(tmp_path: Path) -> None:
    response = client(tmp_path).get(
        "/health",
        headers={"Origin": "https://untrusted.example"},
    )

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_api_does_not_answer_cors_preflight_permissively(tmp_path: Path) -> None:
    response = client(tmp_path).options(
        "/health",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 405
    assert "access-control-allow-origin" not in response.headers


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
    assert payload["fidelity_summary"] == {"columns": [], "records": []}
    assert payload["fidelity_coverage"] == {"columns": [], "records": []}
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
    payload = summary.json()
    assert payload["summary"]["columns"] == ["field", "value"]
    assert payload["fidelity_summary"]["columns"] == ["field", "value"]
    assert payload["fidelity_coverage"]["columns"] == [
        "fidelity",
        "is_target",
        "modeled_evaluation_cost",
        "observed_rows",
        "active_suggestions",
        "objective_mean",
        "objective_std",
        "objective_best",
        "best_row_id",
        "latest_observed_iteration",
    ]
    assert payload["fidelity_coverage"]["records"]
    assert log_path.read_bytes() == before


def test_api_fidelity_coverage_serializes_sparse_values_as_json_null(
    tmp_path: Path,
) -> None:
    ref = copy_campaign(
        tmp_path,
        "22_discrete_multi_fidelity_qmfkg.yaml",
        "22_discrete_multi_fidelity_qmfkg_campaign_log.csv",
    )
    log_path = tmp_path / ref["log_path"]
    df = pd.read_csv(log_path, keep_default_na=False)
    target_only = df.loc[pd.to_numeric(df["fidelity"]) == 1.0]
    target_only.to_csv(log_path, index=False)
    before = log_path.read_bytes()

    response = client(tmp_path).post("/campaign/summary", json=ref)

    assert response.status_code == 200
    records = response.json()["fidelity_coverage"]["records"]
    unused_level = next(record for record in records if record["fidelity"] == 0.25)
    assert unused_level["objective_mean"] is None
    assert unused_level["objective_std"] is None
    assert unused_level["objective_best"] is None
    assert unused_level["best_row_id"] is None
    assert unused_level["latest_observed_iteration"] is None
    assert "NaN" not in response.text
    assert log_path.read_bytes() == before


def test_api_discrete_qmfkg_batch_dry_run_and_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = copy_campaign(
        tmp_path,
        "22_discrete_multi_fidelity_qmfkg.yaml",
        "22_discrete_multi_fidelity_qmfkg_campaign_log.csv",
    )
    cfg = CampaignConfig.from_yaml(tmp_path / ref["config_path"])
    candidates = values_to_unit_cube(cfg, [(0.2, 70.0, 0.5), (0.4, 100.0, 1.0)])

    class FakePosterior:
        mean = torch.tensor([[1.1], [1.4]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x_unit: torch.Tensor) -> FakePosterior:
            return FakePosterior()

    monkeypatch.setattr(
        suggestions_module,
        "fit_multi_fidelity_gp_model",
        lambda *_args, **_kwargs: FakeModel(),
    )
    monkeypatch.setattr(
        suggestions_module,
        "optimize_posterior_mean_at_target_fidelity",
        lambda **_kwargs: torch.tensor([1.0], dtype=torch.double),
    )
    monkeypatch.setattr(
        suggestions_module,
        "optimize_qmf_kg",
        lambda **_kwargs: (candidates, torch.tensor(0.4), "qmf_kg"),
    )
    log_path = tmp_path / ref["log_path"]
    before = log_path.read_bytes()
    api_client = client(tmp_path)

    dry_run = api_client.post(
        "/campaign/suggestions/dry-run",
        json={**ref, "batch_size": 2},
    )

    assert dry_run.status_code == 200, dry_run.text
    payload = dry_run.json()
    assert len(payload["suggestions"]["records"]) == 2
    assert log_path.read_bytes() == before
    append_payload(api_client, ref, payload["staged_bundle"])
    assert len(pd.read_csv(log_path, keep_default_na=False)) == 8


def test_api_discrete_qmfkg_rejects_batch_above_four_without_mutation(
    tmp_path: Path,
) -> None:
    ref = copy_campaign(
        tmp_path,
        "22_discrete_multi_fidelity_qmfkg.yaml",
        "22_discrete_multi_fidelity_qmfkg_campaign_log.csv",
    )
    log_path = tmp_path / ref["log_path"]
    before = log_path.read_bytes()
    api_client = client(tmp_path)

    response = api_client.post(
        "/campaign/suggestions/dry-run",
        json={**ref, "batch_size": 5},
    )

    assert response.status_code == 400
    assert "batch_size from 1 through 4" in response.json()["error"]["message"]
    assert log_path.read_bytes() == before


def test_api_translates_qmfkg_timeout_without_mutating_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = copy_campaign(
        tmp_path,
        "22_discrete_multi_fidelity_qmfkg.yaml",
        "22_discrete_multi_fidelity_qmfkg_campaign_log.csv",
    )
    log_path = tmp_path / ref["log_path"]
    before = log_path.read_bytes()

    def fail_suggest(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise SuggestionError("qMFKG acquisition optimization timed out")

    monkeypatch.setattr(CampaignSession, "suggest_next", fail_suggest)

    response = client(tmp_path).post(
        "/campaign/suggestions/dry-run",
        json={**ref, "batch_size": 1},
    )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "code": "bo_forge_error",
        "message": "qMFKG acquisition optimization timed out",
        "retryable": False,
        "suggested_action": "Correct the campaign state or request before retrying.",
    }
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
    assert payload["stage"]["status"] == "active"
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


def test_api_qlog_nehvi_dry_run_and_append_use_pending_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = copy_campaign(
        tmp_path,
        "19_multi_objective_qlognehvi.yaml",
        "19_multi_objective_qlognehvi_campaign_log.csv",
    )
    cfg = CampaignConfig.from_yaml(tmp_path / ref["config_path"])
    candidate = values_to_unit_cube(cfg, [(72.0, "MeCN")])
    captured: dict[str, object] = {}

    def fake_optimizer(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        captured["x_pending"] = kwargs["x_pending"]
        return candidate, torch.tensor(0.25, dtype=torch.double), "qlog_nehvi"

    monkeypatch.setattr(suggestions_module, "optimize_qlog_nehvi", fake_optimizer)
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
    assert x_pending.shape == (1, candidate.shape[1])
    assert payload["suggestions"]["records"][0]["source"] == "qlog_nehvi"
    assert (
        payload["staged_bundle"]["suggestions"]["records"][0]["source"]
        == "qlog_nehvi"
    )
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
    api_client = client(tmp_path)

    response = api_client.post(
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
    assert payload["stage"]["context_values"] == {"feedstock_acidity": 0.25}
    recovered = api_client.get(
        f"/campaign/stages/{payload['stage']['stage_id']}"
    )
    assert recovered.status_code == 200
    assert recovered.json()["stage"]["context_values"] == {
        "feedstock_acidity": 0.25
    }
    listed = api_client.get("/campaign/stages").json()["stages"]
    listed_stage = next(
        item for item in listed if item["stage_id"] == payload["stage"]["stage_id"]
    )
    assert listed_stage["context_variable_names"] == ["feedstock_acidity"]
    assert "context_values" not in listed_stage
    assert log_path.read_bytes() == before


def test_api_contextual_stage_metadata_uses_resolved_default_values(
    tmp_path: Path,
) -> None:
    ref = copy_campaign(
        tmp_path,
        "16_contextual_logei.yaml",
        "16_contextual_logei_campaign_log.csv",
    )
    api_client = client(tmp_path)

    response = api_client.post(
        "/campaign/suggestions/dry-run",
        json={**ref, "batch_size": 1},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["suggestions"]["records"][0]["feedstock_acidity"] == pytest.approx(
        0.5
    )
    assert payload["stage"]["context_values"] == {"feedstock_acidity": 0.5}
    listing = api_client.get("/campaign/stages").json()["stages"]
    listed_stage = next(
        item for item in listing if item["stage_id"] == payload["stage"]["stage_id"]
    )
    assert listed_stage["context_variable_names"] == ["feedstock_acidity"]


def test_api_contextual_replicate_dry_run_is_context_matched_and_non_mutating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = copy_campaign(
        tmp_path,
        "21_contextual_replicate_logei.yaml",
        "21_contextual_replicate_campaign_log.csv",
    )
    log_path = tmp_path / ref["log_path"]
    before = log_path.read_bytes()

    class FakePosterior:
        mean = torch.tensor([[2.0], [1.0], [10.0], [0.0]], dtype=torch.double)
        variance = torch.full((4, 1), 0.04, dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())

    response = client(tmp_path).post(
        "/campaign/suggestions/dry-run",
        json={
            **ref,
            "batch_size": 1,
            "context_values": {"feedstock_acidity": 0.25},
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    suggestion = payload["suggestions"]["records"][0]
    assert payload["staged_bundle"]["context_values"] == {"feedstock_acidity": 0.25}
    assert suggestion["feedstock_acidity"] == 0.25
    assert suggestion["replicate_group"] == "group_acid25_best"
    assert suggestion["replicate_index"] == 2
    assert log_path.read_bytes() == before


def test_api_contextual_replicate_review_cost_round_trip_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = copy_campaign(
        tmp_path,
        "21_contextual_replicate_logei.yaml",
        "21_contextual_replicate_campaign_log.csv",
    )
    api_client = client(tmp_path)
    log_path = tmp_path / ref["log_path"]

    class FakePosterior:
        mean = torch.tensor([[2.0], [1.0], [10.0], [0.0]], dtype=torch.double)
        variance = torch.full((4, 1), 0.04, dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    before_dry_run = log_path.read_bytes()
    dry_run = api_client.post(
        "/campaign/suggestions/dry-run",
        json={
            **ref,
            "batch_size": 1,
            "context_values": {"feedstock_acidity": 0.25},
        },
    )
    assert dry_run.status_code == 200, dry_run.text
    payload = dry_run.json()
    suggestion = payload["suggestions"]["records"][0]
    assert log_path.read_bytes() == before_dry_run

    append = api_client.post(
        "/campaign/suggestions/append",
        json={**ref, "staged_bundle": payload["staged_bundle"]},
    )
    assert append.status_code == 200, append.text
    row_id = str(suggestion["row_id"])
    review = api_client.post(
        "/campaign/review",
        json={
            **ref,
            "row_id": row_id,
            "decision": "accept",
            "note": "approved",
            "expected_log_fingerprint": append.json()["log_fingerprint"],
        },
    )
    assert review.status_code == 200, review.text
    before_invalid_observation = log_path.read_bytes()
    invalid_observation = api_client.post(
        "/campaign/observations",
        json={
            **ref,
            "row_id": row_id,
            "objective_value": 0.91,
            "actual_cost": "NaN",
            "expected_log_fingerprint": review.json()["log_fingerprint"],
        },
    )
    assert invalid_observation.status_code in {400, 422}
    assert log_path.read_bytes() == before_invalid_observation

    observed = api_client.post(
        "/campaign/observations",
        json={
            **ref,
            "row_id": row_id,
            "objective_value": 0.91,
            "actual_cost": 4.0,
            "expected_log_fingerprint": review.json()["log_fingerprint"],
        },
    )
    assert observed.status_code == 200, observed.text
    row = pd.read_csv(log_path, keep_default_na=False).query("row_id == @row_id").iloc[0]
    assert row["status"] == "observed"
    assert row["replicate_group"] == "group_acid25_best"
    assert int(row["replicate_index"]) == 2
    assert float(row["cost_actual"]) == pytest.approx(4.0)


def test_api_contextual_cost_review_round_trip_with_actual_cost(
    tmp_path: Path,
) -> None:
    ref = copy_campaign(
        tmp_path,
        "20_contextual_cost_review_logei.yaml",
        "20_contextual_cost_review_campaign_log.csv",
    )
    api_client = client(tmp_path)
    log_path = tmp_path / ref["log_path"]
    before = log_path.read_bytes()

    dry_run = api_client.post(
        "/campaign/suggestions/dry-run",
        json={
            **ref,
            "batch_size": 1,
            "context_values": {"feedstock_acidity": 0.5},
        },
    )
    assert dry_run.status_code == 200, dry_run.text
    payload = dry_run.json()
    suggestion = payload["suggestions"]["records"][0]
    assert suggestion["source"] == "cost_log_ei"
    assert suggestion["review_status"] == "pending"
    assert suggestion["feedstock_acidity"] == 0.5
    assert suggestion["cost_estimate"] > 0
    assert log_path.read_bytes() == before

    append = api_client.post(
        "/campaign/suggestions/append",
        json={**ref, "staged_bundle": payload["staged_bundle"]},
    )
    assert append.status_code == 200, append.text
    row_id = str(suggestion["row_id"])

    review = api_client.post(
        "/campaign/review",
        json={
            **ref,
            "row_id": row_id,
            "decision": "accept",
            "note": "approved",
            "expected_log_fingerprint": append.json()["log_fingerprint"],
        },
    )
    assert review.status_code == 200, review.text

    observed = api_client.post(
        "/campaign/observations",
        json={
            **ref,
            "row_id": row_id,
            "objective_value": 0.84,
            "actual_cost": 4.2,
            "expected_log_fingerprint": review.json()["log_fingerprint"],
        },
    )
    assert observed.status_code == 200, observed.text

    df = pd.read_csv(log_path, keep_default_na=False)
    row = df.loc[df["row_id"] == row_id].iloc[0]
    assert row["status"] == "observed"
    assert row["review_note"] == "approved"
    assert float(row["yield_score"]) == pytest.approx(0.84)
    assert float(row["cost_actual"]) == pytest.approx(4.2)


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


def test_api_contextual_cost_append_rejects_changed_budget_log_without_mutation(
    tmp_path: Path,
) -> None:
    ref = copy_campaign(
        tmp_path,
        "20_contextual_cost_review_logei.yaml",
        "20_contextual_cost_review_campaign_log.csv",
    )
    api_client = client(tmp_path)
    log_path = tmp_path / ref["log_path"]
    dry_run_response = api_client.post(
        "/campaign/suggestions/dry-run",
        json={
            **ref,
            "batch_size": 1,
            "context_values": {"feedstock_acidity": 0.5},
        },
    )
    assert dry_run_response.status_code == 200, dry_run_response.text
    dry_run = dry_run_response.json()

    changed = pd.read_csv(log_path, keep_default_na=False)
    changed.loc[changed["row_id"] == "ctx_cost_seed_0", "cost_actual"] = 80.0
    changed.to_csv(log_path, index=False)
    before_failed_append = log_path.read_bytes()

    response = api_client.post(
        "/campaign/suggestions/append",
        json={**ref, "staged_bundle": dry_run["staged_bundle"]},
    )

    assert response.status_code == 400
    assert "Log file changed after suggestions were staged" in response.json()["error"][
        "message"
    ]
    assert log_path.read_bytes() == before_failed_append


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


def test_api_review_works_and_stale_fingerprint_fails(
    tmp_path: Path,
    suggestion_fixture,
) -> None:
    ref = copy_campaign(
        tmp_path,
        "07_cost_aware_human_review_logei.yaml",
        "07_cost_aware_human_review_campaign_log.csv",
    )
    suggestions = suggestion_fixture("07_cost_aware_human_review_latest_suggestions.csv")
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
    stale_error = stale.json()["error"]
    assert stale_error["code"] == "stale_log"
    assert stale_error["retryable"] is False
    assert "new log fingerprint" in stale_error["suggested_action"]
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


def test_api_mark_observed_single_objective_with_actual_cost(
    tmp_path: Path,
    suggestion_fixture,
) -> None:
    ref = copy_campaign(
        tmp_path,
        "07_cost_aware_human_review_logei.yaml",
        "07_cost_aware_human_review_campaign_log.csv",
    )
    suggestions = suggestion_fixture("07_cost_aware_human_review_latest_suggestions.csv")
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
    suggestion_fixture,
) -> None:
    ref = copy_campaign(
        tmp_path,
        "10_multi_objective_mixed_constrained_qlogehvi.yaml",
        "10_multi_objective_mixed_constrained_campaign_log.csv",
    )
    suggestions = suggestion_fixture(
        "10_multi_objective_mixed_constrained_latest_suggestions.csv"
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


def test_api_mark_observed_multi_objective_actual_cost(
    tmp_path: Path,
    suggestion_fixture,
) -> None:
    ref = copy_campaign(
        tmp_path,
        "12_cost_aware_multi_objective_qlogehvi.yaml",
        "12_cost_aware_multi_objective_campaign_log.csv",
    )
    suggestions = suggestion_fixture("12_cost_aware_multi_objective_latest_suggestions.csv")
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


def test_api_observation_stale_fingerprint_fails_without_mutation(
    tmp_path: Path,
    suggestion_fixture,
) -> None:
    ref = copy_campaign(
        tmp_path,
        "10_multi_objective_mixed_constrained_qlogehvi.yaml",
        "10_multi_objective_mixed_constrained_campaign_log.csv",
    )
    suggestions = suggestion_fixture(
        "10_multi_objective_mixed_constrained_latest_suggestions.csv"
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
    for response in (absolute, outside):
        error = response.json()["error"]
        assert error["code"] == "path_outside_root"
        assert error["retryable"] is False
        assert "inside the configured API root" in error["suggested_action"]


def test_api_request_errors_are_structured_json(tmp_path: Path) -> None:
    response = client(tmp_path).post("/campaign/summary", json={"config_path": "only.yaml"})

    assert response.status_code == 422
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error"]["code"] == "request_validation"
    assert payload["error"]["retryable"] is False
    assert "Correct the request fields" in payload["error"]["suggested_action"]
    assert payload["error"]["details"]
    assert "Traceback" not in response.text


def test_server_stages_only_keeps_request_validation_for_malformed_append(
    tmp_path: Path,
) -> None:
    ref = copy_campaign(
        tmp_path,
        "01_simple_2d_maximise_logei.yaml",
        "01_simple_2d_maximise_logei_campaign_log.csv",
    )
    log_path = tmp_path / ref["log_path"]
    before = log_path.read_bytes()

    response = TestClient(create_app(tmp_path, server_stages_only=True)).post(
        "/campaign/suggestions/append",
        json=ref,
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request_validation"
    assert log_path.read_bytes() == before


def test_api_cli_help_without_importing_api_dependencies() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "bo_forge_app.api_cli", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "bo-forge-api" in completed.stdout
    assert "--stage-ttl-seconds" in completed.stdout
    assert "--max-staged-batches" in completed.stdout
    assert "--allow-network-access" in completed.stdout
    assert "--server-stages-only" in completed.stdout
    assert "--no-docs" in completed.stdout


def test_api_cli_missing_dependencies_show_install_hint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delitem(sys.modules, "bo_forge_app.api", raising=False)
    monkeypatch.delitem(sys.modules, "bo_forge_api.api", raising=False)
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


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "lab-server.local"])
def test_api_network_bind_requires_acknowledgement_before_dependency_import(
    host: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_import = builtins.__import__

    def block_api_dependencies(name: str, *args: object, **kwargs: object) -> object:
        if name == "uvicorn" or name.startswith("fastapi"):
            raise AssertionError("API dependencies must not be imported")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_api_dependencies)

    assert api_cli.run(["--root", str(tmp_path), "--host", host]) == 1
    error = capsys.readouterr().err
    assert "requires --allow-network-access" in error
    assert "--host 127.0.0.1" in error


def test_api_launcher_forwards_deployment_controls_after_network_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    class FakeUvicorn:
        @staticmethod
        def run(app: object, *, host: str, port: int) -> None:
            captured.update({"app": app, "host": host, "port": port})

    def fake_create_app(root: Path, **kwargs: object) -> object:
        captured.update({"root": root, "kwargs": kwargs})
        return object()

    monkeypatch.setitem(sys.modules, "uvicorn", FakeUvicorn())
    monkeypatch.setattr(api_module, "create_app", fake_create_app)

    assert (
        api_cli.run(
            [
                "--root",
                str(tmp_path),
                "--host",
                "0.0.0.0",
                "--allow-network-access",
                "--server-stages-only",
                "--no-docs",
            ]
        )
        == 0
    )
    assert captured["host"] == "0.0.0.0"
    assert captured["kwargs"] == {
        "stage_ttl_seconds": 1800.0,
        "max_staged_batches": 128,
        "server_stages_only": True,
        "interactive_docs": False,
    }
    output = capsys.readouterr().out
    assert "Client-carried bundle append: disabled" in output
    assert "Interactive API docs: disabled" in output
    assert "no built-in authentication" in output


def test_api_launcher_stage_limits_parse_and_render(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = api_cli.parse_args(
        [
            "--root",
            str(tmp_path),
            "--stage-ttl-seconds",
            "45",
            "--max-staged-batches",
            "7",
        ]
    )

    assert args.stage_ttl_seconds == pytest.approx(45)
    assert args.max_staged_batches == 7
    api_cli.print_startup_messages(args, tmp_path)
    output = capsys.readouterr().out
    assert "TTL=45s, max active batches=7" in output
    assert "Client-carried bundle append: enabled" in output
    assert "Interactive API docs: enabled" in output


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--stage-ttl-seconds", "0"),
        ("--stage-ttl-seconds", "nan"),
        ("--max-staged-batches", "0"),
        ("--max-staged-batches", "abc"),
    ],
)
def test_api_launcher_rejects_invalid_stage_limits(option: str, value: str) -> None:
    with pytest.raises(SystemExit):
        api_cli.parse_args([option, value])
