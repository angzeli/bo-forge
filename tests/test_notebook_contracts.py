"""Notebook completion checks use generated evidence, never expensive model fits."""

import json
import shutil
import subprocess
import sys
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bo_forge import CampaignConfig, CampaignSession
from bo_forge.costs import evaluate_cost
from notebook_assurance.contracts import CONTRACTS, PR_NOTEBOOKS, discover, select, validate

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS = [name for name, contract in CONTRACTS.items() if "config" in contract]
PREDICTIVE = "23_predictive_diagnostics.ipynb"
BENCHMARK = "24_closed_loop_benchmarks.ipynb"


def write_artifact(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in {".png", ".pdf"}:
        path.write_bytes(figure_fixture(path.suffix))
    else:
        path.write_text("Generated contract fixture\n")


@lru_cache
def figure_fixture(suffix):
    from matplotlib.figure import Figure

    output = BytesIO()
    figure = Figure(figsize=(1, 1))
    figure.subplots().plot([0, 1], [0, 1])
    figure.savefig(output, format=suffix.lstrip("."))
    return output.getvalue()


def campaign_fixture(root, name):
    contract = CONTRACTS[name]
    config_path = root / contract["config"]
    config_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / contract["config"], config_path)
    config = CampaignConfig.from_yaml(config_path)
    seed_path = next((ROOT / "examples").glob(f"{name[:2]}_*campaign_log.csv"))
    seed = pd.read_csv(seed_path, keep_default_na=False)
    observed = seed.loc[seed.status.eq("observed")]
    additions = []
    target = contract["observed"] if contract["observed"] is not None else 17
    for index in range(target - len(observed)):
        original = observed.iloc[2 if name.startswith("14_") and index == 1 else 0]
        row = original.copy()
        row["row_id"] = f"contract_{index}"
        row["iteration"] = 20 + index
        variable = config.variables[0].name
        row[variable] = float(row[variable]) + .0001 * (index + 1)
        if name.startswith(("16_", "20_")):
            sequence = [0.25, 0.5, 0.75] * 4
            if name.startswith("20_"):
                sequence.insert(0, 0.5)
            row["feedstock_acidity"] = sequence[index]
            if config.cost is not None:
                row["cost_estimate"] = evaluate_cost(config, row.to_dict())
        if config.replicates.enabled:
            row["replicate_group"] = f"group_{index}"
            row["replicate_index"] = 0
        additions.append(row)
    frame = pd.concat([seed, pd.DataFrame(additions)], ignore_index=True)
    if name.startswith("07_"):
        for review in ("rejected", "deferred"):
            row = frame.iloc[-1].copy()
            row["row_id"] = review
            row["status"] = "suggested"
            row["review_status"] = review
            row[config.variables[0].name] = float(row[config.variables[0].name]) + .001
            row[config.objective.name] = ""
            row["cost_actual"] = ""
            frame = pd.concat([frame, row.to_frame().T], ignore_index=True)
    log_path = root / contract["log"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(log_path, index=False)
    suggestion = pd.DataFrame([additions[-1]])
    suggestion["status"] = "suggested"
    suggestion[config.objective_names] = ""
    suggestion.to_csv(root / contract["suggestions"], index=False)
    for artifact in contract["artifacts"]:
        write_artifact(root / artifact)
    return contract, frame


def test_registry_matches_exactly_twenty_notebooks_and_bounded_timeouts():
    assert discover(ROOT) == sorted(CONTRACTS)
    assert len(CONTRACTS) == 20
    for name, contract in CONTRACTS.items():
        expected = (3300, 3600) if name[:2] in {"11", "15", "22"} else (600, 1800)
        if name == BENCHMARK:
            expected = (3720, 3900)
        assert (contract["cell_timeout"], contract["notebook_timeout"]) == expected


@pytest.mark.parametrize("change", ["missing", "extra", "rename"])
def test_discovery_fails_closed(tmp_path, change):
    notebooks = tmp_path / "notebooks"
    notebooks.mkdir()
    for name in CONTRACTS:
        (notebooks / name).touch()
    first = notebooks / next(iter(CONTRACTS))
    if change == "missing":
        first.unlink()
    elif change == "extra":
        (notebooks / "99_unregistered.ipynb").touch()
    else:
        first.rename(notebooks / "renamed.ipynb")
    with pytest.raises(ValueError, match="registry mismatch"):
        discover(tmp_path)


def test_selection_profiles_and_explicit_registry_order():
    assert select(ROOT, "full") == discover(ROOT)
    assert select(ROOT, "pr") == PR_NOTEBOOKS
    assert [name[:2] for name in PR_NOTEBOOKS] == [
        "01", "04", "08", "12", "14", "18", "20", "22", "23", "24",
    ]
    assert select(ROOT, "full", list(reversed(PR_NOTEBOOKS))) == PR_NOTEBOOKS
    assert select(ROOT, "pr", BENCHMARK) == [BENCHMARK]
    for profile, selection in [("unknown", None), ("full", []), ("pr", ["../bad.ipynb"]),
                               ("full", [PREDICTIVE, PREDICTIVE]),
                               ("pr", ["11_four_objective_qlogehvi_campaign.ipynb"])]:
        with pytest.raises(ValueError):
            select(ROOT, profile, selection)


@pytest.mark.parametrize("name", CAMPAIGNS)
def test_campaign_artifacts_reload_and_validate_without_fitting(tmp_path, name):
    contract, frame = campaign_fixture(tmp_path, name)
    result = validate(tmp_path, tmp_path / "captures", name)
    assert result["observed_rows"] == int(frame.status.eq("observed").sum())
    assert result["pending_rows"] == contract["pending"]
    assert len(result["objectives"]) == contract["objectives"]
    if name.startswith("14_"):
        assert result["stages"] == {"screening": 3, "refinement": 3}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("fault", ["short", "invalid", "duplicate", "empty_report", "missing_plot",
                                  "wrong_suggestion", "unobserved_suggestion"])
def test_campaign_contract_rejects_incomplete_or_inconsistent_evidence(tmp_path, fault):
    name = "03_three_variable_campaign.ipynb"
    contract, frame = campaign_fixture(tmp_path, name)
    if fault == "short":
        frame = frame.iloc[:-1]
    elif fault == "invalid":
        frame["activity"] = frame.activity.astype(object)
        frame.loc[0, "activity"] = "not-a-number"
    elif fault == "duplicate":
        frame.loc[1, "row_id"] = frame.loc[0, "row_id"]
    elif fault == "empty_report":
        (tmp_path / contract["artifacts"][0]).write_text("")
    elif fault == "missing_plot":
        (tmp_path / contract["artifacts"][1]).unlink()
    else:
        path = tmp_path / contract["suggestions"]
        suggestion = pd.read_csv(path)
        suggestion.loc[0, "precursor_ratio" if fault == "wrong_suggestion" else "row_id"] = (
            .99 if fault == "wrong_suggestion" else "absent"
        )
        suggestion.to_csv(path, index=False)
    frame.to_csv(tmp_path / contract["log"], index=False)
    with pytest.raises(ValueError, match=name):
        validate(tmp_path, tmp_path / "captures", name)


def test_replicate_completion_counts_groups_not_raw_rows(tmp_path):
    name = "08_replicate_aware_campaign.ipynb"
    contract, frame = campaign_fixture(tmp_path, name)
    repeat = frame.iloc[-1].copy()
    repeat["row_id"] = "repeat"
    repeat["replicate_index"] = 1
    frame = pd.concat([frame, repeat.to_frame().T], ignore_index=True)
    frame.to_csv(tmp_path / contract["log"], index=False)
    assert validate(tmp_path, tmp_path, name)["replicate_groups"] == 15
    frame = frame.loc[~frame.replicate_group.eq("group_0")]
    frame.to_csv(tmp_path / contract["log"], index=False)
    with pytest.raises(ValueError, match="replicate group count"):
        validate(tmp_path, tmp_path, name)


def test_structured_contract_does_not_accept_six_in_one_stage(tmp_path):
    name = "14_structured_campaign_tutorial.ipynb"
    contract, frame = campaign_fixture(tmp_path, name)
    frame["stage"] = "screening"
    frame[["temperature", "residence_time"]] = ""
    frame.to_csv(tmp_path / contract["log"], index=False)
    with pytest.raises(ValueError, match="three observations per stage"):
        validate(tmp_path, tmp_path, name)


@pytest.mark.parametrize("name", ["16_contextual_logei_campaign.ipynb",
                                  "20_contextual_cost_review_logei_campaign.ipynb"])
@pytest.mark.parametrize("fault", ["default_only", "wrong_override", "wrong_order"])
def test_context_contract_checks_requested_sequence(tmp_path, name, fault):
    contract, frame = campaign_fixture(tmp_path, name)
    added = frame.index[frame.iteration.gt(0)]
    if fault == "default_only":
        frame.loc[added, "feedstock_acidity"] = 0.5
    elif fault == "wrong_override":
        frame.loc[added[0], "feedstock_acidity"] = 0.4
    else:
        pair = added[:2]
        frame.loc[pair, "feedstock_acidity"] = (
            frame.loc[pair, "feedstock_acidity"].iloc[::-1].values)
    if name.startswith("20_"):
        config = CampaignConfig.from_yaml(tmp_path / contract["config"])
        frame.loc[added, "cost_estimate"] = [
            evaluate_cost(config, row.to_dict()) for _, row in frame.loc[added].iterrows()]
    frame.to_csv(tmp_path / contract["log"], index=False)
    with pytest.raises(ValueError, match="context sequence"):
        validate(tmp_path, tmp_path, name)


def test_cli_initialization_cell_reloads_seeded_working_log(tmp_path):
    name = "04_cli_four_variable_campaign.ipynb"
    contract = CONTRACTS[name]
    (tmp_path / "configs").mkdir()
    (tmp_path / "examples").mkdir()
    config = Path(contract["config"])
    seed = Path("examples/04_simple_4d_maximise_logei_campaign_log.csv")
    shutil.copyfile(ROOT / config, tmp_path / config)
    shutil.copyfile(ROOT / seed, tmp_path / seed)
    notebook = json.loads((ROOT / "notebooks" / name).read_text())
    code = next("".join(cell["source"]) for cell in notebook["cells"]
                if cell["cell_type"] == "code" and '"init-log"' in "".join(cell["source"]))
    namespace = {"PROJECT_ROOT": tmp_path, "CONFIG_PATH": config, "SEED_LOG_PATH": seed,
                 "WORKING_LOG_PATH": Path(contract["log"]), "subprocess": subprocess,
                 "sys": sys, "pd": pd, "Path": Path, "TemporaryDirectory": TemporaryDirectory}
    exec(compile(code, name, "exec"), namespace)
    session = CampaignSession.from_files(tmp_path / config, tmp_path / contract["log"])
    session.reload()
    session.validate()
    assert len(session.df) == len(pd.read_csv(tmp_path / seed))
    assert not (tmp_path / (contract["log"] + ".manifest.json")).exists()


@pytest.fixture
def predictive_capture(tmp_path, monkeypatch):
    import bo_forge.predictive as predictive

    captures = tmp_path / "captures"
    workspace = captures / "bo-forge-predictive-fixture"
    workspace.mkdir(parents=True)
    notebook = json.loads((ROOT / "notebooks" / PREDICTIVE).read_text())
    setup = ["".join(cell["source"]) for cell in notebook["cells"]
             if cell["cell_type"] == "code"][1]
    # Reuse the real bounded input construction, replacing only temporary ownership.
    setup = setup[setup.index("config_path ="):]
    namespace = {"work_dir": workspace, "CampaignConfig": CampaignConfig,
                 "CampaignSession": CampaignSession, "np": np, "pd": pd,
                 "display": lambda *_: None, "model_summary": lambda *_: None}
    import yaml

    from bo_forge.validation import canonical_columns

    namespace.update(yaml=yaml, canonical_columns=canonical_columns)
    exec(compile(setup, PREDICTIVE, "exec"), namespace)

    def evaluate_fold(config, training, held_out, fold):
        rows = []
        for _, row in held_out.iterrows():
            observed = float(row[config.objective.name])
            rows.append({"model_profile": config.model.profile, "row_id": row.row_id,
                         "fold": fold, "observed": observed, "fit_status": "complete",
                         **predictive._prediction_metrics(observed, observed + .1, 1.0)})
        return rows, {"model_profile": config.model.profile, "fold": fold,
                      "training_rows": len(training), "held_out_rows": len(held_out),
                      "fit_status": "complete", "fit_message": "", "fitting_rng_fingerprint": None}

    monkeypatch.setattr(predictive, "_evaluate_fold", evaluate_fold)
    campaign = namespace["campaign"]
    result = campaign.model_predictive_evaluation(profiles=["default", "smooth"], folds=3, seed=0)
    result.export(workspace / "evaluation")
    for filename in ("predictions.png", "residuals.png"):
        write_artifact(workspace / "evaluation" / filename)
    return captures, workspace


def test_predictive_capture_validates_two_profiles_six_folds_and_exports(predictive_capture):
    captures, _ = predictive_capture
    result = validate(ROOT, captures, PREDICTIVE)
    assert result["folds"] == 6
    assert result["predictions"] == 40


@pytest.mark.parametrize("fault", ["missing", "failed", "duplicate", "membership", "nonfinite",
                                  "input", "metadata"])
def test_predictive_capture_rejects_invalid_results(predictive_capture, fault):
    captures, workspace = predictive_capture
    output = workspace / "evaluation"
    if fault == "missing":
        (output / "residuals.png").unlink()
    elif fault == "metadata":
        (output / "metadata.json").write_text("{}")
    else:
        path = workspace / "observed.csv" if fault == "input" else output / "predictions.csv"
        frame = pd.read_csv(path)
        if fault == "failed":
            frame.loc[0, "fit_status"] = "failed"
        elif fault == "duplicate":
            frame.loc[0, "row_id"] = frame.loc[1, "row_id"]
        elif fault == "membership":
            frame.loc[0, "fold"] = 3
        elif fault == "nonfinite":
            frame.loc[0, "predicted_mean"] = float("inf")
        else:
            frame.loc[0, "response"] = 999.0
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=PREDICTIVE):
        validate(ROOT, captures, PREDICTIVE)


@pytest.fixture(scope="module")
def benchmark_capture_base(tmp_path_factory):
    from benchmarks import runner
    from benchmarks.report import generate_report
    from tests._benchmark_support import controlled_execute

    root = tmp_path_factory.mktemp("notebook-benchmark")
    workspace = root / "bo-forge-benchmark-notebook-fixture"
    with patch.object(runner, "_execute_trial", controlled_execute):
        runner.run_suite(ROOT / "benchmarks/specs/smoke.yaml", workspace / "smoke")
    generate_report(workspace / "smoke", workspace / "smoke/report")
    generate_report(workspace / "smoke", workspace / "regenerated-report")
    return workspace


@pytest.fixture
def benchmark_capture(benchmark_capture_base, tmp_path):
    captures = tmp_path / "captures"
    workspace = shutil.copytree(benchmark_capture_base, captures / benchmark_capture_base.name)
    return captures, workspace


def test_benchmark_capture_checks_six_trials_and_both_reports(benchmark_capture):
    captures, _ = benchmark_capture
    result = validate(ROOT, captures, BENCHMARK)
    assert result["complete_trials"] == 6
    assert result["evaluations"] == 36


@pytest.mark.parametrize("fault", ["status", "trace", "summary", "regenerated", "plot", "run"])
def test_benchmark_capture_rejects_incomplete_or_stale_evidence(benchmark_capture, fault):
    captures, workspace = benchmark_capture
    run = workspace / "smoke"
    trial = next((run / "trials").iterdir())
    if fault in {"status", "run"}:
        path = trial / "status.json" if fault == "status" else run / "run.json"
        value = json.loads(path.read_text())
        value["status"] = "failed"
        path.write_text(json.dumps(value))
    elif fault == "trace":
        (trial / "trace.jsonl").write_text("")
    elif fault == "summary":
        path = run / "report/summary.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "complete"] = 0
        frame.to_csv(path, index=False)
    elif fault == "regenerated":
        (workspace / "regenerated-report/report.md").write_text("stale report")
    else:
        (run / "report/figures/branin-noisy.png").unlink()
    with pytest.raises(ValueError, match=BENCHMARK):
        validate(ROOT, captures, BENCHMARK)


def test_missing_capture_and_unknown_notebook_raise_value_error(tmp_path):
    for name in (PREDICTIVE, BENCHMARK, "missing.ipynb"):
        with pytest.raises(ValueError):
            validate(ROOT, tmp_path, name)
