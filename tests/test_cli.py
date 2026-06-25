import subprocess
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

import bo_forge.cli as cli
from bo_forge.cli import run
from bo_forge.config import (
    BOConfig,
    CampaignConfig,
    CostConfig,
    FidelityConfig,
    ObjectiveConfig,
    ReplicateConfig,
    ReviewConfig,
    VariableConfig,
)
from bo_forge.io import empty_campaign_log
from bo_forge.logs import load_campaign_log
from bo_forge.validation import canonical_columns

matplotlib.use("Agg")


def write_config(path: Path, *, initial_design_size: int = 2) -> Path:
    path.write_text(
        f"""
campaign_name: cli_test
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
bo:
  batch_size: 1
  initial_design_size: {initial_design_size}
  acquisition: log_ei
  random_seed: 7
  raw_samples: 16
  num_restarts: 2
  mc_samples: 16
""",
        encoding="utf-8",
    )
    return path


def write_mixed_config(path: Path, *, initial_design_size: int = 3) -> Path:
    path.write_text(
        f"""
campaign_name: mixed_cli_test
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
  - name: repeats
    type: integer
    lower: 1
    upper: 3
  - name: dose
    type: discrete
    values: [0.1, 0.2, 0.5]
  - name: solvent
    type: categorical
    values: [MeCN, EtOH]
bo:
  batch_size: 1
  initial_design_size: {initial_design_size}
  acquisition: log_ei
  random_seed: 7
  raw_samples: 16
  num_restarts: 2
  mc_samples: 16
""",
        encoding="utf-8",
    )
    return path


def write_cost_review_config(path: Path, *, initial_design_size: int = 2) -> Path:
    path.write_text(
        f"""
campaign_name: cost_review_cli_test
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
cost:
  expression: "1.0 + x"
  weight: 0.5
  budget: 10
  candidate_pool_size: 16
  top_k: 8
review:
  enabled: true
bo:
  batch_size: 1
  initial_design_size: {initial_design_size}
  acquisition: log_ei
  random_seed: 7
  raw_samples: 16
  num_restarts: 2
  mc_samples: 16
""",
        encoding="utf-8",
    )
    return path


def write_replicate_config(path: Path, *, initial_design_size: int = 2) -> Path:
    path.write_text(
        f"""
campaign_name: replicate_cli_test
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
replicates:
  enabled: true
bo:
  batch_size: 1
  initial_design_size: {initial_design_size}
  acquisition: log_ei
  random_seed: 7
  raw_samples: 16
  num_restarts: 2
  mc_samples: 16
""",
        encoding="utf-8",
    )
    return path


def write_multi_objective_config(path: Path, *, initial_design_size: int = 10) -> Path:
    path.write_text(
        f"""
campaign_name: multi_cli_test
objectives:
  - name: yield_score
    direction: maximize
    reference_point: 40
  - name: waste_score
    direction: minimize
    reference_point: 25
variables:
  - name: temperature
    type: continuous
    lower: 20
    upper: 100
bo:
  batch_size: 1
  initial_design_size: {initial_design_size}
  acquisition: qlog_ehvi
  random_seed: 7
  raw_samples: 8
  num_restarts: 2
  mc_samples: 8
""",
        encoding="utf-8",
    )
    return path


def write_multi_objective_cost_config(path: Path, *, initial_design_size: int = 10) -> Path:
    path.write_text(
        f"""
campaign_name: multi_cost_cli_test
objectives:
  - name: yield_score
    direction: maximize
    reference_point: 40
  - name: waste_score
    direction: minimize
    reference_point: 25
variables:
  - name: temperature
    type: continuous
    lower: 20
    upper: 100
cost:
  expression: "1.0 + 0.02 * temperature"
  weight: 0.5
  budget: 20
  candidate_pool_size: 16
  top_k: 8
bo:
  batch_size: 1
  initial_design_size: {initial_design_size}
  acquisition: qlog_ehvi
  random_seed: 7
  raw_samples: 8
  num_restarts: 2
  mc_samples: 8
""",
        encoding="utf-8",
    )
    return path


def write_fidelity_config(
    path: Path,
    *,
    initial_design_size: int = 3,
    review: bool = False,
) -> Path:
    review_block = "review:\n  enabled: true\n" if review else ""
    path.write_text(
        f"""
campaign_name: fidelity_cli_test
objective:
  name: activity
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
  - name: fidelity
    type: continuous
    lower: 0.2
    upper: 1.0
fidelity:
  variable: fidelity
  target: 1.0
  num_fantasies: 8
{review_block}\
bo:
  batch_size: 1
  initial_design_size: {initial_design_size}
  acquisition: qmf_kg
  random_seed: 7
  raw_samples: 8
  num_restarts: 1
  mc_samples: 8
""",
        encoding="utf-8",
    )
    return path


def config(initial_design_size: int = 2) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="cli_test",
        objective=ObjectiveConfig(name="score", direction="maximize"),
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=BOConfig(
            batch_size=1,
            initial_design_size=initial_design_size,
            random_seed=7,
            raw_samples=16,
            num_restarts=2,
            mc_samples=16,
        ),
    )


def fidelity_config(
    initial_design_size: int = 3,
    *,
    review: bool = False,
) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="fidelity_cli_test",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("fidelity", "continuous", 0.2, 1.0),
        ),
        bo=BOConfig(
            batch_size=1,
            initial_design_size=initial_design_size,
            acquisition="qmf_kg",
            random_seed=7,
            raw_samples=8,
            num_restarts=1,
            mc_samples=8,
        ),
        fidelity=FidelityConfig(variable="fidelity", target=1.0, num_fantasies=8),
        review=ReviewConfig(enabled=review),
    )


def mixed_config(initial_design_size: int = 3) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="mixed_cli_test",
        objective=ObjectiveConfig(name="score", direction="maximize"),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("repeats", "integer", 1.0, 3.0),
            VariableConfig("dose", "discrete", values=(0.1, 0.2, 0.5)),
            VariableConfig("solvent", "categorical", values=("MeCN", "EtOH")),
        ),
        bo=BOConfig(
            batch_size=1,
            initial_design_size=initial_design_size,
            random_seed=7,
            raw_samples=16,
            num_restarts=2,
            mc_samples=16,
        ),
    )


def cost_review_config(initial_design_size: int = 2) -> CampaignConfig:
    cfg = config(initial_design_size=initial_design_size)
    return CampaignConfig(
        campaign_name="cost_review_cli_test",
        objective=cfg.objective,
        variables=cfg.variables,
        bo=cfg.bo,
        cost=CostConfig(
            expression="1.0 + x",
            weight=0.5,
            budget=10.0,
            candidate_pool_size=16,
            top_k=8,
        ),
        review=ReviewConfig(enabled=True),
    )


def replicate_config(initial_design_size: int = 2) -> CampaignConfig:
    cfg = config(initial_design_size=initial_design_size)
    return CampaignConfig(
        campaign_name="replicate_cli_test",
        objective=cfg.objective,
        variables=cfg.variables,
        bo=cfg.bo,
        replicates=ReplicateConfig(enabled=True),
    )


def multi_objective_config(initial_design_size: int = 10, *, cost: bool = False) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="multi_cli_test",
        objective=ObjectiveConfig(name="yield_score", direction="maximize", reference_point=40.0),
        objectives=(
            ObjectiveConfig(name="yield_score", direction="maximize", reference_point=40.0),
            ObjectiveConfig(name="waste_score", direction="minimize", reference_point=25.0),
        ),
        variables=(VariableConfig("temperature", "continuous", 20.0, 100.0),),
        bo=BOConfig(
            batch_size=1,
            initial_design_size=initial_design_size,
            acquisition="qlog_ehvi",
            random_seed=7,
            raw_samples=8,
            num_restarts=2,
            mc_samples=8,
        ),
        cost=CostConfig(
            expression="1.0 + 0.02 * temperature",
            weight=0.5,
            budget=20.0,
            candidate_pool_size=16,
            top_k=8,
        )
        if cost
        else None,
    )


def observed_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "obs_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "x": 0.2,
                "score": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "obs_1",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "x": 0.8,
                "score": 1.5,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def fidelity_observed_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "mf_obs_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "x": 0.1,
                "fidelity": 0.25,
                "activity": 0.7,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "mf_obs_1",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "x": 0.3,
                "fidelity": 0.5,
                "activity": 1.1,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "mf_obs_2",
                "iteration": 2,
                "status": "observed",
                "source": "manual",
                "x": 0.6,
                "fidelity": 0.75,
                "activity": 1.4,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "mf_obs_3",
                "iteration": 3,
                "status": "observed",
                "source": "manual",
                "x": 0.85,
                "fidelity": 1.0,
                "activity": 1.3,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def mixed_observed_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "mixed_obs_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "x": 0.1,
                "repeats": 1,
                "dose": 0.1,
                "solvent": "MeCN",
                "score": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "mixed_obs_1",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "x": 0.4,
                "repeats": 2,
                "dose": 0.2,
                "solvent": "EtOH",
                "score": 1.5,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "mixed_obs_2",
                "iteration": 2,
                "status": "observed",
                "source": "manual",
                "x": 0.8,
                "repeats": 3,
                "dose": 0.5,
                "solvent": "MeCN",
                "score": 1.2,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def cost_review_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "obs_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "review_status": "accepted",
                "review_note": "",
                "x": 0.2,
                "score": 1.0,
                "cost_estimate": 1.2,
                "cost_actual": 1.1,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
                "utility": "",
            },
            {
                "row_id": "suggested_0",
                "iteration": 1,
                "status": "suggested",
                "source": "sobol",
                "review_status": "pending",
                "review_note": "",
                "x": 0.5,
                "score": "",
                "cost_estimate": 1.5,
                "cost_actual": "",
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
                "utility": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def replicate_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "rep_0a",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "replicate_group": "group_0",
                "replicate_index": 0,
                "x": 0.2,
                "score": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "rep_0b",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "replicate_group": "group_0",
                "replicate_index": 1,
                "x": 0.2,
                "score": 1.6,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "rep_1a",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "replicate_group": "group_1",
                "replicate_index": 0,
                "x": 0.8,
                "score": 1.4,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def multi_objective_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "obs_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "temperature": 35.0,
                "yield_score": 55.0,
                "waste_score": 20.0,
                "cost_estimate": 1.7,
                "cost_actual": "",
                "predicted_mean_yield_score": "",
                "predicted_std_yield_score": "",
                "predicted_mean_waste_score": "",
                "predicted_std_waste_score": "",
                "acquisition": "",
                "utility": "",
            },
            {
                "row_id": "suggested_0",
                "iteration": 1,
                "status": "suggested",
                "source": "sobol",
                "temperature": 65.0,
                "yield_score": "",
                "waste_score": "",
                "cost_estimate": 2.3,
                "cost_actual": "",
                "predicted_mean_yield_score": "",
                "predicted_std_yield_score": "",
                "predicted_mean_waste_score": "",
                "predicted_std_waste_score": "",
                "acquisition": "",
                "utility": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def write_log(path: Path, cfg: CampaignConfig, df: pd.DataFrame | None = None) -> Path:
    if df is None:
        df = empty_campaign_log(cfg)
    df.to_csv(path, index=False)
    return path


def base_args(config_path: Path, log_path: Path) -> list[str]:
    return ["--config", str(config_path), "--log", str(log_path)]


def output_under_file_parent(tmp_path: Path, filename: str) -> Path:
    parent = tmp_path / "not_a_dir"
    parent.write_text("not a directory", encoding="utf-8")
    return parent / filename


def run_python_module(module: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )


def test_version_outputs_clean_line(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["--version"]) == 0

    captured = capsys.readouterr()
    assert captured.out == "bo-forge 1.5.1\n"
    assert captured.err == ""


@pytest.mark.parametrize("module", ["bo_forge", "bo_forge.cli"])
def test_python_module_entrypoint_version(module: str) -> None:
    completed = run_python_module(module, "--version")

    assert completed.returncode == 0
    assert completed.stdout == "bo-forge 1.5.1\n"
    assert completed.stderr == ""


def test_python_module_entrypoint_validate_success(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config())

    completed = run_python_module(
        "bo_forge",
        "validate",
        *base_args(config_path, log_path),
    )

    assert completed.returncode == 0
    assert completed.stdout == "Campaign log is valid.\n"
    assert completed.stderr == ""


def test_python_module_entrypoint_missing_arguments_returns_argparse_error() -> None:
    completed = run_python_module("bo_forge", "validate")

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "usage:" in completed.stderr
    assert "required" in completed.stderr


def test_doctor_success_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["doctor"]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "BO Forge doctor" in captured.out
    assert "BO Forge version" in captured.out
    assert "Python executable" in captured.out
    assert "Python version" in captured.out
    assert "torch" in captured.out
    assert "botorch" in captured.out
    assert "gpytorch" in captured.out
    assert captured.out.rstrip().endswith("Status: OK")


def test_doctor_import_failure_returns_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_import_module = cli.importlib.import_module

    def fail_torch_import(module_name: str) -> object:
        if module_name == "torch":
            raise ImportError("missing torch")
        return original_import_module(module_name)

    monkeypatch.setattr(cli.importlib, "import_module", fail_torch_import)

    assert run(["doctor"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error: Doctor check failed while importing 'torch': missing torch" in captured.err


def test_init_log_creates_empty_canonical_log(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = tmp_path / "nested" / "campaign.csv"

    assert run(["init-log", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert captured.out == f"Created empty campaign log: {log_path}\n"
    assert captured.err == ""
    cfg = CampaignConfig.from_yaml(config_path)
    df = load_campaign_log(log_path, cfg)
    assert df.empty
    assert list(df.columns) == canonical_columns(cfg)


def test_init_log_creates_cost_review_schema(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_cost_review_config(tmp_path / "campaign.yaml")
    cfg = cost_review_config()
    log_path = tmp_path / "nested" / "campaign.csv"

    assert run(["init-log", *base_args(config_path, log_path)]) == 0

    capsys.readouterr()
    df = load_campaign_log(log_path, cfg)
    assert df.empty
    assert list(df.columns) == canonical_columns(cfg)


def test_init_log_creates_replicate_schema(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_replicate_config(tmp_path / "campaign.yaml")
    cfg = replicate_config()
    log_path = tmp_path / "nested" / "campaign.csv"

    assert run(["init-log", *base_args(config_path, log_path)]) == 0

    capsys.readouterr()
    df = load_campaign_log(log_path, cfg)
    assert df.empty
    assert list(df.columns) == canonical_columns(cfg)


def test_init_log_refuses_to_overwrite_existing_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = tmp_path / "campaign.csv"
    log_path.write_text("existing", encoding="utf-8")

    assert run(["init-log", *base_args(config_path, log_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error:" in captured.err
    assert "file already exists" in captured.err
    assert log_path.read_text(encoding="utf-8") == "existing"


def test_init_log_does_not_create_file_when_config_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "missing.yaml"
    log_path = tmp_path / "campaign.csv"

    assert run(["init-log", *base_args(config_path, log_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Could not read config file" in captured.err
    assert not log_path.exists()


def test_init_log_write_failure_returns_clear_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = output_under_file_parent(tmp_path, "campaign.csv")

    assert run(["init-log", *base_args(config_path, log_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"Error: Could not write empty campaign log '{log_path}'" in captured.err


def test_init_log_missing_required_arguments_return_argparse_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run(["init-log"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage:" in captured.err
    assert "required" in captured.err


def test_validate_success_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config())

    assert run(["validate", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert captured.out == "Campaign log is valid.\n"
    assert captured.err == ""


def test_mixed_validate_success_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = write_mixed_config(tmp_path / "mixed.yaml")
    cfg = mixed_config()
    log_path = write_log(tmp_path / "mixed.csv", cfg, mixed_observed_log(cfg))

    assert run(["validate", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert captured.out == "Campaign log is valid.\n"
    assert captured.err == ""


def test_validate_constrained_log_failure_returns_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/06_mixed_constrained_logei.yaml")
    cfg = CampaignConfig.from_yaml(config_path)
    df = pd.read_csv(
        "examples/06_mixed_constrained_logei_campaign_log.csv",
        keep_default_na=False,
    )
    df.loc[0, "solvent"] = "Water"
    df.loc[0, "base_equivalents"] = 1.0
    df.loc[0, "reaction_time"] = 20
    log_path = write_log(tmp_path / "constrained.csv", cfg, df)

    assert run(["validate", *base_args(config_path, log_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "violates constraint" in captured.err


def test_constrained_suggest_output_is_feasible(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/06_mixed_constrained_logei.yaml")
    log_path = tmp_path / "constrained.csv"
    output_path = tmp_path / "suggestions.csv"
    seed = pd.read_csv(
        "examples/06_mixed_constrained_logei_campaign_log.csv",
        keep_default_na=False,
    )
    seed.to_csv(log_path, index=False)

    assert run(
        [
            "suggest",
            *base_args(config_path, log_path),
            "--output",
            str(output_path),
        ]
    ) == 0

    captured = capsys.readouterr()
    assert "Generated" in captured.out
    suggestions = pd.read_csv(output_path, keep_default_na=False)
    assert not (
        (suggestions["solvent"] == "Water")
        & (suggestions["base_equivalents"].astype(float) >= 0.5)
    ).any()
    assert not (
        (suggestions["solvent"] == "Water")
        & (suggestions["reaction_time"].astype(int) < 35)
    ).any()


def test_contextual_suggest_accepts_context_value(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/16_contextual_logei.yaml")
    log_path = tmp_path / "contextual.csv"
    output_path = tmp_path / "contextual_suggestions.csv"
    seed = pd.read_csv(
        "examples/16_contextual_logei_campaign_log.csv",
        keep_default_na=False,
    )
    seed.to_csv(log_path, index=False)

    assert run(
        [
            "suggest",
            *base_args(config_path, log_path),
            "--context",
            "feedstock_acidity=0.25",
            "--output",
            str(output_path),
        ]
    ) == 0

    captured = capsys.readouterr()
    assert "Generated 1 suggestion(s)." in captured.out
    suggestions = pd.read_csv(output_path, keep_default_na=False)
    assert suggestions["feedstock_acidity"].astype(float).tolist() == [pytest.approx(0.25)]


def test_contextual_suggest_missing_context_does_not_append(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "contextual.yaml"
    config_path.write_text(
        """
campaign_name: contextual_cli
objective: {name: score, direction: maximize}
variables:
  - {name: x, type: continuous, lower: 0, upper: 1}
  - {name: feedstock_acidity, type: continuous, lower: 0, upper: 1}
context:
  variables: [feedstock_acidity]
bo:
  batch_size: 1
  initial_design_size: 2
  acquisition: log_ei
""",
        encoding="utf-8",
    )
    cfg = CampaignConfig.from_yaml(config_path)
    log_path = write_log(tmp_path / "contextual.csv", cfg)
    before = log_path.read_bytes()

    assert run(["suggest", *base_args(config_path, log_path), "--append"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "missing=['feedstock_acidity']" in captured.err
    assert "Hint: Use --context NAME=VALUE" in captured.err
    assert log_path.read_bytes() == before


def test_contextual_suggest_rejects_malformed_context(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/16_contextual_logei.yaml")
    log_path = tmp_path / "contextual.csv"
    pd.read_csv(
        "examples/16_contextual_logei_campaign_log.csv",
        keep_default_na=False,
    ).to_csv(log_path, index=False)

    assert run(["suggest", *base_args(config_path, log_path), "--context", "bad"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Malformed --context value" in captured.err


def test_contextual_cli_context_summary_outputs_table(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/16_contextual_logei.yaml")
    log_path = tmp_path / "contextual.csv"
    pd.read_csv(
        "examples/16_contextual_logei_campaign_log.csv",
        keep_default_na=False,
    ).to_csv(log_path, index=False)

    assert run(["context-summary", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert "context_key" in captured.out
    assert "feedstock_acidity=0.3" in captured.out
    assert "ctx_seed_1" in captured.out


def test_contextual_cli_context_summary_handles_pending_only_log(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = CampaignConfig.from_yaml("configs/16_contextual_logei.yaml")
    pending = {
        "row_id": "pending_0",
        "iteration": 0,
        "status": "suggested",
        "source": "sobol",
        "catalyst_loading": 0.5,
        "reaction_temperature": 80,
        "solvent": "MeCN",
        "feedstock_acidity": 0.25,
        "yield_score": "",
        "predicted_mean": "",
        "predicted_std": "",
        "acquisition": "",
    }
    log_path = tmp_path / "contextual_pending.csv"
    pd.DataFrame(
        [[pending[column] for column in canonical_columns(cfg)]],
        columns=canonical_columns(cfg),
    ).to_csv(log_path, index=False)

    assert (
        run(
            [
                "context-summary",
                *base_args(Path("configs/16_contextual_logei.yaml"), log_path),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert "context_key" in captured.out
    assert "feedstock_acidity=0.25" in captured.out
    assert "pending_suggestions" in captured.out


def test_contextual_cli_context_summary_rejects_non_context_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config(), observed_log(config()))

    assert run(["context-summary", *base_args(config_path, log_path)]) == 1

    captured = capsys.readouterr()
    assert "context-summary requires a contextual config" in captured.err


def test_contextual_cli_plot_context_diagnostics_handles_pending_only_log(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = CampaignConfig.from_yaml("configs/16_contextual_logei.yaml")
    pending = {
        "row_id": "pending_0",
        "iteration": 0,
        "status": "suggested",
        "source": "sobol",
        "catalyst_loading": 0.5,
        "reaction_temperature": 80,
        "solvent": "MeCN",
        "feedstock_acidity": 0.25,
        "yield_score": "",
        "predicted_mean": "",
        "predicted_std": "",
        "acquisition": "",
    }
    log_path = tmp_path / "contextual_pending.csv"
    pd.DataFrame(
        [[pending[column] for column in canonical_columns(cfg)]],
        columns=canonical_columns(cfg),
    ).to_csv(log_path, index=False)
    output_path = tmp_path / "reports" / "context_pending.png"

    assert (
        run(
            [
                "plot",
                *base_args(Path("configs/16_contextual_logei.yaml"), log_path),
                "--kind",
                "context-diagnostics",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert f"Wrote context-diagnostics plot: {output_path}" in captured.out
    assert output_path.exists()


def test_contextual_cli_plot_context_diagnostics_writes_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/16_contextual_logei.yaml")
    log_path = tmp_path / "contextual.csv"
    pd.read_csv(
        "examples/16_contextual_logei_campaign_log.csv",
        keep_default_na=False,
    ).to_csv(log_path, index=False)
    output_path = tmp_path / "reports" / "context.png"

    assert (
        run(
            [
                "plot",
                *base_args(config_path, log_path),
                "--kind",
                "context-diagnostics",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert f"Wrote context-diagnostics plot: {output_path}" in captured.out
    assert output_path.exists()


def test_config_load_failure_returns_hint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "missing.yaml"
    log_path = tmp_path / "campaign.csv"

    assert run(["validate", *base_args(config_path, log_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error:" in captured.err
    assert "Could not read config file" in captured.err
    assert "Hint: Check the YAML config path and campaign settings." in captured.err


def test_validate_failure_returns_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    cfg = config()
    invalid = observed_log(cfg)
    invalid.loc[0, "x"] = 2.0
    log_path = write_log(tmp_path / "campaign.csv", cfg, invalid)

    assert run(["validate", *base_args(config_path, log_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error:" in captured.err
    assert "outside bounds" in captured.err
    assert (
        "Hint: Check the CSV schema, statuses, objective values, and variable bounds."
        in captured.err
    )


def test_suggest_with_pending_suggestions_returns_hint_without_mutating_csv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config())
    assert run(["suggest", *base_args(config_path, log_path), "--append"]) == 0
    capsys.readouterr()
    before_csv = log_path.read_text(encoding="utf-8")

    assert run(["suggest", *base_args(config_path, log_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error:" in captured.err
    assert "unresolved status='suggested'" in captured.err
    assert (
        "Hint: Resolve pending suggestions or review the campaign state before "
        "requesting new suggestions."
        in captured.err
    )
    assert log_path.read_text(encoding="utf-8") == before_csv


def test_structured_suggest_requires_stage_without_mutating_csv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/13_structured_campaign_core.yaml")
    log_path = tmp_path / "structured.csv"
    cfg = CampaignConfig.from_yaml(config_path)
    write_log(log_path, cfg)
    before = log_path.read_bytes()

    assert run(["suggest", *base_args(config_path, log_path), "--append"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Structured campaign suggestions require an explicit stage" in captured.err
    assert "Hint: Use --stage with one configured structured stage name" in captured.err
    assert log_path.read_bytes() == before


def test_structured_suggest_dry_run_accepts_stage_without_mutating_csv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/13_structured_campaign_core.yaml")
    log_path = tmp_path / "structured.csv"
    cfg = CampaignConfig.from_yaml(config_path)
    write_log(log_path, cfg)
    before = log_path.read_bytes()

    assert run(["suggest", *base_args(config_path, log_path), "--stage", "screen"]) == 0

    captured = capsys.readouterr()
    assert "Generated 1 suggestion(s)." in captured.out
    assert "screen" in captured.out
    assert log_path.read_bytes() == before


def test_structured_documented_init_log_then_suggest_flow_succeeds(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/13_structured_campaign_core.yaml")
    log_path = tmp_path / "structured_working.csv"

    assert run(["init-log", *base_args(config_path, log_path)]) == 0
    assert run(["suggest", *base_args(config_path, log_path), "--stage", "screen"]) == 0

    captured = capsys.readouterr()
    assert "Created empty campaign log" in captured.out
    assert "Generated 1 suggestion(s)." in captured.out
    assert "screen" in captured.out


def test_structured_suggest_append_writes_stage_aware_rows(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/13_structured_campaign_core.yaml")
    log_path = tmp_path / "structured.csv"
    cfg = CampaignConfig.from_yaml(config_path)
    write_log(log_path, cfg)

    assert (
        run(
            [
                "suggest",
                *base_args(config_path, log_path),
                "--stage",
                "screen",
                "--append",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert "Appended suggestions to campaign log" in captured.out
    df = load_campaign_log(log_path, cfg)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["stage"] == "screen"
    assert pd.notna(row["precursor_ratio"])
    assert pd.notna(row["electrolyte"])
    assert row["annealing_temperature"] == ""


def test_structured_suggest_unknown_stage_fails_without_mutating_csv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/13_structured_campaign_core.yaml")
    log_path = tmp_path / "structured.csv"
    cfg = CampaignConfig.from_yaml(config_path)
    write_log(log_path, cfg)
    before = log_path.read_bytes()

    assert (
        run(
            [
                "suggest",
                *base_args(config_path, log_path),
                "--stage",
                "missing",
                "--append",
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Unknown structured campaign stage 'missing'" in captured.err
    assert log_path.read_bytes() == before


def test_structured_suggest_invalid_stage_format_returns_stage_hint_without_mutating(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/13_structured_campaign_core.yaml")
    log_path = tmp_path / "structured.csv"
    cfg = CampaignConfig.from_yaml(config_path)
    write_log(log_path, cfg)
    before = log_path.read_bytes()

    assert (
        run(
            [
                "suggest",
                *base_args(config_path, log_path),
                "--stage",
                " screen",
                "--append",
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Invalid structured campaign stage" in captured.err
    assert "Hint: Use --stage with one configured structured stage name" in captured.err
    assert log_path.read_bytes() == before


def test_stage_summary_cli_prints_structured_stage_table(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/13_structured_campaign_core.yaml")
    log_path = tmp_path / "structured.csv"
    log_path.write_bytes(
        Path("examples/13_structured_campaign_core_campaign_log.csv").read_bytes()
    )

    assert run(["stage-summary", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert "stage" in captured.out
    assert "screen" in captured.out
    assert "refine" in captured.out
    assert "active_variables" in captured.out
    assert "No observed rows for stage." not in captured.err


def test_stage_summary_cli_rejects_non_structured_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg))

    assert run(["stage-summary", *base_args(config_path, log_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "stage-summary requires a structured campaign config" in captured.err


def test_mark_observed_missing_row_returns_hint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config())

    assert (
        run(
            [
                "mark-observed",
                *base_args(config_path, log_path),
                "--row-id",
                "missing",
                "--objective-value",
                "1.0",
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error:" in captured.err
    assert "row_id was not found" in captured.err
    assert (
        "Hint: Check the row_id, pending status, campaign log path, and file permissions."
        in captured.err
    )


@pytest.mark.parametrize(
    "objective_args, expected_error",
    [
        (["--objective", "yield_score"], "Malformed --objective value"),
        (
            [
                "--objective",
                "yield_score=60",
                "--objective",
                "yield_score=61",
                "--objective",
                "waste_score=12",
            ],
            "Duplicate --objective value",
        ),
        (["--objective", "yield_score=60"], "missing=['waste_score']"),
        (
            [
                "--objective",
                "yield_score=60",
                "--objective",
                "waste_score=12",
                "--objective",
                "unknown=1",
            ],
            "extra=['unknown']",
        ),
        (
            ["--objective", "yield_score=bad", "--objective", "waste_score=12"],
            "must be numeric",
        ),
        (
            [
                "--objective-value",
                "1.0",
                "--objective",
                "yield_score=60",
                "--objective",
                "waste_score=12",
            ],
            "Pass either --objective-value or --objective",
        ),
        (
            [
                "--objective",
                "yield_score=60",
                "--objective",
                "waste_score=12",
                "--actual-cost",
                "3.2",
            ],
            "--actual-cost requires a config with a cost section",
        ),
    ],
)
def test_multi_objective_cli_mark_observed_failures_are_byte_atomic(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    objective_args: list[str],
    expected_error: str,
) -> None:
    config_path = write_multi_objective_config(tmp_path / "multi.yaml")
    cfg = multi_objective_config()
    log_path = write_log(tmp_path / "multi.csv", cfg, multi_objective_log(cfg))
    before = log_path.read_bytes()

    assert (
        run(
            [
                "mark-observed",
                *base_args(config_path, log_path),
                "--row-id",
                "suggested_0",
                *objective_args,
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error:" in captured.err
    assert expected_error in captured.err
    assert log_path.read_bytes() == before


def test_multi_objective_cli_mark_observed_accepts_actual_cost_for_cost_config(
    tmp_path: Path,
) -> None:
    config_path = write_multi_objective_cost_config(tmp_path / "multi_cost.yaml")
    cfg = multi_objective_config(cost=True)
    log_path = write_log(tmp_path / "multi_cost.csv", cfg, multi_objective_log(cfg))

    assert (
        run(
            [
                "mark-observed",
                *base_args(config_path, log_path),
                "--row-id",
                "suggested_0",
                "--objective",
                "yield_score=70",
                "--objective",
                "waste_score=14",
                "--actual-cost",
                "2.5",
            ]
        )
        == 0
    )

    df = load_campaign_log(log_path, cfg)
    row = df.loc[df["row_id"] == "suggested_0"].iloc[0]
    assert row["status"] == "observed"
    assert float(row["yield_score"]) == pytest.approx(70.0)
    assert float(row["waste_score"]) == pytest.approx(14.0)
    assert float(row["cost_actual"]) == pytest.approx(2.5)


def test_summary_status_next_action_and_report_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg))

    assert run(["summary", *base_args(config_path, log_path)]) == 0
    summary_out = capsys.readouterr().out
    assert "campaign_status" in summary_out
    assert "ready_for_bo" in summary_out

    assert run(["status", *base_args(config_path, log_path)]) == 0
    status_out = capsys.readouterr().out
    assert status_out == "ready_for_bo\n"

    assert run(["next-action", *base_args(config_path, log_path)]) == 0
    action_out = capsys.readouterr().out
    assert "suggest_bo" in action_out
    assert "ready_for_bo" in action_out

    assert run(["report", *base_args(config_path, log_path)]) == 0
    report_out = capsys.readouterr().out
    assert "BO Forge Campaign Report" in report_out
    assert "Best Raw Observation" in report_out


def test_cost_summary_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = write_cost_review_config(tmp_path / "campaign.yaml")
    cfg = cost_review_config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, cost_review_log(cfg))

    assert run(["cost-summary", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert "total_observed_cost" in captured.out
    assert "accepted_pending_cost" in captured.out
    assert "budget_remaining" in captured.out


def test_replicate_summary_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = write_replicate_config(tmp_path / "campaign.yaml")
    cfg = replicate_config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, replicate_log(cfg))

    assert run(["replicate-summary", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert "replicate_group" in captured.out
    assert "objective_mean" in captured.out
    assert "group_0" in captured.out


def test_report_output_uses_export_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg))
    report_path = tmp_path / "reports" / "latest.txt"

    assert run(["report", *base_args(config_path, log_path), "--output", str(report_path)]) == 0

    captured = capsys.readouterr()
    assert captured.out == f"Wrote campaign report: {report_path}\n"
    assert report_path.exists()
    assert "BO Forge Campaign Report" in report_path.read_text(encoding="utf-8")


def test_report_output_write_failure_returns_clear_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg))
    report_path = output_under_file_parent(tmp_path, "latest.txt")

    assert run(["report", *base_args(config_path, log_path), "--output", str(report_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"Error: Could not write campaign report '{report_path}'" in captured.err


def test_suggest_without_append_does_not_change_csv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config())
    before_csv = log_path.read_text(encoding="utf-8")

    assert run(["suggest", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert "Generated 1 suggestion(s)." in captured.out
    assert "row_id" in captured.out
    assert log_path.read_text(encoding="utf-8") == before_csv


def test_suggest_append_changes_csv_but_does_not_mark_observed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config())

    assert run(["suggest", *base_args(config_path, log_path), "--append"]) == 0

    captured = capsys.readouterr()
    assert "Generated 1 suggestion(s)." in captured.out
    assert f"Appended suggestions to campaign log: {log_path}" in captured.out
    df = pd.read_csv(log_path)
    assert len(df) == 1
    assert df.loc[0, "status"] == "suggested"
    assert pd.isna(df.loc[0, "score"])


def test_multi_fidelity_cli_suggest_append_writes_qmfkg_row(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_fidelity_config(tmp_path / "fidelity.yaml")
    cfg = fidelity_config()
    log_path = write_log(tmp_path / "fidelity.csv", cfg, fidelity_observed_log(cfg))

    assert run(["suggest", *base_args(config_path, log_path), "--append"]) == 0

    captured = capsys.readouterr()
    assert "Generated 1 suggestion(s)." in captured.out
    assert f"Appended suggestions to campaign log: {log_path}" in captured.out
    df = load_campaign_log(log_path, cfg)
    row = df.iloc[-1]
    assert row["status"] == "suggested"
    assert row["source"] == "qmf_kg"
    assert float(row["x"]) >= 0.0
    assert float(row["x"]) <= 1.0
    assert float(row["fidelity"]) >= 0.2
    assert float(row["fidelity"]) <= 1.0
    assert row["activity"] == ""
    assert list(df.columns) == canonical_columns(cfg)


def test_multi_fidelity_cli_fidelity_summary_outputs_table(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_fidelity_config(tmp_path / "fidelity.yaml")
    cfg = fidelity_config()
    log_path = write_log(tmp_path / "fidelity.csv", cfg, fidelity_observed_log(cfg))

    assert run(["fidelity-summary", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert "fidelity_variable" in captured.out
    assert "target_fidelity_observed_rows" in captured.out
    assert "best_target_fidelity_row_id" in captured.out
    assert "mf_obs_3" in captured.out


def test_multi_fidelity_cli_fidelity_summary_counts_review_blocking_qmfkg(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_fidelity_config(tmp_path / "fidelity.yaml", review=True)
    cfg = fidelity_config(review=True)
    rows = [
        {
            "row_id": "target_0",
            "iteration": 0,
            "status": "observed",
            "source": "manual",
            "review_status": "accepted",
            "review_note": "",
            "x": 0.5,
            "fidelity": 1.0,
            "activity": 1.2,
            "predicted_mean": "",
            "predicted_std": "",
            "acquisition": "",
        },
        {
            "row_id": "pending_qmfkg",
            "iteration": 1,
            "status": "suggested",
            "source": "qmf_kg",
            "review_status": "pending",
            "review_note": "",
            "x": 0.2,
            "fidelity": 0.8,
            "activity": "",
            "predicted_mean": 1.1,
            "predicted_std": 0.1,
            "acquisition": 0.2,
        },
        {
            "row_id": "accepted_qmfkg",
            "iteration": 2,
            "status": "suggested",
            "source": "qmf_kg",
            "review_status": "accepted",
            "review_note": "",
            "x": 0.3,
            "fidelity": 0.8,
            "activity": "",
            "predicted_mean": 1.2,
            "predicted_std": 0.1,
            "acquisition": 0.3,
        },
        {
            "row_id": "rejected_qmfkg",
            "iteration": 3,
            "status": "suggested",
            "source": "qmf_kg",
            "review_status": "rejected",
            "review_note": "",
            "x": 0.4,
            "fidelity": 0.8,
            "activity": "",
            "predicted_mean": 1.3,
            "predicted_std": 0.1,
            "acquisition": 0.4,
        },
        {
            "row_id": "deferred_qmfkg",
            "iteration": 4,
            "status": "suggested",
            "source": "qmf_kg",
            "review_status": "deferred",
            "review_note": "",
            "x": 0.6,
            "fidelity": 0.8,
            "activity": "",
            "predicted_mean": 1.4,
            "predicted_std": 0.1,
            "acquisition": 0.5,
        },
    ]
    log_path = write_log(
        tmp_path / "fidelity.csv",
        cfg,
        pd.DataFrame(rows, columns=canonical_columns(cfg)),
    )

    assert run(["fidelity-summary", *base_args(config_path, log_path)]) == 0

    captured = capsys.readouterr()
    assert "pending_qmfkg_suggestions" in captured.out
    pending_line = next(
        line for line in captured.out.splitlines() if "pending_qmfkg_suggestions" in line
    )
    assert pending_line.split()[-1] == "2"


def test_multi_fidelity_cli_fidelity_summary_rejects_non_fidelity_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config(), observed_log(config()))

    assert run(["fidelity-summary", *base_args(config_path, log_path)]) == 1

    captured = capsys.readouterr()
    assert "fidelity-summary requires a multi-fidelity config" in captured.err


def test_multi_fidelity_cli_plot_fidelity_diagnostics_writes_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_fidelity_config(tmp_path / "fidelity.yaml")
    cfg = fidelity_config()
    log_path = write_log(tmp_path / "fidelity.csv", cfg, fidelity_observed_log(cfg))
    output_path = tmp_path / "reports" / "fidelity.png"

    assert (
        run(
            [
                "plot",
                *base_args(config_path, log_path),
                "--kind",
                "fidelity-diagnostics",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert f"Wrote fidelity-diagnostics plot: {output_path}" in captured.out
    assert output_path.exists()


def test_multi_fidelity_cli_batch_size_failure_has_specific_hint_without_mutation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_fidelity_config(tmp_path / "fidelity.yaml")
    cfg = fidelity_config()
    log_path = write_log(tmp_path / "fidelity.csv", cfg, fidelity_observed_log(cfg))
    before = log_path.read_bytes()

    assert run(["suggest", *base_args(config_path, log_path), "--batch-size", "2", "--append"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "qMFKG model-based suggestions support batch_size=1" in captured.err
    assert "Hint: Use --batch-size 1 for model-based qMFKG suggestions." in captured.err
    assert log_path.read_bytes() == before


def test_suggest_output_and_append_writes_output_and_log(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config())
    output_path = tmp_path / "exports" / "suggestions.csv"

    assert (
        run(
            [
                "suggest",
                *base_args(config_path, log_path),
                "--output",
                str(output_path),
                "--append",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert f"Wrote suggestions CSV: {output_path}" in captured.out
    assert f"Appended suggestions to campaign log: {log_path}" in captured.out
    suggestions = pd.read_csv(output_path)
    log = pd.read_csv(log_path)
    assert len(suggestions) == 1
    assert len(log) == 1
    assert suggestions.loc[0, "row_id"] == log.loc[0, "row_id"]


def test_mixed_suggest_append_writes_valid_mixed_row(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_mixed_config(tmp_path / "mixed.yaml", initial_design_size=4)
    cfg = mixed_config(initial_design_size=4)
    log_path = write_log(tmp_path / "mixed.csv", cfg)

    assert run(["suggest", *base_args(config_path, log_path), "--append"]) == 0

    captured = capsys.readouterr()
    assert "Generated 1 suggestion(s)." in captured.out
    df = load_campaign_log(log_path, cfg)
    assert len(df) == 1
    assert df.loc[0, "status"] == "suggested"
    assert df.loc[0, "solvent"] in {"MeCN", "EtOH"}


def test_cost_review_suggest_append_writes_cost_and_review_columns(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_cost_review_config(tmp_path / "campaign.yaml", initial_design_size=3)
    cfg = cost_review_config(initial_design_size=3)
    log_path = write_log(tmp_path / "campaign.csv", cfg)

    assert run(["suggest", *base_args(config_path, log_path), "--append"]) == 0

    captured = capsys.readouterr()
    assert "Generated 1 suggestion(s)." in captured.out
    df = load_campaign_log(log_path, cfg)
    assert df.loc[0, "review_status"] == "pending"
    assert float(df.loc[0, "cost_estimate"]) > 0
    assert df.loc[0, "utility"] == ""


def test_suggest_output_write_failure_returns_clear_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config())
    output_path = output_under_file_parent(tmp_path, "suggestions.csv")
    before_csv = log_path.read_text(encoding="utf-8")

    assert run(["suggest", *base_args(config_path, log_path), "--output", str(output_path)]) == 1

    captured = capsys.readouterr()
    assert "Generated 1 suggestion(s)." in captured.out
    assert f"Error: Could not write suggestions CSV '{output_path}'" in captured.err
    assert log_path.read_text(encoding="utf-8") == before_csv


def test_mark_observed_resolves_only_specified_pending_suggestion(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config(initial_design_size=3))
    assert run(["suggest", *base_args(config_path, log_path), "--batch-size", "2", "--append"]) == 0
    capsys.readouterr()
    pending = pd.read_csv(log_path)
    row_id = str(pending.loc[0, "row_id"])
    other_row_id = str(pending.loc[1, "row_id"])

    assert (
        run(
            [
                "mark-observed",
                *base_args(config_path, log_path),
                "--row-id",
                row_id,
                "--objective-value",
                "1.23",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out == f"Marked row {row_id} as observed in campaign log: {log_path}\n"
    df = pd.read_csv(log_path)
    assert df.loc[df["row_id"] == row_id, "status"].iloc[0] == "observed"
    assert float(df.loc[df["row_id"] == row_id, "score"].iloc[0]) == pytest.approx(1.23)
    assert df.loc[df["row_id"] == other_row_id, "status"].iloc[0] == "suggested"


def test_review_and_mark_observed_with_actual_cost(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_cost_review_config(tmp_path / "campaign.yaml")
    cfg = cost_review_config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, cost_review_log(cfg))

    assert (
        run(
            [
                "review",
                *base_args(config_path, log_path),
                "--row-id",
                "suggested_0",
                "--decision",
                "accept",
                "--note",
                " approved ",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out == (
        f"Reviewed row suggested_0 as accept in campaign log: {log_path}\n"
    )

    assert (
        run(
            [
                "mark-observed",
                *base_args(config_path, log_path),
                "--row-id",
                "suggested_0",
                "--objective-value",
                "1.8",
                "--actual-cost",
                "1.7",
            ]
        )
        == 0
    )

    df = load_campaign_log(log_path, cfg)
    row = df.loc[df["row_id"] == "suggested_0"].iloc[0]
    assert row["status"] == "observed"
    assert row["review_note"] == "approved"
    assert float(row["score"]) == pytest.approx(1.8)
    assert float(row["cost_actual"]) == pytest.approx(1.7)


@pytest.mark.parametrize("kind", ["progress", "diagnostics"])
def test_plot_writes_nested_output_path(
    kind: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg))
    output_path = tmp_path / "figures" / f"{kind}.png"
    before_csv = log_path.read_text(encoding="utf-8")

    assert (
        run(
            [
                "plot",
                *base_args(config_path, log_path),
                "--kind",
                kind,
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out == f"Wrote {kind} plot: {output_path}\n"
    assert output_path.exists()
    assert log_path.read_text(encoding="utf-8") == before_csv


def test_plot_cost_progress_writes_nested_output_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_cost_review_config(tmp_path / "campaign.yaml")
    cfg = cost_review_config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, cost_review_log(cfg))
    output_path = tmp_path / "figures" / "cost-progress.png"
    before_csv = log_path.read_text(encoding="utf-8")

    assert (
        run(
            [
                "plot",
                *base_args(config_path, log_path),
                "--kind",
                "cost-progress",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out == f"Wrote cost-progress plot: {output_path}\n"
    assert output_path.exists()
    assert log_path.read_text(encoding="utf-8") == before_csv


def test_plot_replicates_writes_nested_output_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_replicate_config(tmp_path / "campaign.yaml")
    cfg = replicate_config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, replicate_log(cfg))
    output_path = tmp_path / "figures" / "replicates.png"
    before_csv = log_path.read_text(encoding="utf-8")

    assert (
        run(
            [
                "plot",
                *base_args(config_path, log_path),
                "--kind",
                "replicates",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out == f"Wrote replicates plot: {output_path}\n"
    assert output_path.exists()
    assert log_path.read_text(encoding="utf-8") == before_csv


def test_plot_stage_diagnostics_writes_nested_output_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path("configs/13_structured_campaign_core.yaml")
    log_path = tmp_path / "structured.csv"
    log_path.write_bytes(
        Path("examples/13_structured_campaign_core_campaign_log.csv").read_bytes()
    )
    output_path = tmp_path / "figures" / "stage-diagnostics.png"
    before_csv = log_path.read_text(encoding="utf-8")

    assert (
        run(
            [
                "plot",
                *base_args(config_path, log_path),
                "--kind",
                "stage-diagnostics",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out == f"Wrote stage-diagnostics plot: {output_path}\n"
    assert output_path.exists()
    assert log_path.read_text(encoding="utf-8") == before_csv


@pytest.mark.parametrize("kind", ["progress", "diagnostics"])
def test_plot_output_write_failure_returns_clear_error(
    kind: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg))
    output_path = output_under_file_parent(tmp_path, f"{kind}.png")
    before_csv = log_path.read_text(encoding="utf-8")

    assert (
        run(
            [
                "plot",
                *base_args(config_path, log_path),
                "--kind",
                kind,
                "--output",
                str(output_path),
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"Error: Could not write {kind} plot '{output_path}'" in captured.err
    assert log_path.read_text(encoding="utf-8") == before_csv


def test_plot_cost_progress_output_write_failure_returns_clear_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_cost_review_config(tmp_path / "campaign.yaml")
    cfg = cost_review_config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, cost_review_log(cfg))
    output_path = output_under_file_parent(tmp_path, "cost-progress.png")
    before_csv = log_path.read_text(encoding="utf-8")

    assert (
        run(
            [
                "plot",
                *base_args(config_path, log_path),
                "--kind",
                "cost-progress",
                "--output",
                str(output_path),
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"Error: Could not write cost-progress plot '{output_path}'" in captured.err
    assert log_path.read_text(encoding="utf-8") == before_csv


def test_plot_replicates_output_write_failure_returns_clear_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_replicate_config(tmp_path / "campaign.yaml")
    cfg = replicate_config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, replicate_log(cfg))
    output_path = output_under_file_parent(tmp_path, "replicates.png")
    before_csv = log_path.read_text(encoding="utf-8")

    assert (
        run(
            [
                "plot",
                *base_args(config_path, log_path),
                "--kind",
                "replicates",
                "--output",
                str(output_path),
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"Error: Could not write replicates plot '{output_path}'" in captured.err
    assert log_path.read_text(encoding="utf-8") == before_csv


def test_missing_required_arguments_return_argparse_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run(["validate"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage:" in captured.err
    assert "required" in captured.err


def test_unexpected_errors_are_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_unexpected_error(args: object) -> int:
        raise RuntimeError("unexpected boom")

    monkeypatch.setattr(cli, "_cmd_validate", raise_unexpected_error)

    with pytest.raises(RuntimeError, match="unexpected boom"):
        cli.run(["validate", "--config", "campaign.yaml", "--log", "campaign.csv"])
