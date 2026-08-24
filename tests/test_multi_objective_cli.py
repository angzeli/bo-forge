"""Multi-objective CLI and edge-case plot tests."""

from tests._multi_objective_support import (
    CampaignSession,
    Path,
    canonical_columns,
    four_objective_config,
    multi_config,
    observed_four_objective_log,
    observed_multi_log,
    pd,
    pytest,
    suggest_next,
)


def test_pareto_parallel_plot_single_row(tmp_path: Path) -> None:
    cfg = four_objective_config()
    df = observed_four_objective_log(cfg).iloc[[0]].copy()
    campaign = CampaignSession(
        config_path=tmp_path / "config.yaml",
        log_path=tmp_path / "log.csv",
        config=cfg,
        df=df,
    )

    _, ax = campaign.plot_pareto_parallel(save_path=tmp_path / "single_parallel.png")

    assert (tmp_path / "single_parallel.png").exists()
    assert len(ax.lines) == 1
    assert list(ax.lines[0].get_ydata()) == pytest.approx([0.5, 0.5, 0.5, 0.5])

def test_empty_four_objective_plots_export(tmp_path: Path) -> None:
    cfg = four_objective_config()
    campaign = CampaignSession(
        config_path=tmp_path / "config.yaml",
        log_path=tmp_path / "log.csv",
        config=cfg,
        df=pd.DataFrame(columns=canonical_columns(cfg)),
    )

    for name, plotter in [
        ("pareto.png", campaign.plot_pareto),
        ("parallel.png", campaign.plot_pareto_parallel),
        ("hypervolume.png", campaign.plot_hypervolume),
    ]:
        path = tmp_path / name
        plotter(save_path=path)
        assert path.exists()

def test_cli_multi_objective_mark_observed_errors(tmp_path: Path) -> None:
    from bo_forge.cli import run

    cfg = multi_config(initial_design_size=10)
    config_path = tmp_path / "config.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text(
        """
campaign_name: multi
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
  - name: solvent
    type: categorical
    values: [MeCN, Water]
bo:
  acquisition: qlog_ehvi
  initial_design_size: 10
""",
        encoding="utf-8",
    )
    df = observed_multi_log(cfg)
    suggestions = suggest_next(cfg, df, batch_size=1)
    pd.concat([df, suggestions], ignore_index=True).to_csv(log_path, index=False)
    row_id = str(suggestions["row_id"].iloc[0])

    assert run(
        [
            "mark-observed",
            "--config",
            str(config_path),
            "--log",
            str(log_path),
            "--row-id",
            row_id,
            "--objective-value",
            "1.0",
        ]
    ) == 1
    assert run(
        [
            "mark-observed",
            "--config",
            str(config_path),
            "--log",
            str(log_path),
            "--row-id",
            row_id,
            "--objective",
            "yield_score=60",
        ]
    ) == 1
    assert run(
        [
            "mark-observed",
            "--config",
            str(config_path),
            "--log",
            str(log_path),
            "--row-id",
            row_id,
            "--objective",
            "yield_score=60",
            "--objective",
            "waste_score=12",
        ]
    ) == 0

def test_cli_pareto_commands_and_plots(tmp_path: Path) -> None:
    from bo_forge.cli import run

    cfg = four_objective_config()
    config_path = tmp_path / "config.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text(
        """
campaign_name: four_objective
objectives:
  - name: yield
    direction: maximize
    reference_point: 0.2
  - name: selectivity
    direction: maximize
    reference_point: 0.2
  - name: waste
    direction: minimize
    reference_point: 0.9
  - name: energy_use
    direction: minimize
    reference_point: 0.9
variables:
  - name: catalyst_loading
    type: continuous
    lower: 0.02
    upper: 0.20
  - name: reaction_time
    type: integer
    lower: 20
    upper: 90
  - name: base_equivalents
    type: discrete
    values: [0.5, 1.0, 1.5]
  - name: solvent
    type: categorical
    values: [MeCN, DMF, Water]
constraints:
  - name: water_needs_time
    expression: "solvent != 'Water' or reaction_time >= 45"
bo:
  acquisition: qlog_ehvi
  initial_design_size: 4
  raw_samples: 8
  num_restarts: 2
  mc_samples: 8
""",
        encoding="utf-8",
    )
    observed_four_objective_log(cfg).to_csv(log_path, index=False)

    pareto_path = tmp_path / "plots" / "pareto.png"
    parallel_path = tmp_path / "plots" / "parallel.png"

    common = ["--config", str(config_path), "--log", str(log_path)]
    assert run(["pareto-front", *common]) == 0
    assert run(["pareto-summary", *common]) == 0
    assert run(["plot", *common, "--kind", "pareto", "--output", str(pareto_path)]) == 0
    assert (
        run(["plot", *common, "--kind", "pareto-parallel", "--output", str(parallel_path)])
        == 0
    )
    assert pareto_path.exists()
    assert parallel_path.exists()

def test_cli_multi_objective_review_replicate_workflow(tmp_path: Path) -> None:
    from bo_forge.cli import run

    cfg = multi_config(review=True, replicates=True, initial_design_size=10)
    config_path = tmp_path / "config.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text(
        """
campaign_name: multi
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
  - name: solvent
    type: categorical
    values: [MeCN, Water]
review:
  enabled: true
replicates:
  enabled: true
bo:
  acquisition: qlog_ehvi
  initial_design_size: 10
  raw_samples: 8
  num_restarts: 2
  mc_samples: 8
""",
        encoding="utf-8",
    )
    df = observed_multi_log(cfg)
    df.to_csv(log_path, index=False)

    common = ["--config", str(config_path), "--log", str(log_path)]
    assert run(["suggest", *common, "--batch-size", "1", "--append"]) == 0
    written = pd.read_csv(log_path, keep_default_na=False)
    row_id = str(written.loc[written["status"] == "suggested", "row_id"].iloc[0])
    assert run(["review", *common, "--row-id", row_id, "--decision", "accept"]) == 0
    assert (
        run(
            [
                "mark-observed",
                *common,
                "--row-id",
                row_id,
                "--objective",
                "yield_score=62",
                "--objective",
                "waste_score=13",
            ]
        )
        == 0
    )
    assert run(["replicate-summary", *common]) == 0
    assert run(["pareto-front", *common]) == 0
    assert run(["pareto-summary", *common]) == 0

def test_cli_pareto_parallel_requires_three_objectives(tmp_path: Path) -> None:
    from bo_forge.cli import run

    cfg = multi_config()
    config_path = tmp_path / "config.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text(
        """
campaign_name: multi
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
  - name: solvent
    type: categorical
    values: [MeCN, Water]
bo:
  acquisition: qlog_ehvi
  initial_design_size: 3
""",
        encoding="utf-8",
    )
    observed_multi_log(cfg).to_csv(log_path, index=False)

    assert (
        run(
            [
                "plot",
                "--config",
                str(config_path),
                "--log",
                str(log_path),
                "--kind",
                "pareto-parallel",
                "--output",
                str(tmp_path / "parallel.png"),
            ]
        )
        == 1
    )
