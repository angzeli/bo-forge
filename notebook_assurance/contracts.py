"""Completion contracts for the source-only notebook execution runner.

Campaign paths are relative to the extracted source root. Notebooks 23/24 need
the latest pre-cleanup workspace copied to captures/<temporary-directory-name>.
The runner, which knows its runtime tmp path, separately verifies final cleanup.
"""

from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path


def _campaign(config, stem, reports, *, observed=15, pending=0, objectives=1,
              cell_timeout=600, notebook_timeout=1800, inline_figures=0):
    return {
        "config": f"configs/{config}.yaml",
        "log": f"examples/{stem}_working_log.csv",
        "suggestions": f"examples/{stem}_latest_suggestions.csv",
        "artifacts": [f"reports/{name}" for name in reports],
        "observed": observed, "pending": pending, "objectives": objectives,
        "cell_timeout": cell_timeout, "notebook_timeout": notebook_timeout,
        "inline_figures": inline_figures,
    }


CONTRACTS = {
    "01_maximisation_logei_campaign.ipynb": _campaign(
        "01_simple_2d_maximise_logei", "01_simple_2d_maximise_logei",
        ["01_simple_2d_maximise_logei_campaign_report.txt"], inline_figures=2),
    "02_minimisation_qlogei_campaign.ipynb": _campaign(
        "02_simple_2d_minimise_qlogei", "02_simple_2d_minimise_qlogei",
        ["02_simple_2d_minimise_qlogei_campaign_report.txt"], inline_figures=2),
    "03_three_variable_campaign.ipynb": _campaign(
        "03_simple_3d_maximise_logei", "03_simple_3d_maximise_logei",
        ["03_simple_3d_campaign_report.txt", "03_simple_3d_progress.pdf",
         "03_simple_3d_diagnostics.pdf"]),
    "04_cli_four_variable_campaign.ipynb": _campaign(
        "04_simple_4d_maximise_logei", "04_simple_4d_maximise_logei",
        ["04_cli_4d_campaign_report.txt", "04_cli_4d_progress.pdf",
         "04_cli_4d_diagnostics.pdf"]),
    "05_mixed_variable_campaign.ipynb": _campaign(
        "05_simple_mixed_logei", "05_simple_mixed_logei",
        ["05_simple_mixed_campaign_report.txt", "05_simple_mixed_progress.pdf",
         "05_simple_mixed_diagnostics.pdf"]),
    "06_constrained_mixed_campaign.ipynb": _campaign(
        "06_mixed_constrained_logei", "06_mixed_constrained_logei",
        ["06_constrained_mixed_campaign_report.txt", "06_constrained_mixed_progress.pdf",
         "06_constrained_mixed_diagnostics.pdf"]),
    "07_cost_aware_human_review_campaign.ipynb": _campaign(
        "07_cost_aware_human_review_logei", "07_cost_aware_human_review",
        ["07_cost_aware_human_review_report.txt", "07_cost_aware_human_review_progress.pdf",
         "07_cost_aware_human_review_cost_progress.pdf"], pending=2),
    "08_replicate_aware_campaign.ipynb": {
        **_campaign("08_replicate_aware_logei", "08_replicate_aware",
                    ["08_replicate_aware_report.txt", "08_replicate_aware_progress.pdf",
                     "08_replicate_aware_replicates.pdf"], observed=None),
        "replicate_groups": 15,
    },
    "10_multi_objective_qlogehvi_campaign.ipynb": _campaign(
        "10_multi_objective_mixed_constrained_qlogehvi", "10_multi_objective_mixed_constrained",
        ["10_multi_objective_campaign_report.txt", "10_multi_objective_pareto.pdf",
         "10_multi_objective_hypervolume.pdf"], objectives=2),
    "11_four_objective_qlogehvi_campaign.ipynb": _campaign(
        "11_four_objective_mixed_constrained_qlogehvi", "11_four_objective_mixed_constrained",
        ["11_four_objective_campaign_report.txt", "11_four_objective_pairwise_pareto.pdf",
         "11_four_objective_parallel_pareto.pdf", "11_four_objective_hypervolume.pdf"],
        observed=50, objectives=4, cell_timeout=3300, notebook_timeout=3600),
    "12_cost_aware_multi_objective_qlogehvi_campaign.ipynb": _campaign(
        "12_cost_aware_multi_objective_qlogehvi", "12_cost_aware_multi_objective",
        ["12_cost_aware_multi_objective_report.txt",
         "12_cost_aware_multi_objective_cost_progress.pdf",
         "12_cost_aware_multi_objective_pareto.pdf"], objectives=3),
    "14_structured_campaign_tutorial.ipynb": _campaign(
        "14_structured_campaign_tutorial", "14_structured_campaign_tutorial",
        ["14_structured_campaign_tutorial_report.txt",
         "14_structured_campaign_tutorial_stage_diagnostics.png"], observed=6),
    "15_multi_fidelity_qmfkg_campaign.ipynb": _campaign(
        "15_multi_fidelity_qmfkg", "15_multi_fidelity_qmfkg",
        ["15_multi_fidelity_qmfkg_report.txt", "15_multi_fidelity_qmfkg_progress.png",
         "15_multi_fidelity_qmfkg_diagnostics.png",
         "15_multi_fidelity_qmfkg_fidelity_diagnostics.png"],
        cell_timeout=3300, notebook_timeout=3600),
    "16_contextual_logei_campaign.ipynb": _campaign(
        "16_contextual_logei", "16_contextual_logei",
        ["16_contextual_logei_report.txt", "16_contextual_logei_progress.png",
         "16_contextual_logei_diagnostics.png", "16_contextual_logei_context_diagnostics.png"]),
    "17_model_profile_logei_campaign.ipynb": _campaign(
        "17_model_profile_logei", "17_model_profile_logei",
        ["17_model_profile_report.md", "17_model_profile_diagnostics.png"]),
    "18_noisy_pending_qlognei_campaign.ipynb": _campaign(
        "18_noisy_pending_qlognei", "18_noisy_pending_qlognei",
        ["18_noisy_pending_qlognei_report.md", "18_noisy_pending_qlognei_progress.png",
         "18_noisy_pending_qlognei_diagnostics.png", "18_noisy_pending_qlognei_pending.png"],
        pending=1),
    "20_contextual_cost_review_logei_campaign.ipynb": _campaign(
        "20_contextual_cost_review_logei", "20_contextual_cost_review",
        ["20_contextual_cost_review_report.md", "20_contextual_cost_review_context_diagnostics.png",
         "20_contextual_cost_review_cost_progress.png", "20_contextual_cost_review_progress.png"]),
    "22_discrete_multi_fidelity_qmfkg_campaign.ipynb": _campaign(
        "22_discrete_multi_fidelity_qmfkg", "22_discrete_multi_fidelity_qmfkg",
        ["22_discrete_multi_fidelity_qmfkg_report.txt",
         "22_discrete_multi_fidelity_qmfkg_progress.png",
         "22_discrete_multi_fidelity_qmfkg_fidelity_diagnostics.png",
         "22_discrete_multi_fidelity_qmfkg_fidelity_progress.png"],
        cell_timeout=3300, notebook_timeout=3600),
    "23_predictive_diagnostics.ipynb": {
        "capture_prefix": "bo-forge-predictive-", "cell_timeout": 600, "notebook_timeout": 1800,
    },
    "24_closed_loop_benchmarks.ipynb": {
        "capture_prefix": "bo-forge-benchmark-notebook-",
        "cell_timeout": 3720, "notebook_timeout": 3900,
    },
}

PR_NOTEBOOKS = [
    "01_maximisation_logei_campaign.ipynb", "04_cli_four_variable_campaign.ipynb",
    "08_replicate_aware_campaign.ipynb", "12_cost_aware_multi_objective_qlogehvi_campaign.ipynb",
    "14_structured_campaign_tutorial.ipynb", "18_noisy_pending_qlognei_campaign.ipynb",
    "20_contextual_cost_review_logei_campaign.ipynb",
    "22_discrete_multi_fidelity_qmfkg_campaign.ipynb",
    "23_predictive_diagnostics.ipynb", "24_closed_loop_benchmarks.ipynb",
]


def discover(root: Path) -> list[str]:
    """Fail closed on missing, renamed, or unregistered notebooks."""
    names = sorted(path.name for path in (Path(root) / "notebooks").glob("*.ipynb"))
    expected = set(CONTRACTS)
    if len(expected) != 20 or set(names) != expected:
        raise ValueError(f"Notebook registry mismatch: missing={sorted(expected - set(names))}; "
                         f"unregistered={sorted(set(names) - expected)}; registry={len(expected)}")
    return names


def select(root: Path, profile: str, selection=None) -> list[str]:
    """Select a stable registry-ordered profile or explicit filename subset."""
    names = discover(root)
    if profile not in {"pr", "full"}:
        raise ValueError(f"Unknown notebook profile: {profile!r}")
    allowed = names if profile == "full" else list(PR_NOTEBOOKS)
    if selection is None:
        return allowed
    requested = [selection] if isinstance(selection, str) else list(selection)
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(allowed):
        raise ValueError(f"Invalid notebook selection: {requested!r}")
    return [name for name in names if name in requested]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _artifact(path):
    _require(path.is_file() and path.stat().st_size > 0, f"Missing or empty artifact: {path}")
    try:
        if path.suffix == ".png":
            from PIL import Image

            with Image.open(path) as image:
                _require(image.format == "PNG", "Expected PNG format")
                image.verify()
            with Image.open(path) as image:
                image.load()
        elif path.suffix == ".pdf":
            _matplotlib_pdf(path.read_bytes())
    except (OSError, ValueError, SyntaxError) as exc:
        raise ValueError(f"Invalid image artifact: {path}: {exc}") from exc


def _matplotlib_pdf(data):
    """Check the classic xref/page structure emitted by Matplotlib, not rendering."""
    ending = re.search(rb"startxref\s+(\d+)\s+%%EOF\s*\Z", data)
    _require(data.startswith(b"%PDF-") and ending is not None, "Missing PDF trailer")
    offset = int(ending.group(1))
    table = data[offset:ending.start()].split(b"trailer", 1)
    _require(len(table) == 2 and table[0].startswith(b"xref\n"), "Invalid PDF xref")
    lines = table[0].splitlines()
    section = re.fullmatch(rb"0 (\d+)", lines[1]) if len(lines) > 1 else None
    _require(section is not None, "Invalid PDF xref section")
    count = int(section.group(1))
    _require(len(lines[2:]) == count, "Incomplete PDF xref entries")
    objects = []
    for number, entry in enumerate(lines[2:]):
        fields = entry.split()
        _require(len(fields) == 3 and fields[2] in {b"n", b"f"}, "Invalid PDF xref entry")
        if fields[2] == b"n":
            position = int(fields[0])
            header = f"{number} {int(fields[1])} obj".encode()
            _require(0 <= position < offset and data[position:].startswith(header),
                     "PDF object offset mismatch")
            end = data.find(b"endobj", position, offset)
            _require(end >= 0, "Incomplete PDF object")
            objects.append(data[position:end].split(b"stream", 1)[0])
    pages = sum(bool(re.search(rb"/Type\s*/Page\b", obj)) for obj in objects)
    trees = [obj for obj in objects if re.search(rb"/Type\s*/Pages\b", obj)]
    total = re.search(rb"/Count\s+(\d+)", trees[0]) if len(trees) == 1 else None
    _require(pages > 0 and total is not None and int(total.group(1)) == pages,
             "PDF page tree is missing or inconsistent")


def _session(config, log):
    from bo_forge import CampaignSession

    _artifact(config)
    _artifact(log)
    campaign = CampaignSession.from_files(config, log)
    campaign.reload()
    campaign.validate()
    return campaign


def _context_sequence(observed, name):
    import numpy as np

    if not name.startswith(("16_", "20_")):
        return {}
    rows = observed.loc[observed.iteration.gt(0)].sort_values("iteration", kind="stable")
    cycle = [0.25, 0.5, 0.75]
    expected = ([0.5] + (cycle * 4)[:10] if name.startswith("20_") else (cycle * 4)[:11])
    values = rows.feedstock_acidity.astype(float).tolist()
    _require(len(values) == len(expected) and np.allclose(values, expected, rtol=0, atol=1e-12),
             "Requested context sequence was not preserved")
    return {"context_sequence": values}


def _campaign_checks(root, name, contract):
    import numpy as np
    import pandas as pd

    campaign = _session(root / contract["config"], root / contract["log"])
    observed = campaign.observed_data()
    pending = campaign.df.loc[campaign.df.status.eq("suggested")]
    if contract["observed"] is not None:
        _require(len(observed) == contract["observed"], "Unexpected observed row count")
    _require(len(pending) == contract["pending"], "Unexpected pending row count")
    _require(len(campaign.df) == len(observed) + len(pending), "Unexpected campaign row state")
    _require(len(campaign.config.objective_names) == contract["objectives"],
             "Unexpected objective count")
    _require(np.isfinite(observed[campaign.config.objective_names].to_numpy(dtype=float)).all(),
             "Non-finite observed objectives")
    context_checks = _context_sequence(observed, name)
    if "replicate_groups" in contract:
        _require(len(campaign.replicate_summary()) == contract["replicate_groups"],
                 "Unexpected replicate group count")
        _require(len(observed) >= 17, "Replicate seed observations were lost")
    if name.startswith("14_"):
        _require(observed.stage.value_counts().to_dict() == {"screening": 3, "refinement": 3},
                 "Expected three observations per stage")
    if name.startswith("07_"):
        _require(pending.review_status.value_counts().to_dict() == {"rejected": 1, "deferred": 1},
                 "Expected one rejected and one deferred suggestion")
    if name.startswith("18_"):
        _require(pending.row_id.tolist() == ["nei_pending_0"]
                 and pending.review_status.eq("accepted").all(),
                 "Original accepted pending suggestion was not preserved")
    if campaign.config.review.enabled:
        _require(observed.review_status.eq("accepted").all(), "Unapproved observed rows")
    suggestions_path = root / contract["suggestions"]
    _artifact(suggestions_path)
    suggestions = pd.read_csv(suggestions_path, keep_default_na=False)
    _require(not suggestions.empty and suggestions.row_id.is_unique
             and suggestions.status.eq("suggested").all(), "Invalid latest suggestions")
    recorded = observed.set_index("row_id")
    _require(set(suggestions.row_id) <= set(recorded.index), "Latest suggestions not observed")
    for column in campaign.config.variable_names:
        left = suggestions[column].astype(str).tolist()
        right = recorded.loc[suggestions.row_id, column].astype(str).tolist()
        if left != right:
            # CSV inference can render an integer-valued real as either 30 or 30.0.
            _require(np.allclose(pd.to_numeric(suggestions[column]),
                                 pd.to_numeric(recorded.loc[suggestions.row_id, column])),
                     f"Latest suggestion differs from recorded {column}")
    for relative in contract["artifacts"]:
        _artifact(root / relative)
    result = {"observed_rows": len(observed), "pending_rows": len(pending),
              "objectives": list(campaign.config.objective_names),
              "artifacts": [contract["log"], contract["suggestions"], *contract["artifacts"]]}
    if "replicate_groups" in contract:
        result["replicate_groups"] = len(campaign.replicate_summary())
    if name.startswith("14_"):
        result["stages"] = observed.stage.value_counts().to_dict()
    result.update(context_checks)
    return result


def _capture(captures, prefix, marker):
    matches = [path for path in captures.glob(f"{prefix}*")
               if path.is_dir() and (path / marker).is_file()]
    _require(len(matches) == 1,
             f"Expected one {prefix} snapshot with {marker}, found {len(matches)}")
    return matches[0]


def _predictive_checks(captures, contract):
    import numpy as np
    import pandas as pd

    workspace = _capture(captures, contract["capture_prefix"], "campaign.yaml")
    campaign = _session(workspace / "campaign.yaml", workspace / "observed.csv")
    _require(len(campaign.df) == len(campaign.observed_data()) == 20,
             "Predictive notebook requires 20 observations")
    _require(campaign.config.model.profile == "default", "Predictive campaign profile changed")
    _require(campaign.df.row_id.tolist() == [f"synthetic_{i:02d}" for i in range(20)],
             "Predictive input row identity changed")
    x = np.linspace(0.025, 0.975, 20)
    response = 10 + 3 * np.sin(2 * np.pi * x) + np.random.default_rng(0).normal(0, .2, 20)
    _require(np.allclose(campaign.df.x.astype(float), x)
             and np.allclose(campaign.df.response.astype(float), response),
             "Predictive input observations changed")
    output = workspace / "evaluation"
    filenames = {"summary.csv", "predictions.csv", "fold_outcomes.csv", "metadata.json",
                 "predictions.png", "residuals.png"}
    _require({p.name for p in output.iterdir()} == filenames, "Predictive export set differs")
    for filename in filenames:
        _artifact(output / filename)
    summary = pd.read_csv(output / "summary.csv")
    predictions = pd.read_csv(output / "predictions.csv")
    outcomes = pd.read_csv(output / "fold_outcomes.csv")
    metadata = json.loads((output / "metadata.json").read_text())
    _require(len(summary) == 2 and summary.model_profile.tolist() == ["default", "smooth"],
             "Expected two predictive profiles")
    for frame in (summary, predictions, outcomes):
        _require(frame.fit_status.eq("complete").all(), "Predictive fit incomplete or failed")
    _require(summary.evaluation_scope.eq("out_of_fold").all()
             and summary.observed_rows.eq(20).all() and summary.completed_folds.eq(3).all()
             and summary.total_folds.eq(3).all(), "Predictive summary counts differ")
    metrics = ["rmse", "mae", "mean_nlpd", "interval_coverage", "mean_interval_width"]
    _require(np.isfinite(summary[metrics].to_numpy(dtype=float)).all(),
             "Non-finite predictive metrics")
    _require(len(predictions) == 40 and len(outcomes) == 6, "Predictive row/fold counts differ")
    _require(not predictions.duplicated(["model_profile", "row_id"]).any()
             and not outcomes.duplicated(["model_profile", "fold"]).any(),
             "Duplicate predictive rows or folds")
    for key, expected in {
        "profiles": ["default", "smooth"], "folds": 3, "seed": 0, "requested_fit_count": 6,
        "observed_rows": 20, "evaluation_scope": "out_of_fold", "method": "shuffled_k_fold",
        "objective_name": "response", "prediction_units": "original_objective",
        "variance": "observation_inclusive",
    }.items():
        _require(metadata.get(key) == expected, f"Predictive metadata differs: {key}")
    groups = np.array_split(np.random.default_rng(0).permutation(20), 3)
    expected_membership = []
    for fold, indices in enumerate(groups, 1):
        held_ids = campaign.df.iloc[np.sort(indices)].row_id.tolist()
        train_ids = campaign.df.loc[~campaign.df.row_id.isin(held_ids), "row_id"].tolist()
        expected_membership.append({"fold": fold, "training_row_ids": train_ids,
                                    "held_out_row_ids": held_ids})
        for profile in ("default", "smooth"):
            rows = predictions.loc[predictions.model_profile.eq(profile)
                                   & predictions.fold.eq(fold)]
            outcome = outcomes.loc[outcomes.model_profile.eq(profile) & outcomes.fold.eq(fold)]
            _require(set(rows.row_id) == set(held_ids) and len(outcome) == 1,
                     "Predictive fold membership differs")
            _require(outcome.held_out_rows.iloc[0] == len(held_ids)
                     and outcome.training_rows.iloc[0] == len(train_ids),
                     "Predictive fold sizes differ")
    _require(metadata.get("fold_membership") == expected_membership,
             "Predictive metadata membership differs")
    _require(np.isfinite(predictions[["observed", "predicted_mean", "predicted_std"]]
                        .to_numpy(dtype=float)).all()
             and predictions.predicted_std.gt(0).all(), "Invalid held-out predictions")
    return {"observed_rows": 20, "profiles": ["default", "smooth"], "folds": 6,
            "predictions": 40, "capture": workspace.name, "artifacts": sorted(filenames)}


def _benchmark_checks(root, captures, contract):
    import pandas as pd

    from benchmarks.report import _markdown, load_evidence, tables
    from benchmarks.spec import load_spec

    workspace = _capture(captures, contract["capture_prefix"], "smoke/run.json")
    run = workspace / "smoke"
    metadata, traces, statuses = load_evidence(run)
    _require(metadata["spec"] == load_spec(root / "benchmarks/specs/smoke.yaml"),
             "Benchmark specification differs from notebook smoke")
    _require(metadata.get("status") == "complete" and len(statuses) == 6
             and statuses.status.eq("complete").all()
             and statuses.completed_evaluations.eq(6).all() and len(traces) == 36,
             "Benchmark requires six complete trials and 36 evaluations")
    for column in ("evidence_warning", "trace_warning", "timing_warning"):
        _require(statuses[column].fillna("").eq("").all(), f"Benchmark {column}")
    summary, trajectories = tables(metadata, traces, statuses)
    for directory in (run / "report", workspace / "regenerated-report"):
        for stem, expected in (("summary", summary), ("trajectories", trajectories),
                               ("traces", traces), ("trials", statuses)):
            path = directory / f"{stem}.csv"
            _artifact(path)
            # Compare the serialized view, preserving empty CSV fields and column order.
            reference = pd.read_csv(StringIO(expected.to_csv(index=False)))
            pd.testing.assert_frame_equal(pd.read_csv(path), reference, check_dtype=False,
                                          rtol=1e-10, atol=1e-12)
        report = directory / "report.md"
        _artifact(report)
        _require(report.read_text() == _markdown(metadata, summary, statuses),
                 "Benchmark report text differs from retained evidence")
        for mode in ("deterministic", "noisy"):
            _artifact(directory / "figures" / f"branin-{mode}.png")
    return {"trials": 6, "complete_trials": 6, "evaluations": 36,
            "capture": workspace.name, "reports": ["smoke/report", "regenerated-report"]}


def validate(root: Path, captures: Path, notebook_name: str) -> dict:
    """Reload persisted evidence, returning JSON-safe checks or raising ValueError."""
    from bo_forge.errors import BOForgeError

    if notebook_name not in CONTRACTS:
        raise ValueError(f"Unregistered notebook: {notebook_name!r}")
    contract = CONTRACTS[notebook_name]
    try:
        if notebook_name.startswith("23_"):
            checks = _predictive_checks(Path(captures), contract)
        elif notebook_name.startswith("24_"):
            checks = _benchmark_checks(Path(root), Path(captures), contract)
        else:
            checks = _campaign_checks(Path(root), notebook_name, contract)
    except (BOForgeError, OSError, ValueError, KeyError, TypeError, AttributeError,
            AssertionError) as exc:
        raise ValueError(f"{notebook_name}: {exc}") from exc
    return {"notebook": notebook_name, **checks}
