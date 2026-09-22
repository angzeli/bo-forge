"""Route-specific timing, display, and provenance-safe mutation acceptance."""

import json

import pandas as pd
import pytest

from benchmarks import runner
from benchmarks.report import load_evidence, tables
from benchmarks.spec import schedule
from benchmarks.storage import write_json
from benchmarks.trial import run_trial
from tests._benchmark_support import controlled_suggest, spec_file
from tests.test_benchmark_multi_scores import multi_spec
from tests.test_benchmark_timing import (
    test_cancellation_finalizes_elapsed_after_worker_shutdown as cancellation_check,
)
from tests.test_benchmark_timing import (
    test_cleanup_failures_attach_notes_and_still_attempt_timing as cleanup_check,
)
from tests.test_benchmark_timing import (
    test_normal_terminal_timing_includes_startup_and_shutdown as timing_check,
)


@pytest.mark.parametrize("route", ["multi_objective", "multi_fidelity"])
@pytest.mark.parametrize("case", ["interrupt", "cleanup", "timeout", "startup"])
def test_existing_timing_contract_on_v3_trials(tmp_path, monkeypatch, route, case):
    import tests.test_benchmark_timing as timing

    def prepare(root):
        trial = schedule(multi_spec(route))[0]
        directory = root / trial["trial_id"]
        directory.mkdir()
        write_json(directory / "trial.json", trial)
        write_json(directory / "status.json", {"status": "pending", "completed_evaluations": 0})
        return directory, trial

    monkeypatch.setattr(timing, "prepare_trial", prepare)
    if case == "interrupt":
        cancellation_check(tmp_path, monkeypatch, KeyboardInterrupt)
    elif case == "cleanup":
        cleanup_check(tmp_path, monkeypatch, OSError, False)
    else:
        timing_check(tmp_path, monkeypatch, case, 10.0 if case == "timeout" else 2.0)


@pytest.mark.parametrize("route", ["multi_objective", "multi_fidelity"])
def test_cancelled_scored_trial_retains_partial_evidence(tmp_path, monkeypatch, route):
    interruption = KeyboardInterrupt("original interruption")

    def suggest(campaign):
        assert len(campaign.observed_data()) == 4
        raise interruption

    def execute(directory, timeout):
        try:
            run_trial(directory, suggest=suggest)
        except KeyboardInterrupt:
            runner._finalize_status(directory, "interrupted", "controlled cancellation", 5.)
            raise

    monkeypatch.setattr(runner, "_execute_trial", execute)
    output = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt) as caught:
        runner.run_suite(spec_file(tmp_path, multi_spec(route)), output)
    assert caught.value is interruption
    meta, traces, statuses = load_evidence(output)
    assert len(traces) == 4 and statuses.status.eq("interrupted").all()
    assert statuses.wall_seconds.tolist() == [5., 0., 0.]
    summary, curves = tables(meta, traces, statuses)
    assert summary.complete.eq(0).all() and curves["median"].isna().all()


@pytest.mark.parametrize("values", [{"branin": 1.}, {"branin": 1., "currin": float("nan")}])
def test_bad_coupled_observation_leaves_csv_unobserved(tmp_path, values):
    trial = schedule(multi_spec("multi_objective"))[0]
    directory = tmp_path / trial["trial_id"]
    directory.mkdir()
    write_json(directory / "trial.json", trial)
    with pytest.raises(ValueError):
        run_trial(directory, suggest=controlled_suggest, objective=lambda *_: values)
    frame = pd.read_csv(directory / "campaign.csv", keep_default_na=False)
    assert frame.status.tolist() == ["suggested"]
    assert frame.branin.tolist() == frame.currin.tolist() == [""]
    manifest = json.loads((directory / "campaign.csv.manifest.json").read_text())
    assert not any(e["operation"] == "mark_observed" for e in manifest["events"])


def test_mf_plot_coordinates_and_policy_labels_use_stored_evidence(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    from benchmarks.multi_report import _fidelity_figures
    from bo_forge import plot_style

    traces = pd.DataFrame({"trial_id": ["one"] * 2, "strategy": ["random"] * 2,
                           "seed": [0, 0], "status": ["complete"] * 2,
                           "evaluation": [1, 2], "cumulative_modeled_cost": [.4, 1.4],
                           "target_regret": [None, 3.], "oracle_target_regret": [2., 1.]})
    captured = []

    def capture(fig, axes, **kwargs):
        captured.extend((ax.get_title(), ax.get_xlabel(), ax.lines[0].get_xdata().tolist(),
                         ax.lines[0].get_ydata().tolist()) for ax in axes)

    monkeypatch.setattr(plot_style, "finalise_axes", capture)
    _fidelity_figures(traces, {"spec": {"seeds": [0]}}, tmp_path)
    assert all("target-only after init" in row[0] for row in captured)
    assert captured[0][2] == [.4, 1.4] and captured[1][2] == [1, 2]
    assert captured[1][3] == [2., 1.] and captured[2][3] == [2., 1.]
    assert not plt.get_fignums()
