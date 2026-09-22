"""Independent scientific checks for vector and target-fidelity scoring."""

import copy
import math

import pytest
import torch

from benchmarks.definitions import campaign_config
from benchmarks.designs import Proposals
from benchmarks.multi import evaluate, problem_for_route
from benchmarks.multi_scores import hypervolume_2d, is_target, mf_score, mo_score, pareto_indices
from benchmarks.problems import problem_for, true_value
from benchmarks.spec import V3_ROUTES, load_spec, schedule, validate_spec
from bo_forge.config import parse_campaign_config
from tests._benchmark_support import ROOT


def multi_spec(route, size="smoke"):
    return load_spec(ROOT / f"benchmarks/specs/{route}_{size}.yaml")


def test_v3_budgets_settings_and_scientific_configs():
    smoke = [t for route in V3_ROUTES for t in schedule(multi_spec(route))]
    standard = [t for route in V3_ROUTES for t in schedule(multi_spec(route, "standard"))]
    assert len(smoke) == 6 and sum(t["evaluations"] for t in smoke) == 36
    assert len(standard) == 30 and sum(t["evaluations"] for t in standard) == 600
    assert all(t["timeout_seconds"] == 600 for t in smoke + standard)
    for trial in smoke + standard:
        config = parse_campaign_config(campaign_config(trial))
        assert config.bo.batch_size == 1 and config.model.profile == "default"
        if trial["route"] == "multi_objective":
            assert config.objective_names == ["branin", "currin"]
            assert [o.direction for o in config.objectives] == ["minimize"] * 2
            assert [o.reference_point for o in config.objectives] == [18, 6]
        else:
            assert config.fidelity.variable == "x3"
            assert config.fidelity.fixed_cost == .25 and config.fidelity.fidelity_cost_weight == .75
            assert config.fidelity.num_fantasies == (4 if trial["evaluations"] == 6 else 8)
            assert config.fidelity.optimizer_maxiter == (50 if trial["evaluations"] == 6 else 100)


@pytest.mark.parametrize("route", V3_ROUTES)
@pytest.mark.parametrize("field", ["schema_version", "route", "problems", "modes",
                                  "baseline_policy", "initialization_policy"])
def test_reject_unsupported_v3_protocols(route, field):
    spec = multi_spec(route)
    replacements = {"schema_version": 2, "route": "pending_noisy", "modes": ["noisy"],
                    "problems": [{"name": "branin", "noise_std": 0, "optimum_tolerance": 1e-6}],
                    "baseline_policy": "equal_cost", "initialization_policy": "random"}
    spec[field] = replacements[field]
    with pytest.raises(ValueError):
        validate_spec(spec)


@pytest.mark.parametrize("field,value", [("target", .9), ("fixed_cost", .5),
                                       ("fidelity_cost_weight", 1), ("num_fantasies", 0),
                                       ("optimizer_maxiter", True)])
def test_reject_invalid_fidelity_protocol(field, value):
    spec = multi_spec("multi_fidelity")
    spec["fidelity"][field] = value
    with pytest.raises(ValueError):
        validate_spec(spec)


def test_hand_computable_hypervolume_ties_and_reference_exclusion():
    values = [[1, 4], [2, 2], [4, 1], [2, 2], [3, 3], [0, 6]]
    # Disjoint rectangle strips: (2-1)*(5-4) + (4-2)*(5-2) + (5-4)*(5-1) = 11.
    assert hypervolume_2d(values, [5, 5]) == 11
    assert pareto_indices(values) == [0, 1, 2, 3, 5]
    assert hypervolume_2d([[5, 5], [6, 6]], [5, 5]) == 0
    from botorch.utils.multi_objective.hypervolume import Hypervolume

    hv = Hypervolume(ref_point=torch.tensor([-5., -5.], dtype=torch.double))
    assert hv.compute(-torch.tensor(values, dtype=torch.double)) == 11
    trial = schedule(multi_spec("multi_objective"))[0]
    trial["reference_point"] = [5, 5]
    rows = []
    for i, value in enumerate(values):
        outcome = dict(zip(["branin", "currin"], value, strict=True))
        rows.append({"row_id": str(i), **mo_score(rows, outcome,
                                                trial, str(i))})
    assert rows[-1]["hypervolume"] == 11 and rows[-1]["pareto_count"] == 5
    assert rows[-1]["pareto_row_ids"] == ["0", "1", "2", "3", "5"]


@pytest.mark.parametrize("outcome", [{"branin": 1}, {"branin": 1, "currin": math.nan},
                                    {"branin": math.inf, "currin": 2},
                                    {"branin": 1, "currin": 2, "extra": 3}])
def test_coupled_missing_nonfinite_or_extra_outcomes_rejected(outcome):
    trial = schedule(multi_spec("multi_objective"))[0]
    with pytest.raises(ValueError):
        mo_score([], outcome, trial, "row")


def test_fidelity_quality_uses_only_observed_targets_and_separate_oracle():
    trial = schedule(multi_spec("multi_fidelity"))[0]
    rows = []
    for value, projected, s in [(10., 10., 1.), (-100., 2., .2), (5., 5., 1.)]:
        rows.append(mf_score(rows, value, projected, [0., 1., s], trial))
    assert [r["target_observed_count"] for r in rows] == [1, 1, 2]
    assert [r["target_regret"] for r in rows] == pytest.approx([9.602113, 9.602113, 4.602113])
    assert rows[-1]["oracle_target_regret"] == pytest.approx(1.602113)
    assert [r["modeled_evaluation_cost"] for r in rows] == [1., .4, 1.]
    assert rows[-1]["cumulative_modeled_cost"] == 2.4
    empty = mf_score([], -100., 2., [0., 1., 0.], trial)
    assert empty["target_regret"] is None and empty["modeled_evaluation_cost"] == .25
    assert is_target(1 - 5e-10, trial) and not is_target(.99, trial)
    with pytest.raises(ValueError, match="regret"):
        mf_score([], 0., 0., [0., 1., 1.], trial)


def test_native_augmented_bounds_and_target_projection():
    augmented, branin = problem_for_route("multi_fidelity"), problem_for("branin")
    assert augmented.bounds.tolist() == [[-5., 0., 0.], [10., 15., 1.]]
    for point in ([-5., 0.], [10., 15.], [math.pi, 2.275], [0., 7.]):
        expected = true_value(branin, point)
        assert evaluate(augmented, point + [1.]) == pytest.approx(expected, abs=1e-12)


def test_shared_initial_target_policy_and_baseline_continuation():
    trials = schedule(multi_spec("multi_fidelity"))
    initials = []
    for trial in trials:
        sampler, points = Proposals(trial), []
        for _ in range(4):
            points.append(sampler.draw("sobol", points))
        initials.append(copy.deepcopy(points))
        assert [p[-1] for p in points[:2]] == [1., 1.]
        assert all(0 < p[-1] < 1 for p in points[2:])
        for _ in range(2):
            point = sampler.draw("random" if trial["strategy"] == "random" else "sobol", points)
            points.append(point)
            if trial["strategy"] != "bo":
                assert point[-1] == 1.
    assert initials[0] == initials[1] == initials[2]
