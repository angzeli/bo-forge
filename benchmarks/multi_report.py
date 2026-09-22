"""Route-specific reporting without mixing objective scales or fidelity evidence levels."""

import pandas as pd

GROUPS = ["route", "problem", "mode", "strategy"]


def _metrics(route):
    return (["hypervolume", "pareto_count"] if route == "multi_objective" else
            ["target_regret", "oracle_target_regret", "cumulative_modeled_cost",
             "target_observed_count"])


def _quantiles(frame, metric):
    values = pd.to_numeric(frame[metric]).dropna()
    return {"median": values.median() if len(values) else None,
            "q25": values.quantile(.25) if len(values) else None,
            "q75": values.quantile(.75) if len(values) else None}


def multi_tables(metadata, traces, statuses):
    summary, trajectories = [], []
    for key, group in statuses.groupby(GROUPS, sort=False):
        identity = dict(zip(GROUPS, key, strict=True))
        ids = group.loc[group.status.eq("complete"), "trial_id"]
        relevant = traces.loc[traces.trial_id.isin(group.trial_id)]
        successful = relevant.loc[relevant.trial_id.isin(ids)]
        final = successful.loc[successful.evaluation.eq(metadata["spec"]["evaluations"])]
        item = {**identity, "scheduled": len(group), "complete": len(ids),
                **{state: int(group.status.eq(state).sum())
                   for state in ("failed", "timeout", "interrupted")},
                "completed_evaluations": int(group.completed_evaluations.sum()),
                "wall_seconds": (float(group.wall_seconds.sum())
                                 if group.wall_seconds.notna().all() else None),
                "known_wall_seconds": float(group.wall_seconds.sum()),
                "known_timing_trials": int(group.wall_seconds.notna().sum()),
                "unknown_timing_trials": int(group.wall_seconds.isna().sum())}
        for metric in _metrics(identity["route"]):
            item.update({f"final_{metric}_{stat}": value
                         for stat, value in _quantiles(final, metric).items()})
            _trajectory(trajectories, metadata, identity, group, relevant, successful, ids, metric)
        for field in ("suggestion_seconds", "objective_seconds", "oracle_seconds",
                      "mutation_seconds"):
            item[field] = float(pd.to_numeric(relevant[field]).sum())
        summary.append(item)
    return pd.DataFrame(summary), pd.DataFrame(trajectories)


def _trajectory(result, metadata, identity, group, relevant, successful, ids, metric):
    for evaluation in range(1, metadata["spec"]["evaluations"] + 1):
        at = successful.loc[successful.evaluation.eq(evaluation)]
        partial = relevant.loc[relevant.evaluation.eq(evaluation) & ~relevant.trial_id.isin(ids)]
        result.append({**identity, "evaluation": evaluation, "metric": metric,
                       "scheduled": len(group),
                       "contributing_complete": int(at[metric].notna().sum()),
                       "contributing_partial": int(partial[metric].notna().sum()),
                       **_quantiles(at, metric)})


def multi_markdown(metadata, summary, statuses):
    spec = metadata["spec"]
    metric = _metrics(spec["route"])[0]
    lines = [f"# {spec['name']} closed-loop benchmark", "",
             "Summaries are conditioned on successfully completed trials. Denominators retain "
             "every scheduled trial, including failures and timeouts. Partial trajectories stop "
             "at their last observation; nothing is extrapolated.", "",
             f"Baseline policy: `{spec['baseline_policy']}`. Initialization policy: "
             f"`{spec['initialization_policy']}`. Seed count: {len(spec['seeds'])}.", ""]
    if spec["route"] == "multi_objective":
        lines += ["Primary metric: dominated hypervolume, larger is better. Objective order is "
                  "branin, currin; both minimized; fixed user-space reference point [18, 6]. "
                  "Pareto ties retain distinct row membership. "
                  "This is not exact hypervolume regret.", ""]
    else:
        lines += ["Primary metric: best actually observed target-fidelity regret; "
                  "smaller is better. "
                  "reference minimum 0.397887, tolerance 1e-6. Low-fidelity observations do not "
                  "update this score. The first two initial designs are at target fidelity; "
                  "remaining initial fidelities come from the shared Sobol stream.", "",
                  "Random/Sobol baselines are target-only after initialization. Modeled cost is "
                  "0.25 + 0.75*s, one unit at target s=1; "
                  "not wall time or production cost: budgets. "
                  "Fixed-count endpoints are not equal-cost comparisons.", "",
                  "Oracle target-projected regret is a scoring-only diagnostic at sampled designs, "
                  "not recommendation quality or verified experimental outcomes. It is never "
                  "passed to the optimizer. "
                  "oracle_seconds records its separate computation time.", ""]
    lines += ["| Strategy | Complete / scheduled | Failed / timeout / interrupted "
              f"| Final {metric} median [Q25, Q75] | Wall seconds |",
              "|---|---|---|---|---|"]
    for _, row in summary.iterrows():
        values = [row[f"final_{metric}_{s}"] for s in ("median", "q25", "q75")]
        value = (f"{values[0]:.6g} [{values[1]:.6g}, {values[2]:.6g}]"
                 if all(pd.notna(v) for v in values) else "not available")
        wall = f"{row.wall_seconds:.3f}" if pd.notna(row.wall_seconds) else "not available"
        lines.append(f"| {row.strategy} | {row.complete}/{row.scheduled} "
                     f"| {row.failed}/{row.timeout}/{row.interrupted} | {value} | {wall} "
                     f"(known subtotal {row.known_wall_seconds:.3f}; "
                     f"{row.known_timing_trials} known / {row.unknown_timing_trials} unknown) |")
    lines += ["", "## Failures and evidence warnings", ""]
    for row in statuses.itertuples():
        if (row.status != "complete" or row.evidence_warning or row.timing_warning
                or row.trace_warning):
            lines.append(f"- `{row.trial_id}`: {row.status}, "
                         f"{row.completed_evaluations} evaluations. "
                         f"{row.message} {row.evidence_warning} {row.timing_warning}"
                         f" {row.trace_warning or ''}")
    lines += ["", "Per-seed traces and partial results: traces.csv. Complete-trial quantiles and "
              "contributing counts: trajectories.csv. Cost diagnostics and runtime components: "
              "summary.csv. All terminal records: trials.csv. Settings, seeds, and environment: "
              "run.json. Reports never refit models or rerun objectives. These are controlled "
              "workflow tests, not general superiority claims.", ""]
    return "\n".join(lines)


def multi_figures(traces, trajectories, metadata, directory):
    import matplotlib.pyplot as plt

    from benchmarks.figures import _curves, _final_distribution, _seed_lines
    from bo_forge.plot_style import (
        add_legend,
        finalise_axes,
        new_subplots,
        set_axis_labels,
        set_title,
    )

    route = metadata["spec"]["route"]
    metric = _metrics(route)[0]
    label = "Dominated hypervolume" if route == "multi_objective" else "Observed target regret"
    data = traces.rename(columns={metric: "simple_regret"})
    curve = trajectories.loc[trajectories.metric.eq(metric)]
    fig, axes = new_subplots(1, 3, figsize=(21, 6))
    try:
        _seed_lines(axes[0], data, metadata["spec"]["seeds"])
        _curves(axes[1], curve)
        _final_distribution(axes[2], data, metadata)
        for ax, title in zip(axes, ("Per-seed (hollow: partial)", "Complete trials: median / IQR",
                                    "Complete trials: final distribution"), strict=True):
            set_title(ax, title)
        if route == "multi_fidelity":
            set_title(axes[0], "Per-seed (hollow: partial)\nRandom/Sobol: target-only after init")
        for ax in axes[:2]:
            set_axis_labels(ax, "Evaluation count", label)
            add_legend(ax)
        set_axis_labels(axes[2], "Strategy", label)
        finalise_axes(fig, axes, save_path=directory / f"{route}-quality.png")
    finally:
        plt.close(fig)
    if route == "multi_fidelity":
        _fidelity_figures(traces, metadata, directory)


def _fidelity_figures(traces, metadata, directory):
    import matplotlib.pyplot as plt

    from benchmarks.figures import COLORS
    from bo_forge.plot_style import (
        add_legend,
        finalise_axes,
        new_subplots,
        set_axis_labels,
        set_title,
    )

    styles = {seed: "-" if index == 0 else (0, (index + 1, 1, 1, 1))
              for index, seed in enumerate(metadata["spec"]["seeds"])}
    fig, axes = new_subplots(1, 3, figsize=(21, 6))
    try:
        panels = [("cumulative_modeled_cost", "target_regret", "Observed target regret vs cost"),
                  ("evaluation", "oracle_target_regret", "Oracle diagnostic vs evaluations"),
                  ("cumulative_modeled_cost", "oracle_target_regret", "Oracle diagnostic vs cost")]
        for ax, (x, y, title) in zip(axes, panels, strict=True):
            for (_, strategy), rows in traces.groupby(["trial_id", "strategy"], sort=False):
                complete = rows.status.iloc[0] == "complete"
                face = COLORS[strategy] if complete else "white"
                ax.plot(rows[x], rows[y], color=COLORS[strategy], marker="o",
                        markeredgecolor="black", markerfacecolor=face,
                        linestyle=styles[rows.seed.iloc[0]],
                        label=f"{strategy}, seed {rows.seed.iloc[0]}"
                              f"{' (partial)' if not complete else ''}")
            set_title(ax, title + "\nRandom/Sobol: target-only after init")
            ylabel = "Observed target regret" if y == "target_regret" else "Oracle projected regret"
            set_axis_labels(ax, "Evaluation count" if x == "evaluation" else "Modeled cost (units)",
                            ylabel)
            add_legend(ax)
            if traces.empty:
                ax.text(.5, .5, "No completed evaluations", transform=ax.transAxes, ha="center")
        finalise_axes(fig, axes, save_path=directory / "multi-fidelity-cost-oracle.png")
    finally:
        plt.close(fig)
