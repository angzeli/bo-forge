"""Problem-specific regret figures using existing scoped scientific styles."""

from pathlib import Path

import matplotlib.pyplot as plt

from bo_forge.plot_style import (
    MODEL_COLOR,
    NEUTRAL_COLOR,
    OBSERVED_COLOR,
    add_legend,
    finalise_axes,
    new_subplots,
    scoped_plot_style,
    set_axis_labels,
    set_title,
)

COLORS = {"bo": MODEL_COLOR, "random": NEUTRAL_COLOR, "sobol": OBSERVED_COLOR}


@scoped_plot_style
def render_figures(traces, trajectories, metadata, directory):
    directory = Path(directory)
    directory.mkdir()
    for problem in metadata["spec"]["problems"]:
        for mode in metadata["spec"]["modes"]:
            trace = traces.loc[traces.problem.eq(problem["name"]) & traces["mode"].eq(mode)]
            curve = trajectories.loc[trajectories.problem.eq(problem["name"])
                                     & trajectories["mode"].eq(mode)]
            _figure(trace, curve, metadata, problem["name"], mode, directory)


def _figure(trace, curve, metadata, problem, mode, directory):
    figure, axes = new_subplots(1, 3, figsize=(21, 6))
    try:
        _seed_lines(axes[0], trace, metadata["spec"]["seeds"])
        _curves(axes[1], curve)
        _final_distribution(axes[2], trace, metadata)
        route = metadata["spec"].get("route", "continuous")
        set_title(axes[0], f"{route}: {problem}, {mode}\nPer-seed regret (hollow: partial)")
        set_title(axes[1], "Complete trials only\nMedian and interquartile range")
        set_title(axes[2], "Complete trials only\nFinal regret distribution")
        for ax in axes[:2]:
            set_axis_labels(ax, "Evaluation count", "Noise-free simple regret")
            add_legend(ax)
        set_axis_labels(axes[2], "Strategy", "Final noise-free simple regret")
        finalise_axes(figure, axes, save_path=directory / f"{problem}-{mode}.png")
    finally:
        plt.close(figure)


def _seed_lines(ax, trace, seeds):
    styles = {seed: "-" if index == 0 else (0, (index + 1, 1, 1, 1))
              for index, seed in enumerate(seeds)}
    for (_, strategy), rows in trace.groupby(["trial_id", "strategy"], sort=False):
        complete = rows.status.iloc[0] == "complete"
        ax.plot(rows.evaluation, rows.simple_regret, color=COLORS[strategy],
                linestyle=styles[rows.seed.iloc[0]], marker="o", markeredgecolor="black",
                markerfacecolor=COLORS[strategy] if complete else "white",
                label=f"{strategy}, seed {rows.seed.iloc[0]}{' (partial)' if not complete else ''}")
    if trace.empty:
        ax.text(.5, .5, "No completed evaluations", transform=ax.transAxes, ha="center")


def _curves(ax, curve):
    for strategy, rows in curve.groupby("strategy", sort=False):
        rows = rows.loc[rows.contributing_complete.gt(0)]
        if not rows.empty:
            ax.plot(rows.evaluation, rows["median"], color=COLORS[strategy], marker="o",
                    label=f"{strategy} (n={rows.contributing_complete.iloc[0]}/"
                          f"{rows.scheduled.iloc[0]})")
            ax.fill_between(rows.evaluation, rows.q25, rows.q75, color=COLORS[strategy], alpha=.18)
    if not curve.contributing_complete.gt(0).any():
        ax.text(.5, .5, "No complete trials", transform=ax.transAxes, ha="center")


def _final_distribution(ax, trace, metadata):
    strategies = metadata["spec"]["strategies"]
    final = trace.loc[trace.status.eq("complete")
                      & trace.evaluation.eq(metadata["spec"]["evaluations"])]
    for index, strategy in enumerate(strategies):
        values = final.loc[final.strategy.eq(strategy), "simple_regret"].tolist()
        offsets = [0.0] if len(values) == 1 else [
            .3 * (i / (len(values) - 1) - .5) for i in range(len(values))
        ]
        ax.scatter([index + offset for offset in offsets], values, color=COLORS[strategy],
                   marker="o", edgecolor="black")
    ax.set_xticks(range(len(strategies)), strategies)
    if final.empty:
        ax.text(.5, .5, "No complete trials", transform=ax.transAxes, ha="center")
