"""Regenerate diagnostics from stored evidence, without fitting or objective calls."""

from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import mkdtemp

import pandas as pd

from benchmarks.binding import verify_binding
from benchmarks.evidence import (
    campaign_evidence,
    read_object,
    validate_inputs,
    validate_rows,
    validate_status,
)
from benchmarks.spec import _parse_spec, schedule, validate_spec
from benchmarks.storage import read_trace, sha256
from benchmarks.workflow_evidence import verify_workflow
from bo_forge._filesystem import rename_directory_exclusive

TRACE_COLUMNS = ["trial_id", "problem", "mode", "strategy", "seed", "status", "evaluation",
                 "row_id", "x", "source", "observed", "latent", "noise", "best_observed",
                 "best_latent", "simple_regret_raw", "simple_regret", "incumbent_latent",
                 "incumbent_regret_raw", "incumbent_regret", "suggestion_seconds",
                 "objective_seconds", "mutation_seconds", "fit_evidence", "route",
                 "pending_row_ids", "feasible"]


def _owned(root, path):
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Benchmark evidence path escapes run directory: {path}")
    return path


def load_evidence(run):
    run = Path(run)
    metadata = read_object(run / "run.json")
    if (type(metadata.get("schema_version")) is not int
            or metadata["schema_version"] not in (1, 2, 3)):
        raise ValueError("Unsupported benchmark evidence schema.")
    spec = validate_spec(metadata["spec"])
    if metadata["schema_version"] != spec["schema_version"]:
        raise ValueError("Evidence schema differs from its scheduled specification.")
    if metadata["trials"] != schedule(spec):
        raise ValueError("Scheduled trial identity differs from the resolved specification.")
    spec_bytes = _owned(run, run / "spec.yaml").read_bytes()
    if sha256(spec_bytes) != metadata["spec_sha256"]:
        raise ValueError("Specification snapshot hash mismatch.")
    if _parse_spec(spec_bytes) != spec:
        raise ValueError("Retained specification snapshot differs from the resolved specification.")
    traces, statuses = [], []
    for trial in metadata["trials"]:
        directory = _owned(run, run / "trials" / trial["trial_id"])
        for name in ("events.jsonl", "feasibility.json"):
            _owned(directory, directory / name)
        if read_object(_owned(run, directory / "trial.json")) != trial:
            raise ValueError(f"Trial identity mismatch: {trial['trial_id']}")
        status = read_object(_owned(run, directory / "status.json"))
        validate_status(status)
        rows, warning = read_trace(_owned(run, directory / "trace.jsonl"))
        validate_rows(rows, trial)
        identity = {key: trial[key] for key in ("trial_id", "mode", "strategy", "seed")}
        identity.update(problem=trial["problem"]["name"], status=status["status"],
                        route=trial.get("route", "continuous"))
        evidence_warning = _verify_trace(directory, trial, status, rows, warning)
        workflow = verify_workflow(directory, trial, rows, status["status"] == "complete")
        evidence_warning = "; ".join(filter(None, [evidence_warning,
                                                workflow.get("workflow_warning")]))
        traces.extend({**row, **identity} for row in rows)
        statuses.append({**identity, "completed_evaluations": len(rows),
                         "message": status.get("message", ""), "trace_warning": warning,
                         "wall_seconds": status.get("wall_seconds"),
                         "timing_warning": "" if status.get("wall_seconds") is not None else
                         "Historical trial duration not available.",
                         "evidence_warning": evidence_warning,
                         **workflow})
    columns = TRACE_COLUMNS
    if spec["schema_version"] == 3:
        from benchmarks.multi_evidence import trace_columns

        columns = trace_columns(spec["route"])
    return metadata, pd.DataFrame(traces, columns=columns), pd.DataFrame(statuses)


def _verify_trace(directory, trial, status, rows, warning):
    if status["completed_evaluations"] != len(rows) or len(rows) > trial["evaluations"]:
        raise ValueError("Trace length differs from terminal evaluation count.")
    if status["status"] == "complete" and (len(rows) != trial["evaluations"] or warning):
        raise ValueError("Completed trial has incomplete evidence.")
    config = _owned(directory, directory / "campaign.yaml")
    log = _owned(directory, directory / "campaign.csv")
    _owned(directory, directory / "campaign.csv.manifest.json")
    input_path = _owned(directory, directory / "inputs.json")
    inputs = read_object(input_path) if input_path.exists() else None
    verify_binding(directory, trial, inputs)
    if inputs is not None:
        validate_inputs(inputs, trial)
        if config.exists() and sha256(config.read_bytes()) != inputs["config_sha256"]:
            raise ValueError(f"{trial['trial_id']}: Campaign configuration changed "
                             "after execution.")
    observed, evidence_warning = campaign_evidence(
        directory, config, log, rows, status["status"] == "complete", trial,
    )
    if inputs is None:
        evidence_warning = "; ".join(filter(None, [evidence_warning,
                                                "inputs.json absent; initialization incomplete."]))
    if not rows:
        return evidence_warning
    from benchmarks.scoring import verify_rows

    if inputs is None:
        raise ValueError(f"{trial['trial_id']}: Scored trace requires inputs.json.")
    if trial.get("schema_version") == 3:
        from benchmarks.multi_evidence import verify_multi_rows

        verify_multi_rows(rows, observed, trial)
    else:
        verify_rows(rows, observed, inputs, trial)
    return evidence_warning


def tables(metadata, traces, statuses):
    if metadata["schema_version"] == 3:
        from benchmarks.multi_report import multi_tables

        return multi_tables(metadata, traces, statuses)
    summary, trajectories = [], []
    group_keys = ["route", "problem", "mode", "strategy"]
    for key, group in statuses.groupby(group_keys, sort=False):
        ids = group.loc[group.status.eq("complete"), "trial_id"]
        relevant = traces.loc[traces.trial_id.isin(group.trial_id)]
        successful = relevant.loc[relevant.trial_id.isin(ids)]
        identity = dict(zip(group_keys, key, strict=True))
        last = successful.loc[successful.evaluation.eq(metadata["spec"]["evaluations"])]
        summary.append({
            **identity, "scheduled": len(group), "complete": len(ids),
            "failed": int(group.status.eq("failed").sum()),
            "timeout": int(group.status.eq("timeout").sum()),
            "interrupted": int(group.status.eq("interrupted").sum()),
            "completed_evaluations": int(group.completed_evaluations.sum()),
            "final_regret_median": last.simple_regret.median() if len(last) else None,
            "final_regret_q25": last.simple_regret.quantile(.25) if len(last) else None,
            "final_regret_q75": last.simple_regret.quantile(.75) if len(last) else None,
            "wall_seconds": float(group.wall_seconds.sum()) if group.wall_seconds.notna().all()
            else None,
            "known_wall_seconds": float(group.wall_seconds.sum()),
            "known_timing_trials": int(group.wall_seconds.notna().sum()),
            "unknown_timing_trials": int(group.wall_seconds.isna().sum()),
            **{name: group[name].sum(min_count=1) for name in
               ("proposal_draws", "infeasible_proposals", "duplicate_proposals",
                "workflow_event_count", "suggestions_with_pending")
               if name in group},
            **{name: float(pd.to_numeric(relevant[name]).sum()) for name in
               ("suggestion_seconds", "objective_seconds", "mutation_seconds")},
        })
        for evaluation in range(1, metadata["spec"]["evaluations"] + 1):
            at = successful.loc[successful.evaluation.eq(evaluation)]
            partial = relevant.loc[relevant.evaluation.eq(evaluation)
                                   & ~relevant.trial_id.isin(ids)]
            trajectories.append({
                **identity, "evaluation": evaluation, "scheduled": len(group),
                "contributing_complete": len(at), "contributing_partial": len(partial),
                "median": at.simple_regret.median() if len(at) else None,
                "q25": at.simple_regret.quantile(.25) if len(at) else None,
                "q75": at.simple_regret.quantile(.75) if len(at) else None,
            })
    return pd.DataFrame(summary), pd.DataFrame(trajectories)


def _markdown(metadata, summary, statuses):
    if metadata["schema_version"] == 3:
        from benchmarks.multi_report import multi_markdown

        return multi_markdown(metadata, summary, statuses)
    lines = [f"# {metadata['spec']['name']} closed-loop benchmark", "",
             "Summaries are conditioned on successfully completed trials. Counts retain all "
             "scheduled trials. Partial traces are shown only through their last evaluation; "
             "nothing is extrapolated after failure.", "",
             "Noise-free simple regret uses minimization and each problem's recorded optimum "
             "tolerance. Latent outcomes occur only in scoring artifacts, never campaign inputs.",
             "Observed-best and observation-selected incumbent regret are in traces.csv. "
             f"Scheduled seed count: {len(metadata['spec']['seeds'])}. Results are descriptive "
             "evidence, not a superiority or calibration claim.", "",
             "Runtime components cover completed evaluations; wall time also includes worker "
             "startup and interrupted/failed work. No harness retries or baseline fallback.", "",
             "Pending routes validate workflow, not asynchronous throughput. Mixed fixtures "
             "are controlled tests, not representative scientific applications.", "",
             "| Route | Problem | Mode | Strategy | Complete / scheduled "
             "| Failed / timeout / interrupted | Final regret median [Q25, Q75] | Wall seconds |",
             "|---|---|---|---|---|---|---|---|"]
    for row in summary.itertuples():
        values = [row.final_regret_median, row.final_regret_q25, row.final_regret_q75]
        metric = (f"{values[0]:.6g} [{values[1]:.6g}, {values[2]:.6g}]"
                  if all(pd.notna(value) for value in values) else "not available")
        wall = f"{row.wall_seconds:.3f}" if pd.notna(row.wall_seconds) else "not available"
        lines.append(f"| {row.route} | {row.problem} | {row.mode} | {row.strategy} "
                     f"| {row.complete}/{row.scheduled} "
                     f"| {row.failed}/{row.timeout}/{row.interrupted} | {metric} "
                     f"| {wall} "
                     f"(known subtotal {row.known_wall_seconds:.3f}; "
                     f"{row.known_timing_trials} known / {row.unknown_timing_trials} unknown) |")
    lines.extend(["", "## Failures and timeouts", ""])
    failed = statuses.loc[statuses.status.ne("complete")]
    lines.extend(f"- `{row.trial_id}`: **{row.status}**, {row.completed_evaluations} evaluations. "
                 f"{row.message}" for row in failed.itertuples())
    if failed.empty:
        lines.append("All scheduled trials completed. This does not assert BO beat a baseline.")
    warnings = statuses.loc[statuses.evidence_warning.ne("")]
    if not warnings.empty:
        lines.extend(["", "## Evidence warnings", ""])
        lines.extend(f"- `{row.trial_id}`: {row.evidence_warning}"
                     for row in warnings.itertuples())
    unknown = statuses.loc[statuses.timing_warning.ne("")]
    if not unknown.empty:
        lines.extend(["", "## Timing warnings", ""])
        lines.extend(f"- `{row.trial_id}`: {row.timing_warning}" for row in unknown.itertuples())
    truncated = statuses.loc[statuses.trace_warning.fillna("").ne("")]
    if not truncated.empty:
        lines.extend(["", "## Trace warnings", ""])
        lines.extend(f"- `{row.trial_id}`: {row.trace_warning}" for row in truncated.itertuples())
    lines.extend(["", "Full seed mapping, resolved settings, environment, and Git identity: "
                  "run.json. Per-seed outcomes: traces.csv; denominators: trials.csv.", ""])
    return "\n".join(lines)


def generate_report(run, output):
    """Publish a new report directory; never overwrite evidence or existing reports."""
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Report destination already exists: {output}")
    if output.resolve().is_relative_to((Path(run) / "trials").resolve()):
        raise ValueError("Report destinations must be separate from trial campaign directories.")
    metadata, traces, statuses = load_evidence(run)
    for trial in metadata["trials"]:
        directory = (Path(run) / "trials" / trial["trial_id"]).resolve()
        if output.resolve().is_relative_to(directory):
            raise ValueError(
                "Report destinations must be separate from trial campaign directories.")
    summary, trajectories = tables(metadata, traces, statuses)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(mkdtemp(prefix=".benchmark-report-", dir=output.parent))
    try:
        for name, frame in (("traces", traces), ("trials", statuses), ("summary", summary),
                            ("trajectories", trajectories)):
            frame.to_csv(temporary / f"{name}.csv", index=False)
        (temporary / "report.md").write_text(
            _markdown(metadata, summary, statuses), encoding="utf-8",
        )
        from benchmarks.figures import render_figures

        render_figures(traces, trajectories, metadata, temporary / "figures")
        rename_directory_exclusive(temporary, output)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return output
