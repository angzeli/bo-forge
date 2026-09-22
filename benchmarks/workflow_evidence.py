"""Verify typed and pending-workflow evidence without calling simulator objectives."""

import pandas as pd

from benchmarks.binding import require_equal
from benchmarks.definitions import definition
from benchmarks.designs import DRAW_LIMIT, normalize
from benchmarks.evidence import read_object
from benchmarks.storage import read_trace


def expected_events(ids, trial):
    pending_route = trial["route"] == "pending_noisy"
    events = []
    for start in range(len(ids)):
        initial = start < trial["initial_observations"]
        if pending_route and not initial and (start - trial["initial_observations"]) % 2:
            continue
        count = 2 if pending_route and not initial else 1
        pending = []
        for index in range(start, min(start + count, len(ids))):
            events.append({"operation": "submit", "row_id": ids[index],
                           "submission": index + 1, "pending_row_ids": pending.copy()})
            pending.append(ids[index])
            if pending_route:
                events.append({"operation": "accept", "row_id": ids[index],
                               "submission": index + 1, "pending_row_ids": pending.copy()})
        if len(pending) < count:
            break
        for index in range(start, start + count):
            pending.remove(ids[index])
            events.append({"operation": "observe", "row_id": ids[index],
                           "submission": index + 1, "pending_row_ids": pending.copy()})
    return events


def verify_workflow(directory, trial, rows, complete):
    if trial.get("schema_version", 1) < 2:
        return {}
    values = {"proposal_draws": None, "infeasible_proposals": None, "duplicate_proposals": None,
              "workflow_event_count": 0, "suggestions_with_pending": 0}
    counters = _read_counters(directory, trial, complete)
    if counters is not None:
        values.update(proposal_draws=counters["draws"], infeasible_proposals=counters["infeasible"],
                      duplicate_proposals=counters["duplicate"])
    events, warning = read_trace(directory / "events.jsonl")
    _verify_event_order(events, trial)
    path = directory / "campaign.csv"
    if not path.exists():
        values.update(workflow_event_count=len(events), suggestions_with_pending=sum(
            bool(e["pending_row_ids"]) for e in events if e["operation"] == "submit"),
            workflow_warning="; ".join(filter(None, [warning,
                "Workflow events cannot be reconciled without campaign CSV." if events else ""])))
        return values
    frame = pd.read_csv(path, keep_default_na=False)
    if values["proposal_draws"] is not None:
        _verify_accepted_draws(directory, trial, frame, counters, complete)
    names = [v["name"] for v in definition(trial)["variables"]]
    for point in frame[names].values.tolist():
        normalize(point, trial)
    if trial.get("route") == "multi_fidelity" and trial["strategy"] != "bo":
        for value in frame[names[-1]].iloc[trial["initial_observations"]:]:
            require_equal(trial, "target-only baseline fidelity", trial["fidelity"]["target"],
                          float(value))
    expected = expected_events(frame.row_id.astype(str).tolist(), trial)
    require_equal(trial, "workflow events", expected[:len(events)], events)
    if complete and (warning or len(events) != len(expected)):
        raise ValueError(f"{trial['trial_id']}: Incomplete workflow event evidence.")
    persistence_warning = _verify_persisted_events(frame, events, trial)
    submitted = {e["row_id"]: e for e in events if e["operation"] == "submit"}
    observed = {e["row_id"] for e in events if e["operation"] == "observe"}
    ordered = [e["row_id"] for e in events if e["operation"] == "observe"]
    require_equal(trial, "trace observation order", ordered[:len(rows)],
                  [row["row_id"] for row in rows])
    for row in rows:
        if row["row_id"] not in observed or row["row_id"] not in submitted:
            raise ValueError(f"{trial['trial_id']}: Scored row lacks workflow evidence.")
        require_equal(trial, "pending_row_ids", submitted[row["row_id"]]["pending_row_ids"],
                      row.get("pending_row_ids"))
        require_equal(trial, "submission index", submitted[row["row_id"]]["submission"],
                      row["evaluation"])
        require_equal(trial, "feasible", True, row.get("feasible"))
    values["workflow_event_count"] = len(events)
    values["suggestions_with_pending"] = sum(bool(e["pending_row_ids"])
                                             for e in submitted.values())
    values["workflow_warning"] = "; ".join(filter(None, [warning, persistence_warning]))
    return values


def _verify_event_order(events, trial):
    ids = [e.get("row_id") for e in events if e.get("operation") == "submit"]
    if any(not isinstance(row_id, str) or not row_id for row_id in ids):
        raise ValueError(f"{trial['trial_id']}: workflow events require nonempty row IDs.")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{trial['trial_id']}: workflow events repeat submitted row IDs.")
    if len(ids) > trial["evaluations"]:
        raise ValueError(f"{trial['trial_id']}: workflow events exceed the evaluation budget.")
    expected = expected_events(ids, trial)
    require_equal(trial, "workflow events", expected[:len(events)], events)


def _verify_persisted_events(frame, events, trial):
    persisted = set()
    reviewed = trial["route"] == "pending_noisy"
    by_id = frame.assign(row_id=frame.row_id.astype(str)).set_index("row_id")
    for row_id, row in by_id.iterrows():
        if row.status not in {"suggested", "observed"}:
            require_equal(trial, f"CSV row {row_id}.status", "suggested or observed", row.status)
        persisted.add((row_id, "submit"))
        if reviewed:
            if row.review_status not in {"pending", "accepted"} or row.status == "observed":
                require_equal(trial, f"CSV row {row_id}.review_status", "accepted",
                              row.review_status)
            if row.review_status == "accepted":
                persisted.add((row_id, "accept"))
        if row.status == "observed":
            persisted.add((row_id, "observe"))
    recorded = {(event["row_id"], event["operation"]) for event in events}
    for row_id, operation in sorted(recorded - persisted):
        field, expected = (("review_status", "accepted") if operation == "accept"
                           else ("status", "observed"))
        require_equal(trial, f"workflow row {row_id}.{field}", expected, by_id.loc[row_id, field])
    missing = persisted - recorded
    return (f"{len(missing)} persisted mutation(s) lack workflow events; "
            "interrupted event recording." if missing else "")


def _read_counters(directory, trial, complete):
    path = directory / "feasibility.json"
    if not path.exists():
        if complete:
            raise ValueError(f"{trial['trial_id']}: Completed trial lacks feasibility accounting.")
        return None
    counters = read_object(path)
    require_equal(trial, "draw_limit", DRAW_LIMIT, counters.get("draw_limit"))
    for name in ("draws", "infeasible", "duplicate"):
        value = counters.get(name)
        if type(value) is not int or not 0 <= value <= DRAW_LIMIT:
            raise ValueError(f"{trial['trial_id']}: Invalid proposal accounting: {name}")
    if counters["infeasible"] + counters["duplicate"] > counters["draws"]:
        raise ValueError(f"{trial['trial_id']}: Rejection count exceeds proposal draws.")
    return counters


def _verify_accepted_draws(directory, trial, frame, counters, complete):
    accepted = counters["draws"] - counters["infeasible"] - counters["duplicate"]
    initial = trial["initial_observations"]
    if not (directory / "inputs.json").exists():
        minimum, maximum = 0, initial
    else:
        minimum = initial + (max(0, len(frame) - initial) if trial["strategy"] != "bo" else 0)
        maximum = minimum + int(not complete and trial["strategy"] != "bo")
    if not minimum <= accepted <= maximum:
        raise ValueError(f"{trial['trial_id']}: accepted proposal draws: "
                         f"expected {minimum}..{maximum}; actual {accepted}")
