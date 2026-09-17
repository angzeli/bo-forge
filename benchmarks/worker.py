"""Isolated CPU process entry point. Not an installed command."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from benchmarks.storage import read_trace, utc_now, write_json


def main():
    import torch

    from benchmarks.trial import run_trial

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    directory = Path(sys.argv[1])
    trial = json.loads((directory / "trial.json").read_text())
    torch.manual_seed(trial["seeds"]["fitting"])
    try:
        run_trial(directory)
    except Exception as exc:
        # Process boundary: retain partial evidence for any failed trial; never retry.
        traceback.print_exc()
        rows, warning = read_trace(directory / "trace.jsonl")
        write_json(directory / "status.json", {
            "status": "failed", "completed_evaluations": len(rows),
            "message": f"{type(exc).__name__}: {exc}", "trace_warning": warning,
            "finished_at": utc_now(),
        })
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
