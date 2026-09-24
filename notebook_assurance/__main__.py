"""Repository-only entry point; no installed console script."""

import argparse
import json
import re
import sys
from pathlib import Path

from notebook_assurance.aggregate import aggregate
from notebook_assurance.runner import run


def concise(value):
    text = " ".join(str(value).split())
    return text if len(text) <= 300 else text[:297] + "..."


def error_message(value):
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(value))
    return concise(next((line for line in reversed(text.splitlines()) if line.strip()), ""))


def summary(result, report):
    records = result.get("notebooks", [])
    passed = sum(record.get("status") == "passed" for record in records)
    not_run = sum(record.get("status") == "not_run" for record in records)
    print(f"Notebook assurance: {result['status']}; passed={passed}, "
          f"failed={len(records) - passed - not_run}, not_run={not_run}, "
          f"errors={len(result.get('errors', []))}; evidence: {report}")
    for record in records:
        if record.get("status") == "passed":
            continue
        name = record.get("notebook", "unknown")
        cell = record.get("failing_cell_index")
        cell_id = record.get("failing_cell_id")
        directory = record.get("directory")
        evidence = record.get("evidence_path") or (
            report.parent / directory if directory else report)
        print(f"  {concise(name)}: {concise(record.get('status', 'unknown'))}; "
              f"cell_index={concise(cell)}, cell_id={concise(cell_id)}; evidence: {evidence}")
        if record.get("message"):
            print(f"    {concise(record.get('exception_type', 'Error'))}: "
                  f"{error_message(record['message'])}")
    for error in result.get("errors", []):
        print(f"  {concise(error['exception_type'])}: {error_message(error['message'])}; "
              f"evidence: {error['evidence_path']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "aggregate"):
        command = commands.add_parser(name)
        command.add_argument("--sdist", required=True)
        command.add_argument("--profile", choices=("pr", "full"), required=True)
        command.add_argument("--output", required=True)
        if name == "run":
            command.add_argument("--notebook", action="append")
        else:
            command.add_argument("--evidence", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).absolute()
    retry = "Retry with a fresh output directory (a new --output destination)."
    if output.exists() or output.is_symlink():
        print(f"Notebook assurance: output already exists: {output}. "
              f"Nothing overwritten. {retry}", file=sys.stderr)
        return 1
    report = output / ("result.json" if args.command == "run" else "aggregate.json")
    try:
        if args.command == "run":
            result = run(args.sdist, args.profile, args.output, args.notebook)
        else:
            result = aggregate(args.sdist, args.profile, args.evidence, args.output)
    except Exception as exc:
        # CLI boundary: report arbitrary execution/validation errors without a traceback dump.
        print(f"Notebook assurance: {type(exc).__name__}: {error_message(exc)}", file=sys.stderr)
        # Do not present a competing attempt's report when destination creation races.
        if not isinstance(exc, FileExistsError):
            try:
                retained = json.loads(report.read_text())
                summary({**retained, "status": "failed"}, report)
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                print(f"Evidence unavailable: {report}", file=sys.stderr)
        print(retry, file=sys.stderr)
        return 1
    summary(result, report)
    if result["status"] != "passed":
        print(retry, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
