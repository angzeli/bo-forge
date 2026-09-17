"""Repository CLI: python -m benchmarks run|report. Nothing is installed as a script."""

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Repository-only closed-loop BO benchmarks.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run sequential trials into a new directory.")
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("report", help="Regenerate a report from stored evidence only.")
    report.add_argument("--run", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        from benchmarks.report import generate_report

        if args.command == "report":
            print(generate_report(args.run, args.output))
            return 0
        from benchmarks.runner import run_suite

        output = run_suite(args.spec, args.output)
        print(generate_report(output, output / "report"))
        return int(json.loads((output / "run.json").read_text())["status"] != "complete")
    except (OSError, ValueError, KeyError) as exc:
        print(f"Benchmark error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Benchmark interrupted; partial evidence retained, no automatic resume.",
              file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
