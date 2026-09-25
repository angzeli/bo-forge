"""Repository-only performance run/report entry point."""

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--baseline", required=True)
    run.add_argument("--candidate", required=True)
    run.add_argument("--output", required=True)
    report = commands.add_parser("report")
    report.add_argument("--input", required=True)
    report.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            from benchmarks.performance.runner import run

            run(args.baseline, args.candidate, args.output)
            return 0
        from benchmarks.performance.report import report

        return 0 if report(args.input, args.output) else 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Performance evidence error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
