"""Source-only, paired performance evidence; not a BO Forge public API."""

HARNESS_VERSION = 1
REPETITIONS = 5
PROCESS_TIMEOUT = 600
OVERALL_TIMEOUT = 5400


def cases():
    return [
        {"case_id": name, "kind": name}
        for name in ("import", "version", "help", "validate")
    ] + [
        {"case_id": f"{route}-q{q}", "kind": "suggest", "route": route, "batch_size": q}
        for route in ("log_ei", "qlog_ehvi", "qmf_kg") for q in (1, 2, 4)
    ]


def schedule():
    rows = []
    for case in cases():
        for repetition in range(REPETITIONS + 1):
            versions = ("baseline", "candidate") if repetition % 2 == 0 else (
                "candidate", "baseline")
            for version in versions:
                rows.append({**case, "version": version, "repetition": repetition,
                             "warmup": repetition == 0,
                             "sample_id": f"{case['case_id']}-{repetition}-{version}"})
    return rows
