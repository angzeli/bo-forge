# Notebook Execution Assurance

BO Forge v3.3.4 retains the source-only notebook assurance introduced in v3.3.3. The
`notebook_assurance` package ships in the source distribution (sdist), not the
runtime wheel. It executes unchanged notebook cells from a clean source archive
in isolated workspaces and keeps committed notebooks output-free.

## Acceptance Status

The verified v3.3.3 baseline passed all 20 notebooks in the `full` profile.
Its archive SHA-256 is
`466a48c15347e5d8c4d2635afddbeb85acf53c693bd26dc856e47aa6b0fc9f40`.
Retained evidence lives in `reports/notebooks/v3.3.3/seed-baseline-fix/`, with
the checked summary at
`bo-forge-v333-seedfix-verified-aggregate/aggregate.json` under that directory.
This acceptance belongs to those exact archive bytes; it does not certify a
rebuilt v3.3.4 candidate. The passed paired performance acceptance is recorded
separately in [Performance Benchmarks](PERFORMANCE_BENCHMARKS.md).

Execution acceptance requires successful full-profile results to be inspected. Prepared
code, static tests, package checks, and CI definitions do not grant execution
approval. A PR-subset pass is not a full-profile pass. Failed, interrupted, timed
out, or missing jobs remain failed or incomplete evidence; do not relabel them
as successful acceptance. Exact-commit CI, execution acceptance, and publication
authorization are separate gates. Historical benchmark results are unchanged.

An earlier full run stopped notebook 17 after 11 observations because all eight
candidate retries returned existing boundary designs. Investigation found that
its seed outcomes did not match the simulator used for later observations.
The tutorial now evaluates the same initial designs with its unchanged simulator
in the ignored working log, leaving the committed seed data and config unchanged.
Its 15-observation contract, random seed, optimizer settings, and duplicate
protection remain enforced. Never apply this synthetic preparation to real
experimental measurements. Preserve earlier failed evidence separately from
new acceptance runs. Notebook 20 also exposed an invalid review decision; its
calls now use the existing public `accept` action instead of `accepted`.

## Profiles And Resource Limits

- `pr`: 01, 04, 08, 12, 14, 18, 20, 22, 23, and 24 (10 notebooks).
- `full`: all 20 committed notebooks: 01-08, 10-12, 14-18, 20, 22-24.
- CI schedules one notebook per job, `fail-fast: false`, `max-parallel: 2`, and
  `timeout-minutes: 90`. A job timeout may prevent a final result being written;
  the aggregate must fail if that job's evidence is missing.

| Notebooks | Per-cell limit | Whole-notebook limit |
|---|---:|---:|
| Default | 600 seconds | 1,800 seconds |
| 11, 15, 22 | 3,300 seconds | 3,600 seconds |
| 24 | 3,720 seconds | 3,900 seconds |

Execution is CPU-only with one computational thread. Each notebook gets a new
extraction and kernel, a temporary kernelspec for the selected Python executable,
and isolated temporary/Jupyter/plotting directories. Packaged notebook, config,
seed-log, and benchmark-spec files are write-guarded and checked for changes.
This protects against accidental writes in trusted tutorials, not malicious code.
The kernel uses the normal inline Matplotlib backend so display-only figures
are rendered and retained in the executed notebook, not silently skipped by a
headless `show()`. Published computational cells and explicit export settings
are unchanged.

Use the existing platform-specific hashed constraints in `requirements/`, with
the pinned resolver and install commands in [the environment guide](../requirements/README.md).
They include the notebook kernel/client and optional interface dependencies.
Do not upgrade packages, add notebook dependency fallbacks, shorten notebook
workloads, or modify computational cells to manufacture a passing run.

## Local Commands

Run the source-only CLI from a checkout or extracted sdist in the constrained
development environment. Build the source archive once and retain those exact
bytes throughout execution and aggregation:

```bash
python -m build --sdist --no-isolation --outdir /tmp/bo-forge-notebook-dist
python -m notebook_assurance run \
  --sdist /tmp/bo-forge-notebook-dist/bo_forge-3.3.4.tar.gz \
  --profile pr --output /tmp/bo-forge-notebook-pr
```

Every `--output` must name a new directory. To select individual members of a
profile, repeat `--notebook FILENAME` (basename, not an arbitrary filesystem
path). The CI matrix uses exactly one such option per run:

```bash
python -m notebook_assurance run \
  --sdist /tmp/bo-forge-notebook-dist/bo_forge-3.3.4.tar.gz \
  --profile pr --notebook 01_maximisation_logei_campaign.ipynb \
  --output /tmp/bo-forge-notebook-evidence/01
```

Preserve each run directory under one evidence directory, without flattening or
overwriting its `result.json`. Once every scheduled job has finished, aggregate:

```bash
python -m notebook_assurance aggregate \
  --sdist /tmp/bo-forge-notebook-dist/bo_forge-3.3.4.tar.gz \
  --profile pr --evidence /tmp/bo-forge-notebook-evidence \
  --output /tmp/bo-forge-notebook-aggregate
```

The single-notebook example alone is deliberately insufficient for a passing
aggregate. Run every member of the profile before expecting success. Use
`--profile full` for full acceptance; rebuilding the sdist between jobs creates
different archive evidence and is not a valid substitute for the original file.
The CLI prints passed, failed, and not-run counts, failing cell identity when
available, and the evidence path. After a failed attempt, preserve its evidence
and choose a fresh output directory; an existing destination is never overwritten.

## Evidence Contract

Each run writes `OUTPUT/result.json` with `archive_sha256`, a `scheduled` list,
and a `notebooks` list of per-notebook statuses. Retain the whole output directory,
including available execution diagnostics, not just a console success message.
Statuses are `passed`, `failed`, `interrupted`, `timeout`, and `not_run`. Retain
the extracted sources, captures, per-notebook `status.json`, executed notebooks,
environment records, logs, and the intentional archive copy `source.tar.gz`.
Aggregation writes `OUTPUT/aggregate.json`, checks evidence against the
supplied archive and expected profile, and fails on missing jobs, missing
executed proof, or mismatched identities. It scans `**/result.json`, not
`status.json`. Available successes cannot stand in for the complete schedule.
After creating its output directory, aggregation retains a failed report even
when extraction, evidence parsing, or identity validation raises an error,
provided the filesystem permits writing it. A report-write failure does not
replace the original error. Preserve unsuccessful results for inspection rather
than deleting them or silently retrying them.

## CI And Exact-Tag Release

The reusable `.github/workflows/notebook-execution.yml` workflow downloads a
single previously built sdist, derives its notebook matrix, executes each member,
and aggregates even after a notebook job fails. Downloaded evidence remains in
separate per-job subdirectories. Only evidence download tolerates a missing
artifact so aggregation can diagnose missing jobs; execution and aggregation
themselves must succeed for the gate to pass. Evidence artifacts are retained
for 14 days.

PNG artifacts must decode fully. Matplotlib PDF artifacts must contain a complete
cross-reference table and consistent nonzero page count; these structural checks
do not replace visual review. Contextual tutorial completion also checks the
requested override and context sequence, not just the final row count.

- Required PR/push CI consumes the archive built by its package job.
- `.github/workflows/notebook-full.yml` is manually dispatched and builds one
  sdist before running the full profile.
- `.github/workflows/release-gate.yml` validates an exact version-matching tag,
  passes its resolved commit and already-built artifact to the full profile,
  and does not rebuild the archive for notebook jobs.

Private retained build artifacts alone do not mean the release gate passed.
Inspect the full workflow, aggregate, and archive identity before recording
acceptance in this guide and the [release checklist](RELEASE_CHECKLIST.md).
Nothing here authorizes a tag, public upload, registry publication, or release.
