# BO Forge Performance Benchmarks

Performance evidence is host- and protocol-specific, not a machine-independent
CI threshold or a scientific-quality comparison.

## v3.3.4 Same-Host Protocol

**Performance acceptance: passed (156/156 processes; 13/13 valid cases).**
v3.3.x implementation is complete. The local release gate must pass before commit;
exact-commit CI remains pending. The measured result below is archive-bound local
evidence, not exact-commit CI or publication approval.

Compare an sdist built from baseline commit `6c0d57db` (v3.3.3) with the v3.3.4
candidate sdist on the same host. Build the baseline in an isolated detached Git
worktree; never switch or overwrite the candidate checkout to build the baseline.
Retain both exact archives and their identities with the execution evidence.

| Protocol item | Required boundary |
| --- | --- |
| Workload | 13 import/startup and representative q=1/2/4 cases |
| Schedule | 13 cases x 2 versions x 6 samples = 156 fresh processes |
| Per-process deadline | 600 seconds |
| Overall harness budget | 90 minutes |
| Environments | Isolated, non-editable baseline and candidate installations |
| Preparation | Environment creation and installation excluded from sample timings |
| Dependencies | Existing platform-specific hashed development constraints; no upgrades |
| Ratios | Advisory candidate/baseline timing ratios, not pass/fail speed thresholds |

Branin LogEI and Branin-Currin qLogEHVI reuse their standard specifications'
seed-0 trial, derived fitting seed, six Sobol observations, and optimizer settings
(`raw_samples=128`, `num_restarts=5`, `mc_samples=128`). Discrete qMFKG copies
example 22's config and six observed rows unchanged, including its seed 22,
four fidelity levels, and eight fantasies. Each route requests q=1, 2, and 4
through the existing batch-size argument. Batched LogEI retains the existing
`qlog_ei` source label. Every call is model-based and dry-run; nothing is appended.

Six samples are scheduled per case/version: one retained warm-up and five
measured repeats. The warm-up is excluded from the measured summaries.
Report successful sample counts, spread, failures, timeouts, interruptions, and
work not started. Missing timing is unknown, not zero, and incomplete schedules
must not be presented as complete performance acceptance. Record available memory
measurements with their collection method and platform limits; do not silently
compare incompatible memory metrics. Timing changes do not imply optimizer-quality
changes, and ratios alone do not authorize a dependency or support-status change.

### Commands And Manual CI

Run from the candidate checkout or extracted candidate sdist in the constrained
development environment. The harness prepares the two isolated measurement
environments; editable installs are not measurement evidence.

```bash
python -m benchmarks.performance run --baseline PATH --candidate PATH --output PATH
python -m benchmarks.performance report --input PATH --output NEWPATH
```

For `run`, the two input paths are sdists and the output is a fresh run directory.
For `report`, the input is that retained run directory and the output is a different,
new report directory. Report regeneration must not rerun the workloads. Preserve
failed attempts and retry into new destinations; never overwrite previous evidence.

The manual workflow [performance.yml](../.github/workflows/performance.yml) uses
`workflow_dispatch` only, one `ubuntu-24.04` job, Python 3.12, and the existing
`uv==0.11.3` / hashed Linux development constraints. The job is limited to
120 minutes; the harness retains its 90-minute overall budget and 600-second
per-process deadline. Baseline and candidate builds and measurements stay in that
one job on the same host. Available input archives, run evidence, and regenerated
reports are uploaded with `always()` and retained for 14 days, including failed
attempts. Full performance runs are not PR CI and are not added to the tag gate.

### Measured Acceptance

The paired local run on **2026-09-24** completed **156/156 processes**, with
**13/13 valid case pairs** and no failed, timed-out, interrupted, or not-run samples.
This comprises 26 retained warm-ups and 130 measured samples. `run.json` records
`status: complete`; the regenerated `summary.json` records `valid: true`, and
every case has 12 complete samples out of 12 scheduled across the two versions.
The run spanned 11:01:29-11:14:38 UTC (788.9518 seconds); preparation accounted
for 133.5912 seconds and is excluded from every sample timing.

#### Archive And Environment Identity

| Input | Version / identity |
| --- | --- |
| Baseline | v3.3.3, baseline reference `6c0d57db` |
| Baseline sdist SHA-256 | `466a48c15347e5d8c4d2635afddbeb85acf53c693bd26dc856e47aa6b0fc9f40` |
| Candidate | v3.3.4, evaluated before report-validation and interruption-checkpoint fixes plus closeout docs |
| Candidate sdist SHA-256 | `e24137cc7ea94b287c613658dcc741c79489462059164a7b1c229e85e324e971` |
| Retained constraints SHA-256 | `4b92bbae1866b238228c2a21c009d1e3f85fffb61b2d1db4bde897eadadf2a5a` |
| Regenerated report's `reporter_sha256` | `3de4067a70fb32a85ca2d92e1ae30e41a68414e74df16615f76f622d23065c56` |

The measured host was `macOS-26.3-arm64-arm-64bit`, CPU label `arm`, with
14 physical and 14 logical CPUs reported, and Python **3.12.0**. This was not
the future Ubuntu manual-CI job. Measurements were CPU-only, with OMP, MKL,
OpenBLAS, and NumExpr thread limits set to one and Matplotlib set to `Agg`.

Both isolated non-editable wheel installations recorded the same dependency
versions: BoTorch 0.17.2, PyTorch 2.11.0, GPyTorch 1.15.2, NumPy 2.5.2,
SciPy 1.18.1, pandas 3.0.5, Matplotlib 3.11.1, filelock 3.32.4, and PyYAML 6.0.3.
The complete package manifests are retained in each version's `identity.json`;
only `bo-forge` differs (3.3.3 versus 3.3.4). Import paths point into each
version's own `venv/lib/python3.12/site-packages/bo_forge/`, and the preflight
and postflight probe records agree. Production implementation is unchanged
apart from the 3.3.4 version declaration; these measurements do not evaluate an
optimizer modification. The candidate archive predates report-validation and
interruption-checkpoint fixes plus closeout docs. The measurement worker,
workloads, and production code are unchanged. Original samples remain preserved;
the regenerated report validates the same retained evidence and records the actual
reporter hash above, separately from the original run's harness identities.
A later rebuilt archive or final commit has a different identity and must not
inherit an exact-commit CI claim from this run.

#### Process Timing

Values are **median (IQR) in seconds** over five measured fresh processes per
version/case; IQR uses inclusive quartiles. Process time includes startup and
shutdown on a warmed filesystem, not cold-cache startup. Ratios are the
candidate median divided by the baseline median, computed before rounding.

| Case | Baseline median (IQR), s | Candidate median (IQR), s | Ratio |
| --- | ---: | ---: | ---: |
| import | 0.0962 (0.0017) | 0.0961 (0.0007) | 0.9990 |
| version | 0.4141 (0.0168) | 0.4103 (0.0502) | 0.9908 |
| help | 0.4196 (0.0115) | 0.4197 (0.0366) | 1.0001 |
| validate | 0.4577 (0.1318) | 0.4954 (0.0870) | 1.0822 |
| log_ei-q1 | 2.9720 (0.0756) | 2.9918 (0.5264) | 1.0067 |
| log_ei-q2 | 2.9333 (0.1945) | 2.9645 (0.0471) | 1.0106 |
| log_ei-q4 | 3.1568 (0.1734) | 3.2296 (0.1231) | 1.0230 |
| qlog_ehvi-q1 | 3.0703 (0.1278) | 2.9770 (0.0993) | 0.9696 |
| qlog_ehvi-q2 | 3.1819 (0.1026) | 3.0737 (0.0768) | 0.9660 |
| qlog_ehvi-q4 | 6.1329 (0.7155) | 6.0834 (0.1251) | 0.9919 |
| qmf_kg-q1 | 4.6471 (0.1504) | 4.6942 (0.0803) | 1.0101 |
| qmf_kg-q2 | 7.0468 (0.0692) | 6.9940 (0.4309) | 0.9925 |
| qmf_kg-q4 | 11.3041 (0.6675) | 11.2934 (0.2715) | 0.9991 |

#### Public Suggestion Call And Memory

Call time covers only the public suggestion call, not process startup/shutdown.
It is not interchangeable with process time. RSS is the worker's lifetime peak
resident memory, not incremental model memory; the report retains bytes and the
table converts them to MiB (`1 MiB = 1,048,576 bytes`). Entries remain median
(IQR). The four import/CLI cases have no call-time or RSS measurements; those
metrics are unavailable, not zero.

| Case | Baseline call, s | Candidate call, s | Call ratio | Baseline RSS, MiB | Candidate RSS, MiB | RSS ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| log_ei-q1 | 0.3342 (0.0709) | 0.3377 (0.1336) | 1.0105 | 357.7188 (0.2656) | 358.3906 (0.6406) | 1.0019 |
| log_ei-q2 | 0.5281 (0.0073) | 0.4759 (0.0097) | 0.9012 | 358.7969 (1.5469) | 357.3906 (1.7812) | 0.9961 |
| log_ei-q4 | 0.6421 (0.1798) | 0.6931 (0.0627) | 1.0794 | 359.0781 (1.1875) | 360.1250 (0.7969) | 1.0029 |
| qlog_ehvi-q1 | 0.5853 (0.0201) | 0.5457 (0.0829) | 0.9323 | 361.5156 (0.6562) | 360.7344 (1.0938) | 0.9978 |
| qlog_ehvi-q2 | 0.7196 (0.0716) | 0.7015 (0.0323) | 0.9749 | 366.5781 (2.9844) | 364.3438 (5.8125) | 0.9939 |
| qlog_ehvi-q4 | 3.4248 (0.3758) | 3.4454 (0.1169) | 1.0060 | 414.2500 (1.5781) | 417.7969 (0.9531) | 1.0086 |
| qmf_kg-q1 | 2.2089 (0.1423) | 2.2019 (0.1442) | 0.9968 | 378.7344 (2.4531) | 378.6562 (3.9375) | 0.9998 |
| qmf_kg-q2 | 4.4809 (0.1047) | 4.5128 (0.0372) | 1.0071 | 405.8906 (1.6562) | 406.7344 (2.6250) | 1.0021 |
| qmf_kg-q4 | 8.9817 (0.3738) | 8.9950 (0.1842) | 1.0015 | 406.2812 (4.1250) | 404.4219 (3.0938) | 0.9954 |

Across all 13 cases, process-median ratios span **0.9660-1.0822** (approximately
-3.40% to +8.22%). Across the nine suggestion cases they span **0.9660-1.0230**;
public-call ratios span **0.9012-1.0794**, and peak-RSS ratios **0.9939-1.0086**.
These are advisory descriptive ranges, not statistical-significance tests or
speedup claims. Five repeats do not establish equivalence, superiority, or a
portable regression threshold; do not pool machines or infer scientific quality.

#### Retained Evidence And Readiness

Local retention locations are
[`reports/performance/v3.3.4/paired-acceptance/`](../reports/performance/v3.3.4/paired-acceptance/)
and [`reports/performance/v3.3.4/paired-report/`](../reports/performance/v3.3.4/paired-report/).
These are ignored local evidence, not committed source or artifacts supplied by
a normal clone. The report retains `summary.json`, `report.md`, and `samples.csv`;
the run retains the exact sdists, built wheels, identity/probe records, constraints,
all inputs, raw samples, logs, and harness sources. Virtual environments and
extracted source trees are excluded from future CI uploads; the portable evidence
remains included. The manual workflow will upload its own future run's available
evidence for 14 days; this local result is not a claim that CI has run.

Performance acceptance has passed and v3.3.x implementation is complete.
The local release gate must pass before commit. Exact-commit CI remains pending;
push, tagging, and publication remain separate authorization gates.

The separate verified notebook baseline completed 20/20 full-profile notebooks
from archive
`466a48c15347e5d8c4d2635afddbeb85acf53c693bd26dc856e47aa6b0fc9f40`.
Its retained evidence is under `reports/notebooks/v3.3.3/seed-baseline-fix/`;
see [Notebook Execution](NOTEBOOK_EXECUTION.md). This certifies neither performance
acceptance nor a rebuilt v3.3.4 archive. Exact-commit CI and publication approval
remain separate gates.

## Historical Timings: Not Comparable To v3.3.4

The following release-workstation measurements retain their original values and
protocol. They are not a v3.3.4 baseline: hardware, interpreter, dependencies, and
sampling differ. Each value is the median of five warm-cache subprocess runs after
one unrecorded warm-up run on the release workstation.

## v2.4.1 Startup And qMFKG Hardening

Recorded on 2026-08-05 with Python 3.11.14 and BoTorch 0.17.2.

| Command | v2.4.0 median (s) | v2.4.1 median (s) | Change |
| --- | ---: | ---: | ---: |
| `bo_forge --version` | 2.2599 | 0.3595 | -84.1% |
| CLI help | 2.4678 | 0.3676 | -85.1% |
| Discrete qMFKG validation | 2.2384 | 0.3736 | -83.3% |
| Discrete qMFKG `q=1` suggestion | 4.3978 | 5.0533 | +14.9% |
| Discrete qMFKG `q=2` suggestion | 6.9443 | 6.6397 | -4.4% |
| Discrete qMFKG `q=4` suggestion | 14.6199 | 12.3885 | -15.3% |

The startup improvement comes from lazy public and CLI imports. qMFKG runtime
settings remain unchanged when the new controls are omitted, so suggestion
times should be interpreted as normal stochastic and machine-level variation.
