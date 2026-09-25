# Closed-Loop Benchmarks

BO Forge v3.3.2 adds schema-v3 multi-objective and multi-fidelity evidence.
The v3.3.1 evidence-integrity and cancellation-timing fixes remain in place with
bounded `mixed`, `constrained_mixed`, and `pending_noisy` routes. The source-only
harness retains continuous Branin, Hartmann3, and Hartmann6 comparisons between
`bo`, `random`, and `sobol` in `deterministic` and `noisy` modes. It changes no BO
algorithm, public campaign API, or Streamlit workflow. Measured acceptance is
recorded below, including five constrained-BO failures; historical v3.3.0
results remain recorded separately.
Neither preparation nor those results establish general baseline superiority.

## Run From Source

### v3.3.2 Vector And Fidelity Protocols

These separate specs use benchmark `schema_version: 3`; versions 1 and 2 retain
their original layouts, trial identities, seed derivations, and reporting semantics.
No production campaign schema or capability changes. Runs remain sequential CPU
double-precision workers with one computational thread, a 600-second per-trial
timeout, and no harness retry or substitution.

| Route | Problem / BO strategy | Baselines after initialization | Smoke initial / total | Standard initial / total |
| --- | --- | --- | --- | --- |
| `multi_objective` | BoTorch Branin-Currin / qLogEHVI | Uniform random and continued scrambled Sobol | 4 / 6 | 6 / 24 |
| `multi_fidelity` | BoTorch Augmented Branin / qMFKG | Target-only random and continued scrambled Sobol | 4 / 6 | 8 / 16 |

Smoke uses seed 0: **6 trials / 36 evaluations** across the two routes. Standard
uses seeds 0 through 4: **30 trials / 600 evaluations**. Budgets include initialization.

```bash
python -m benchmarks run --spec benchmarks/specs/multi_objective_smoke.yaml --output /tmp/bo-forge-mo-smoke
python -m benchmarks run --spec benchmarks/specs/multi_fidelity_smoke.yaml --output /tmp/bo-forge-mf-smoke
# Explicit standard acceptance, not notebook Run All or routine CI:
python -m benchmarks run --spec benchmarks/specs/multi_objective_standard.yaml --output /tmp/bo-forge-mo-standard
python -m benchmarks run --spec benchmarks/specs/multi_fidelity_standard.yaml --output /tmp/bo-forge-mf-standard
```

**Coupled MO:** two continuous variables in [0, 1], ordered objectives `branin`,
`currin`, both minimized. The fixed user-space reference point is [18, 6]. Every
observation is coupled through `mark_observed(objective_values=...)`; deterministic
observed and latent vectors agree. Traces store both ordered vectors, reference
point, Pareto row membership (ties retained), Pareto count, and hypervolume after
each observation. Reference-excluded points contribute no volume. Hypervolume
is reported directly, never as an exact gap to BoTorch's approximate maximum.
Independent rectangle examples check sign conversion and the two-dimensional
hypervolume calculation.

**Continuous MF:** Augmented Branin uses native bounds [-5, 10] x [0, 15] x [0, 1],
with the last coordinate as fidelity `s` and target `s=1`. All strategies share
Sobol initialization: first two designs at target fidelity, remaining initial
fidelities from the seeded sequence. After initialization, qMFKG chooses all
three coordinates; baselines project their proposals to the target. Specs and
inputs record `first_two_target_then_sobol` and `target_only` policies.

Primary quality is best **actually observed target-fidelity regret**, using
reference minimum 0.397887 and tolerance 1e-6. Target matching uses numeric
tolerance (relative and absolute 1e-9). Lower-fidelity outcomes cannot update it.
Absent target observations yield blank metrics, not zeros. Substantial negative
regret is rejected; only rounding within the declared optimum tolerance is zeroed.

Modeled evaluation cost is `0.25 + 0.75*s` because fidelity already spans the
unit interval: a target evaluation costs one modeled unit. This is not wall
time and not production `cost:` budget support. Figures show observed target
quality against evaluation count and cumulative modeled cost. **Fixed-count
endpoints are not equal-cost comparisons.** Cost curves retain each seed's
actual evaluation points; they do not extrapolate or silently align unequal budgets.

The separately labeled **oracle diagnostic** evaluates the target projection of
each sampled design only for scoring. Oracle values are never written into
campaign observations or passed to the optimizer. They do not establish
recommendation quality or verified experimental outcomes. `oracle_seconds` is
separate from observation objective time, and both are distinct from wall time.
Actual observations are persisted before oracle scoring. If that diagnostic fails
or is interrupted, the trial remains failed/interrupted, its actual observation
stays in the campaign, and reports disclose the observation without a scoring trace.

MO reuses the original smoke/standard optimizer settings. MF smoke uses
`raw_samples=8`, `num_restarts=1`, `mc_samples=16`, `num_fantasies=4`,
`optimizer_maxiter=50`; standard uses 32, 2, 64, 8, 100 respectively. These are
benchmark-only settings; production defaults and numerical paths are untouched.

Reports use route-specific columns: no scalar-regret placeholders in MO and no
hypervolume placeholders in MF. Complete-trial quantiles carry contributing
counts; partial trajectories and all terminal records remain visible. Scheduled
binding checks objectives/order/directions, reference points, fidelity/cost settings,
initialization, baseline policy, sources, and seeds, including unscored partial
campaigns. Report regeneration validates stored evidence without model fits or
objective calls. Oracle truth itself is retained execution evidence, not recomputed
by report generation. Missing timing remains unknown rather than zero.
Missing artifacts are disclosed, but do not disable checks of retained CSV hashes,
scheduled initialization, or existing workflow events. Truncated scoring records
are disclosed in both the trial table and Markdown report. Report destinations
inside resolved trial directories are rejected, including symlink aliases.

The native functions come from the installed supported
[BoTorch test functions](https://botorch.readthedocs.io/en/v0.17.2/test_functions.html),
not reimplemented formulas. No noisy MO, discrete-fidelity, combined MO/MF route,
production optimizer repair, or additional dependency is included.

### v3.3.2 Measured Acceptance

Local acceptance completed **30/30 standard trials and 600 evaluations**, with
no timeout, interruption, unknown duration, or evidence-validation warning. The
two new smokes completed **6/6 trials and 36 evaluations**. Trials used seeds
0 through 4 for standards and seed 0 for smokes; no failed trial was replaced.

MO, 24 evaluations per trial (6 initial), n=5 complete trials per strategy:

| Strategy | Final hypervolume median [Q25, Q75] | Final Pareto count median |
| --- | --- | --- |
| qLogEHVI | 53.0163 [51.5876, 54.4172] | 11 |
| Random | 4.32061 [0, 7.87111] | 6 |
| Sobol | 16.9545 [3.59104, 17.6260] | 3 |

MF, 16 evaluations per trial (8 initial), n=5 complete trials per strategy:

| Strategy | Observed-target regret median [Q25, Q75] | Oracle projected regret median | Cumulative modeled cost median | Target observations median |
| --- | --- | --- | --- | --- |
| qMFKG | 53.6478 [38.4258, 60.4258] | 2.16585 | 7.71575 | 2 |
| Target-only random | 5.32276 [1.99202, 12.2570] | 1.99202 | 13.7157 | 10 |
| Target-only Sobol | 5.19790 [4.08801, 8.73516] | 4.08801 | 13.7157 | 10 |

All five qMFKG trials selected only lower-fidelity evaluations after initialization.
Consequently their primary observed-target metric did not improve after the first
two observations. The better oracle-projected scores do not repair this lack of
target observations. At this fixed count and these settings qMFKG used less modeled
cost but had worse observed-target quality than the baselines. This is retained
scientific evidence, not a reason to tune seeds or alter optimizer defaults. MO
results also remain descriptive fixed-protocol evidence, not a general superiority
claim. Five seeds do not establish broad scientific performance.

Full evidence is retained in `reports/benchmarks/v3.3.2-multi-objective-standard/`
and `reports/benchmarks/v3.3.2-multi-fidelity-standard/`, with corresponding
`*-smoke/` runs. The environment recorded BO Forge 3.3.2, Python 3.12.0,
BoTorch 0.17.2, Torch 2.11.0, GPyTorch 1.15.2, NumPy 2.5.2, pandas 3.0.5,
macOS 26.3 ARM64, one CPU thread per worker, and revision
`35f66202a6b71dd4a195b662a6bf1fb1f93adfaf` with uncommitted changes. These are not
fixed-runner performance measurements; runtime components are retained in reports.

All nine retained v3.3.0/v3.3.1 runs were revalidated and regenerated into separate
`report-v332-verified/` destinations without optimization or objective evaluation.
Their traces, summaries, trajectories, and trial tables match the prior reviewed
reports. The five constrained-BO failures and the earlier failed pending smoke
remain intact and are **not resolved** by v3.3.2. No production BO changes were made.
The new benchmark checks do not replace exact-commit CI or publication approval.

### Source Commands

Use an installed BO Forge environment from the root of a checkout or extracted
source archive. See [Installation](INSTALLATION.md). The sdist includes
`benchmarks/**/*.py` and `benchmarks/specs/*.yaml`; the runtime wheel excludes
`benchmarks/`. Installing the sdist alone does not install benchmark commands:
retain the extracted tree and run from its root. There is no new console entry
point or dependency.

```bash
python -m benchmarks --help
python -m benchmarks run --spec benchmarks/specs/smoke.yaml --output /tmp/bo-forge-benchmark-smoke
python -m benchmarks report --run /tmp/bo-forge-benchmark-smoke --output /tmp/bo-forge-benchmark-report
```

The general commands are `python -m benchmarks run --spec PATH --output NEW_DIR`
and `python -m benchmarks report --run DIR --output NEW_DIR`. Use new output
directories, not existing campaign or previous report directories. Any trial
directory is rejected as a report destination, including a new path within it.
A run automatically generates `report/`. Regeneration reads stored artifacts without
fitting models, evaluating objectives, or mutating the run directory.

## Budgets And Specifications

The original two specs retain `schema_version: 1`. Evaluation totals include initial
observations, not additional post-initialisation steps.

| Spec | Problems | Seeds | Modes/strategies | Initial / total evaluations | Trials |
| --- | --- | --- | --- | --- | --- |
| `smoke.yaml` | Branin | 0 | both modes, all three strategies | 4 / 6 | 6 |
| `standard.yaml` | Branin, Hartmann3, Hartmann6 | 0, 1, 2, 3, 4 | both modes, all three strategies | `2 * (d + 1)` / 24 | 90 |

Trials run sequentially, one trial worker process at a time, with
`timeout_seconds: 600` per trial and no trial parallelism. The smoke has two BO
trials with two post-initialisation steps each: only four BO suggestion calls.
The notebook uses only the full six-trial smoke. The CI numerical job retains
that smoke and adds the three schema-v2 route smokes below, never the standards.
The timeout is a per-trial bound, not a promised whole-run duration.

The standard is an explicit, separately launched run, never part of notebook
Run All. Its initial counts are 6, 8, and 14 for dimensions 2, 3, and 6.

```bash
# Opt-in only; retain all outcomes, including failed and partial trials.
python -m benchmarks run --spec benchmarks/specs/standard.yaml --output /tmp/bo-forge-benchmark-standard
```

The smoke schema is:

```yaml
schema_version: 1
name: smoke
seeds: [0]
modes: [deterministic, noisy]
strategies: [bo, random, sobol]
evaluations: 6
initial_observations: 4
timeout_seconds: 600
problems:
  - name: branin
    noise_std: 1.0
    optimum_tolerance: 1.0e-6
bo:
  raw_samples: 8
  num_restarts: 1
  mc_samples: 16
  min_normalized_distance: 0
```

`initial_observations` is an integer or `twice_dimension_plus_one`, which means
`2 * (d + 1)` in this schema. The standard uses that symbolic value, 24
evaluations, seeds 0 through 4, and all three problem entries. Branin's noisy
standard deviation is 1.0; Hartmann3/Hartmann6 use 0.05. Branin uses
`optimum_tolerance: 1.0e-6`; Hartmann3/Hartmann6 use `1.0e-5` to account for
BoTorch's rounded reference optima. Deterministic mode adds no observation noise.
Standard BO settings are `raw_samples: 128`, `num_restarts: 5`,
`mc_samples: 128`, and `min_normalized_distance: 0`; smoke settings are 8, 1,
16, and 0 respectively. These small smoke settings are for integration coverage,
not an optimisation-quality recommendation.

### Schema-v2 Route Specifications

Six additional files use `schema_version: 2` and an explicit `route` selector.
All paths below are under `benchmarks/specs/`; each uses `bo`, `random`, and
`sobol` with the same per-route seeds, initialization, and noise streams.

| Spec | Route / problem | Mode | Seeds | Initial / total evaluations | Trials |
| --- | --- | --- | --- | --- | --- |
| `mixed_smoke.yaml` | `mixed` / `mixed_quadratic` | deterministic | 0 | 4 / 6 | 3 |
| `constrained_mixed_smoke.yaml` | `constrained_mixed` / `mixed_quadratic` | deterministic | 0 | 4 / 6 | 3 |
| `pending_noisy_smoke.yaml` | `pending_noisy` / `branin` | noisy, `noise_std: 1.0` | 0 | 4 / 6 | 3 |
| `mixed_standard.yaml` | `mixed` / `mixed_quadratic` | deterministic | 0, 1, 2, 3, 4 | 8 / 24 | 15 |
| `constrained_mixed_standard.yaml` | `constrained_mixed` / `mixed_quadratic` | deterministic | 0, 1, 2, 3, 4 | 8 / 24 | 15 |
| `pending_noisy_standard.yaml` | `pending_noisy` / `branin` | noisy, `noise_std: 1.0` | 0, 1, 2, 3, 4 | 6 / 24 | 15 |

The new smoke budget is **9 trials / 54 evaluations** in total; each route has
3 trials / 18 evaluations. The new standard budget is **45 trials / 1,080
evaluations**; each route has 15 trials / 360 evaluations. These totals exclude
the original schema-v1 runs and describe scheduled budgets, not results.
The 600-second per-trial bound and sequential worker model remain unchanged.

```bash
python -m benchmarks run --spec benchmarks/specs/mixed_smoke.yaml --output /tmp/bo-forge-benchmark-mixed-smoke
python -m benchmarks run --spec benchmarks/specs/constrained_mixed_smoke.yaml --output /tmp/bo-forge-benchmark-constrained-mixed-smoke
python -m benchmarks run --spec benchmarks/specs/pending_noisy_smoke.yaml --output /tmp/bo-forge-benchmark-pending-noisy-smoke
# Standards are separate, explicit acceptance runs, never notebook Run All.
python -m benchmarks run --spec benchmarks/specs/mixed_standard.yaml --output /tmp/bo-forge-benchmark-mixed-standard
python -m benchmarks run --spec benchmarks/specs/constrained_mixed_standard.yaml --output /tmp/bo-forge-benchmark-constrained-mixed-standard
python -m benchmarks run --spec benchmarks/specs/pending_noisy_standard.yaml --output /tmp/bo-forge-benchmark-pending-noisy-standard
```

The mixed routes exercise continuous, integer, discrete, and categorical inputs
through existing `log_ei` behavior; the constrained variant additionally checks
feasibility. The pending route uses `qlog_nei` with delayed observations and
accepted pending rows supplied through `X_pending`. This is a sequential
workflow check, not an asynchronous-throughput benchmark. Inspect feasibility
and pending-row evidence alongside quality and timing; never replace failed
BO with a baseline or hide a failed attempt behind a retry.

The controlled mixed fixture minimizes
`(x - 0.25)^2 + 0.125*(k - 2)^2 + 0.25*(z - 0.5)^2 + penalty[c]`, with
`x` in `[-1, 1]`, integer `k` in `[0, 4]`, `z` in `{0, 0.5, 1}`, and
`c` in `{A, B, C}`. Penalties are `{A: 0.5, B: 0, C: 0.25}`. Its minimum
is zero. The constrained route adds `c != 'B' or k >= 3` and
`x + 0.125*k <= 0.625`, giving a feasible minimum of `0.125`.
Tests independently enumerate the finite choices and minimize the remaining
one-dimensional quadratic. These are controlled fixtures, not scientific applications.

Finite choices are decoded using equal-width bins. A shared feasible, unique
Sobol initialization precedes each strategy. A total limit of 10,000 proposal
draws per trial covers initialization and baseline sampling, including infeasible
and duplicate rejections. BO candidates are never repaired or retried by the harness.
For pending trials, initialization is fully observed. Each subsequent cycle
submits and accepts A, then submits and accepts B while A is pending, before
computing or observing either outcome. Observation order is A then B; noise is
indexed by submission. Completion requires an empty pending queue.

## Scientific And Random-Stream Contract

The continuous simulator uses the installed supported BoTorch `Branin` and `Hartmann`
implementations, native bounds, double precision, CPU, and `negate=False`.
Campaigns minimize `outcome`. Deterministic campaigns use `log_ei`; noisy
campaigns use `qlog_nei`. Baselines pass external single-point `random` or `sobol`
suggestions through the same managed initialization, append, mark-observed, and
reload operations. Strategy identity belongs in `trial.json`, not new CSV columns.

For each schema-v1 problem/seed/mode, SHA-256 of compact JSON
`[1, problem, seed, mode, stream_name]`, truncated to the first four bytes as an
unsigned big-endian integer, derives four streams: `initialization`, `baseline`,
`observation_noise`, and `fitting`. The mapping is recorded per trial. Strategies
share scrambled Sobol initial points and initial observations. The Sobol baseline
continues that sequence; the random baseline uses its separate NumPy generator.
Version 2 instead hashes `[2, route, problem, seed, mode, stream_name]`; it never
changes the v1 derivation. Typed mixed designs use canonical compact JSON
fingerprints, preserving integer and categorical values rather than object-array
memory bytes. Reports verify v2 typed metadata and the seeded initialization
fingerprint without fitting a model or evaluating an objective.
Observation noise uses another local generator indexed by evaluation count, so
optimizer random-number consumption cannot alter simulator noise. Backend fitting
receives the fitting seed through existing config semantics; this is not a claim
of bit-identical fits across software versions or platforms.

Only observed outcomes enter campaign YAML/CSV and the optimizer. True objective
values, optimum references, and simulator noise parameters stay in benchmark
metadata/scoring artifacts. This separation follows the
[BoTorch closed-loop noisy-optimization example](https://botorch.org/docs/v0.17.2/tutorials/closed_loop_botorch_only).
Simple regret is the best sampled noise-free value minus the reference minimum.
The observation-selected incumbent is the first design with the smallest observed
outcome; its latent regret can worsen as noisy selection changes. Only negative
regret within the explicit optimum-rounding tolerance is set to zero; raw regret
is retained, and larger negative values fail the trial.

## Retained Artifacts

```text
RUN_DIR/
  run.json                         # Run metadata and planned trial IDs
  spec.yaml                        # Specification snapshot
  trials/<id>/
    campaign.yaml
    campaign.csv
    campaign.csv.manifest.json     # Campaign provenance manifest
    status.json
    trace.jsonl
    worker.log
    trial.json
    inputs.json                       # Bounds, scoring references, input identities
    warnings.jsonl                    # When warnings occurred
  report/
    traces.csv
    trials.csv
    summary.csv
    trajectories.csv
    report.md
    figures/{problem}-{mode}.png
```

The trial layout describes generated artifacts, not a promise that every file
exists for a worker that fails before initialisation. Preserve the manifest and
any referenced provenance assets with the campaign. `run.json` supplies planned
trial IDs so missing or never-started work is not silently dropped.

Statuses are `pending`, `running`, `complete`, `failed`, `timeout`, and
`interrupted`. Partial traces and worker logs remain diagnostic evidence after
failure, timeout, or interruption; they are not completed trials. Inspect
`status.json`, `trial.json` when present, and `worker.log` before attributing a
failure to the optimiser. Report generation is not proof that a run completed.

The command handles Ctrl+C and SIGTERM by stopping its owned worker and recording
unfinished trials as interrupted (exit 130 and 143 respectively). Signal handlers
are restored when the run exits. Forced termination such as SIGKILL, power loss,
or an unwritable filesystem cannot guarantee cleanup; preserve the directory and
inspect its evidence rather than treating it as a completed run or resuming it.
Numerical CI uploads each smoke directory even after step failure, with a 14-day
artifact-retention limit; download evidence needed for longer-lived investigation.

`trials.csv` exposes per-trial outcomes and denominators. `summary.csv` groups
by route, problem, mode, and strategy, with completed-trial final regret median and
interquartile bounds (`final_regret_median`, `final_regret_q25`, and
`final_regret_q75`). `trajectories.csv` reports `contributing_complete`,
`contributing_partial`, and `scheduled` at each evaluation count; trajectory
median and quartiles use completed trials only. Partial traces remain visible
without contributing to completed-trial quality aggregates. Figures use the
same route/problem/mode grouping, not a single pooled cross-problem ranking.
For v2, `feasibility.json` records proposals and rejection counts; `events.jsonl`
records ordered submission, acceptance, and observation evidence. Reports include
`proposal_draws`, `infeasible_proposals`, `duplicate_proposals`,
`workflow_event_count`, and `suggestions_with_pending`. Scoring order must match
observation order; interrupted trailing events remain explicit warnings.

Reports bind campaign YAML, acquisition, model profile, optimizer settings,
initialization, source labels, seed mapping, and `inputs.json` to the scheduled
trial. Internal agreement among copied artifacts alone is insufficient.
The strictly parsed `spec.yaml` snapshot must also match the resolved specification
in `run.json`, not merely its stored byte hash. The runner parses and retains one
byte snapshot so edits to the input spec after loading cannot alter the recorded plan.
Initialization must use `sobol`; baseline sources must match `random` or `sobol`;
BO sources must match `log_ei` or `qlog_nei` as scheduled. Contradictions identify
the trial, field, expected value, and actual value.

Complete trials require a valid matching manifest and exact observed CSV/trace
agreement. Missing or changed provenance, extra campaign rows, malformed evidence,
and trace fields that override scheduled identities are rejected. Present malformed
or mismatched provenance is also rejected for failed, timed-out, and interrupted
trials; a failed status does not turn contradictory evidence into a warning. This is an
integrity check, not a signature or tamper-proof audit. Reports never recover a
pending transaction or repair source files. Incomplete trials can retain partial
traces with an explicit `evidence_warning` in `trials.csv` and the Markdown report;
in particular, an observation persisted before an interrupted trace write is
reported as unscored, not silently counted as a scored evaluation. It does not
enter completed-trial aggregates.
An absent artifact or a pending transaction whose CSV matches its previous or
intended resulting hash can be disclosed for incomplete trials without repair.
An unknown pending state is rejected. Workflow acceptance/observation events must
agree with persisted review/observation states. A persisted mutation with a missing
event is disclosed as interrupted event recording, not fabricated as an event;
an event claiming a mutation absent from the CSV is rejected.

Per-seed plots use strategy colors and distinct seed line patterns. Hollow circle
markers identify partial trials; filled circles identify completed trials. The
report states the specification's actual seed count, including single-seed smoke
runs, rather than assuming every run contains five seeds.

`suggestion_seconds` includes model fitting and acquisition optimization for BO;
`objective_seconds` covers simulation/scoring, and `mutation_seconds` covers
campaign persistence and reload. These sum completed steps only. Per-trial wall
time additionally includes startup and any failed or interrupted work. On
cancellation, active-trial monotonic elapsed time includes worker shutdown.
Missing historical `wall_seconds` is unknown, never `0.0`; a measured zero
remains distinct. Reports expose known-duration subtotals and unknown timing
counts, so partial timing is not presented as a complete runtime total. These are
diagnostic timings on the recorded environment, not fixed-runner performance claims.
Summary fields `known_wall_seconds`, `known_timing_trials`, and
`unknown_timing_trials` accompany `wall_seconds`; the latter stays blank whenever
any contributor lacks timing. Markdown displays "not available" instead of zero.

## Read The Evidence

- Compare strategies within the same problem, mode, and seed before aggregating.
  Paired seed comparisons are not independent replications of each strategy.
- Distinguish observed noisy values from the latent noise-free objective at the
  sampled design. A favourable noisy observation is not improved latent quality;
  observed best-so-far and latent regret answer different questions.
- Quality summaries are conditioned on completed trials. Read planned and
  completed counts together with failures, timeouts, interruptions, runtime,
  and partial trajectories. Survivor-only quality is not an unconditional
  success rate, and failed trials must not become zero-regret observations.
- A single-seed smoke checks integration, not scientific acceptance. The
  repeated-seed standard also requires inspection of variation and failure
  rates, not exact stochastic trajectories or a universal calibration claim.
- Retain the spec snapshot and environment/run metadata when reporting results.
  Do not replace an unsuccessful run with an undisclosed successful rerun.

The output-free
[24_closed_loop_benchmarks.ipynb](../notebooks/24_closed_loop_benchmarks.ipynb)
runs only the smoke in a temporary directory, displays stored reports and trial
statuses, and regenerates a report without rerunning optimisation. It deliberately
does not leave benchmark results in the checkout. Its computational cells remain
unchanged; appended Markdown describes opt-in route commands only.

## Release Boundary

### v3.3.1 Measured Acceptance

Local acceptance on 2026-09-18 retained every scheduled terminal outcome. These
measurements do not certify general scientific performance or exact-commit CI.

| Gate | Planned scope | Measured result |
| --- | --- | --- |
| New route smoke | 9 trials / 54 evaluations | 9 complete, 54 evaluations |
| New route standard | 45 trials / 1,080 evaluations | 40 complete, 5 failed; 1,003 evaluations |
| Read-only report regeneration | New routes and retained v3.3.0 | Coherent evidence; no optimization rerun |
| Exact-commit CI | Release commit | Pending; separate from local evidence |

| Route | Complete / failed | Evaluations | Measured worker wall seconds |
| --- | ---: | ---: | ---: |
| Mixed | 15 / 0 | 360 | 145.13 |
| Constrained mixed | 10 / 5 | 283 | 115.86 |
| Pending noisy | 15 / 0 | 360 | 176.85 |

There were no timeouts, interruptions, unknown timings, or evidence-integrity
warnings in these standard runs. The five constrained BO trials exhausted the
existing eight candidate retries on infeasible designs after 8, 8, 9, 9, and 9
observations (seeds 0 through 4). All remain in failure denominators with partial
trajectories; their final-regret summaries are unavailable. No optimizer settings,
seeds, or algorithms were changed to improve those outcomes. Baseline sampling
recorded 221 infeasible proposals across the constrained suite. All three
strategies followed the delayed schedule, with 135 total submissions while one
accepted design was pending; 45 of those submissions belong to BO.

Median final noise-free regret for completed trials only (five seeds per available
cell; compare within a route, never pool these scales):

| Route | BO | Random | Sobol |
| --- | ---: | ---: | ---: |
| Mixed | 0.00140385 | 0.312713 | 0.0746520 |
| Constrained mixed | not available (0/5 complete) | 0.186451 | 0.232780 |
| Pending noisy | 0.0616857 | 2.098872 | 0.862824 |

Evidence is retained under `reports/benchmarks/v3.3.1-mixed-standard/`,
`v3.3.1-constrained-mixed-standard/`, and `v3.3.1-pending-noisy-standard/`.
Their `report-integrity-reviewed/` directories contain the final regenerated tables and figures;
original reports remain preserved.
The corrected route smokes are retained alongside them as `v3.3.1-*-smoke/`.
An earlier pending smoke failed all three trials before observation because the
harness passed the status `accepted` instead of the API decision `accept`.
That corrected harness defect and its original evidence remain separately visible
under `v3.3.1-pending-smoke-review-call-failure/`; the later passing smoke did not
replace it. The original v1 smoke also completed 6/6 trials, and the retained v3.3.0
90-trial evidence passed the stronger checks without rerunning optimization.

Runs recorded BO Forge 3.3.1, Python 3.12.0, BoTorch 0.17.2, Torch 2.11.0,
macOS 26.3 ARM64, and one computational thread per worker. Source identity was
`46c00a6614fd9f80ac5f2b17210098f163085be7` with uncommitted v3.3.1 changes.
Timings include process startup/shutdown, exclude report generation, and are not
fixed-runner performance evidence. Generated evidence remains local and ignored;
no publication approval is implied.

### Historical v3.3.0 Local Standard Acceptance

On 2026-09-17, the standard specification completed all 90 scheduled trials and
2,160 evaluations, with zero failures, timeouts, or interruptions. All 440
model-based calls recorded successful fits, no fallback, and no captured fit
warnings. Full local evidence is retained under
`reports/benchmarks/v3.3.0-standard/` (ignored generated artifacts, not packaged
source). The run recorded BO Forge 3.3.0, Python 3.12.0, BoTorch 0.17.2, Torch
2.11.0, macOS 26.3 ARM64, and one computational thread per worker. Its Git
identity is `3c153f52f804fb8cec7daf7d09b1ede8826e7561` plus the uncommitted v3.3.0
implementation, not an exact-commit release verification. Trial execution took
797.76 seconds; report rendering is outside that timing.

Median final noise-free simple regret after 24 evaluations (five complete seeds
per cell; lower is better within each row):

| Problem | Observation mode | BO | Random | Sobol |
| --- | --- | ---: | ---: | ---: |
| Branin | deterministic | 0.116690 | 1.104932 | 1.033431 |
| Branin | noisy | 0.124859 | 1.649314 | 1.754872 |
| Hartmann3 | deterministic | 0.011139 | 0.514356 | 0.404618 |
| Hartmann3 | noisy | 0.009447 | 0.504305 | 0.591744 |
| Hartmann6 | deterministic | 0.799000 | 1.965795 | 2.068123 |
| Hartmann6 | noisy | 0.621083 | 1.905249 | 2.039987 |

The generated report retains quartiles, individual seed trajectories, noisy
observation-selected incumbents, and runtime components. These measurements
describe this fixed budget, seed set, software, and environment. They are not a
cross-problem score, significance test, calibration result, or assurance that BO
wins on each seed or future campaign. Both real smoke routes and notebook #24
also completed; the notebook's committed execution state remains output-free.

Local smoke results, standard-run measurements, exact-commit CI, and publication
authorization are separate gates. Append measured acceptance results only after
the corresponding runs have been inspected. Mixed/constrained and pending-aware
routes are prepared in v3.3.1 with the failures disclosed above;
multi-objective/multi-fidelity evidence is recorded separately for v3.3.2,
the verified full notebook baseline for v3.3.3, and passed local performance acceptance
for v3.3.4 in [Performance Benchmarks](PERFORMANCE_BENCHMARKS.md). See the
[roadmap](../ROADMAP_V3_X.md) and [release checklist](RELEASE_CHECKLIST.md).
