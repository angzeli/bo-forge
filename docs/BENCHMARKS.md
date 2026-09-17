# Closed-Loop Benchmarks

BO Forge v3.3.0 prepares a bounded, source-only benchmark harness for continuous
single-objective optimisation. It compares `bo`, `random`, and `sobol` on Branin,
Hartmann3, and Hartmann6 in `deterministic` and `noisy` modes. It changes no BO
algorithm, public campaign API, or Streamlit workflow. Measured acceptance is
recorded below; it does not establish general superiority over either baseline.

## Run From Source

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

Both checked-in specs use `schema_version: 1`. Evaluation totals include initial
observations, not additional post-initialisation steps.

| Spec | Problems | Seeds | Modes/strategies | Initial / total evaluations | Trials |
| --- | --- | --- | --- | --- | --- |
| `smoke.yaml` | Branin | 0 | both modes, all three strategies | 4 / 6 | 6 |
| `standard.yaml` | Branin, Hartmann3, Hartmann6 | 0, 1, 2, 3, 4 | both modes, all three strategies | `2 * (d + 1)` / 24 | 90 |

Trials run sequentially, one trial worker process at a time, with
`timeout_seconds: 600` per trial and no trial parallelism. The smoke has two BO
trials with two post-initialisation steps each: only four BO suggestion calls.
The CI numerical job and notebook use the full six-trial smoke, not the standard.
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

## Scientific And Random-Stream Contract

The simulator uses the installed supported BoTorch `Branin` and `Hartmann`
implementations, native bounds, double precision, CPU, and `negate=False`.
Campaigns minimize `outcome`. Deterministic campaigns use `log_ei`; noisy
campaigns use `qlog_nei`. Baselines pass external single-point `random` or `sobol`
suggestions through the same managed initialization, append, mark-observed, and
reload operations. Strategy identity belongs in `trial.json`, not new CSV columns.

For each problem/seed/mode, SHA-256 of compact JSON
`[1, problem, seed, mode, stream_name]`, truncated to the first four bytes as an
unsigned big-endian integer, derives four streams: `initialization`, `baseline`,
`observation_noise`, and `fitting`. The mapping is recorded per trial. Strategies
share scrambled Sobol initial points and initial observations. The Sobol baseline
continues that sequence; the random baseline uses its separate NumPy generator.
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
Numerical CI uploads the smoke directory even after step failure, with a 14-day
artifact-retention limit; download evidence needed for longer-lived investigation.

`trials.csv` exposes per-trial outcomes and denominators. `summary.csv` groups
by problem, mode, and strategy, with completed-trial final regret median and
interquartile bounds (`final_regret_median`, `final_regret_q25`, and
`final_regret_q75`). `trajectories.csv` reports `contributing_complete`,
`contributing_partial`, and `scheduled` at each evaluation count; trajectory
median and quartiles use completed trials only. Partial traces remain visible
without contributing to completed-trial quality aggregates. Figures use the
same problem/mode grouping, not a single pooled cross-problem ranking.

Complete trials require a valid matching manifest and exact observed CSV/trace
agreement. Missing or changed provenance, extra campaign rows, malformed evidence,
and trace fields that override scheduled identities are rejected. This is an
integrity check, not a signature or tamper-proof audit. Reports never recover a
pending transaction or repair source files. Incomplete trials can retain partial
traces with an explicit `evidence_warning` in `trials.csv` and the Markdown report;
in particular, an observation persisted before an interrupted trace write is
reported as unscored, not silently counted as a scored evaluation. It does not
enter completed-trial aggregates.

Per-seed plots use strategy colors and distinct seed line patterns. Hollow circle
markers identify partial trials; filled circles identify completed trials. The
report states the specification's actual seed count, including single-seed smoke
runs, rather than assuming every run contains five seeds.

`suggestion_seconds` includes model fitting and acquisition optimization for BO;
`objective_seconds` covers simulation/scoring, and `mutation_seconds` covers
campaign persistence and reload. These sum completed steps only. Per-trial wall
time additionally includes startup and any failed or interrupted work. These are
diagnostic timings on the recorded environment, not fixed-runner performance claims.

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
does not leave benchmark results in the checkout.

## Release Boundary

### Local Standard Acceptance

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
routes remain v3.3.1 work; multi-objective/multi-fidelity evidence is v3.3.2,
full notebook execution v3.3.3, and performance closeout v3.3.4. See the
[roadmap](../ROADMAP_V3_X.md) and [release checklist](RELEASE_CHECKLIST.md).
