# Predictive Evaluation

This is the scientific and workflow reference for predictive diagnostics through
v3.2.3. It describes the implementation and how to inspect its results; it
does not certify universal model calibration or publication approval.
See the [release checklist](RELEASE_CHECKLIST.md) and
[roadmap](../ROADMAP_V3_X.md) for those separate gates.

## In-Sample Versus Held-Out

`model_profile_comparison` and the existing model-diagnostics plots evaluate the
rows used for fitting. Their existing columns, including `rmse_model_space`,
`mae_model_space`, and `mean_predicted_std`, are retained with
`evaluation_scope=in_sample`. They are training-fit diagnostics, not held-out
prediction or evidence for automatically selecting a model.

`model_predictive_evaluation` instead fits a fresh model for each profile and
training fold, then predicts the excluded observations. Its summary and metadata
use `evaluation_scope=out_of_fold`. Evaluation is explicit, does not generate
suggestions, and does not modify the campaign CSV or configured profile.

## Supported Inputs And Splits

- Standard single-objective campaigns only; context, replicates, structured
  stages, fidelity, and multi-objective campaigns are rejected.
- The full campaign data must validate. Only observed rows enter evaluation;
  pending/suggested rows are not held-out outcomes or acquisition `X_pending`.
- There must be 5..200 observed rows, with no duplicate observed designs. There
  is no automatic subsampling or aggregation of repeated designs.
- `folds` is an integer from 2 through 5, default `5`; each training fold must
  contain at least two observations.
- `seed` is a nonnegative integer, default `0`. It controls shuffled fold
  assignment, not an independent guarantee of deterministic optimizer retries.
- `profiles=None` evaluates only the configured profile. Otherwise request one
  to four distinct names from `default`, `smooth`, `rough`, and `robust`.

Observed rows are sorted by string row ID before seeded shuffling and partitioning
into nearly equal folds. Every profile receives the same splits, and every
observation is held out once per profile. Each fold learns its own input
normalization, outcome standardization, and GP parameters from training rows.
Configured variable bounds and categories define the shared input encoding.

Each fold records the CPU Torch RNG fingerprint immediately before fitting under
the fit-warning lock. These fingerprints distinguish stochastic retry inputs;
they do not restore RNG state or guarantee identical fitted parameters. Evaluation
does not reseed the process. The existing fitter's retry policy remains unchanged.

## Original-Unit Predictive Uncertainty

Let `y_i` be an observed objective value and `mu_i` its out-of-fold predictive
mean. The evaluator requests `model.posterior(..., observation_noise=True)`.
For the supported learned-noise single-objective GP, the reported variance is
conceptually

\[
\sigma_i^2 = \operatorname{Var}(f(x_i)\mid D_{-k}) + \sigma_{\mathrm{noise},k}^2,
\qquad \sigma_i = \sqrt{\sigma_i^2}.
\]

Here `D_{-k}` excludes the observation's entire held-out fold. This is predictive
uncertainty for an **observation**, not only the latent function. It includes the
fitted observation-noise term but is not a full integration over uncertainty in
estimated hyperparameters or model choice.

BoTorch's outcome transform returns posterior moments to the training objective
scale. BO Forge then reverses its maximization sign convention for minimization
means. Variance does not change under a sign reversal. If the objective has units
`U`, means, raw residuals, standard deviations, and interval endpoints have units
`U`; variance has units `U^2`. Standardized residuals are dimensionless.

The per-observation calculations are

\[
r_i = y_i-\mu_i, \qquad z_i = r_i/\sigma_i,
\]

\[
\mathrm{NLPD}_i = \tfrac12\left[\log(2\pi)+\log(\sigma_i^2)+z_i^2\right],
\qquad I_i = [\mu_i-c\sigma_i,\;\mu_i+c\sigma_i],
\]

where `c = 1.959963984540054` and `I_i` is a nominal 95% Gaussian predictive
interval. Coverage uses inclusive endpoints. Nonfinite means, nonpositive or
nonfinite variance, or nonfinite derived prediction metrics fail the fold.

Coverage is computed as `abs(z_i) <= c`, and width as `2 * c * sigma_i`,
without comparing or subtracting rounded original-unit endpoints. At large
objective offsets, the stored floating-point endpoints can coincide even when
predictive uncertainty and interval width are positive.

For a **hand-worked example**, not fitted profiles or an executed campaign
result, hold `y = 12 U` and `mu = 10 U` fixed. Both illustrations have absolute
error `2 U` (and single-row RMSE and MAE of `2 U`):

| Illustration | Observation std (U) | Standardized residual | Covered | Interval width (U) | NLPD |
| --- | --- | --- | --- | --- | --- |
| A | 0.5 | 4.0 | no | 1.9600 | 8.2258 |
| B | 2.0 | 1.0 | yes | 7.8399 | 2.1121 |

The intervals are approximately `[9.0200, 10.9800] U` and
`[6.0801, 13.9199] U`, respectively. Increasing uncertainty changes coverage,
width, and density despite identical errors. This one outcome cannot establish
that either illustration is calibrated or recommend a BO Forge profile.
Using latent variance instead would answer a different question and typically
narrow the interval incorrectly for an observed outcome.

NLPD uses natural logarithms and the numerical objective units in the input.
Lower is better for the same outcomes and units, but NLPD is not unit-invariant:
rescaling the objective by a factor `a` shifts NLPD by `log(abs(a))`. Negative
NLPD is possible for a continuous density and is not itself an error.
For example, multiply `y`, `mu`, and `sigma` by `10`: error and width multiply
by `10`, variance by `100`, standardized residual and coverage stay unchanged,
and NLPD increases by approximately `2.3026`. This is a change of numerical
units, not a change in predictive quality.

## Result Contract

Both entry points return `PredictiveEvaluationResult(summary, predictions,
fold_outcomes, metadata)`.

| Member | Meaning |
| --- | --- |
| `summary` | One row per requested profile, scope, `fit_status`, observed-row count, completed/total folds, the five aggregate metrics below, and appended `fit_message`. |
| `predictions` | One row per observed row per profile, with `row_id`, `fold`, `observed`, `fit_status`, `fit_message`, and predictive quantities. |
| `fold_outcomes` | One row per fold/profile, training and held-out counts, completion/failure status, message, and fit-warning evidence. |
| `metadata` | Method, split seed, profiles, fold membership, objective name/direction, input identities, software versions, units, interval probability, and interpretation limits. |

Prediction quantities are `predicted_mean`, `predicted_variance`, `predicted_std`,
`residual`, `standardized_residual`, `negative_log_predictive_density`,
`interval_lower`, `interval_upper`, and `interval_covered`.

For a complete profile, the `n` out-of-fold rows are pooled with equal weight
per observation, not equal weight per fold:

| Summary metric | Measures / definition | Units | Valid comparison | Does not establish |
| --- | --- | --- | --- | --- |
| `rmse` | Error magnitude, weighted toward large errors: `sqrt(sum(r_i^2) / n)`. | `U` | Same held-out outcomes, units, splits, and fitting conditions; lower means less squared error here. | Calibration or future BO performance. |
| `mae` | Mean absolute error: `sum(abs(r_i)) / n`. | `U` | Same comparison inputs as RMSE; lower means less absolute error here. | Absence of a few large errors or reliable uncertainty. |
| `mean_nlpd` | Gaussian density assigned to outcomes: `sum(NLPD_i) / n`. | Natural-log density score; depends on numerical units | Same outcomes, units, splits, and observation-inclusive variance convention; lower is better here. | A unit-independent score or a calibration certificate. |
| `interval_coverage` | Fraction inside nominal 95% intervals: `sum(abs(z_i) <= c) / n`. | Dimensionless fraction | Same inputs and nominal probability; read with width and residuals, not as a higher-is-better ranking. | Calibration from proximity to 0.95 on one small dataset. |
| `mean_interval_width` | Mean interval span: `sum(2 * c * sigma_i) / n`. | `U` | Same inputs and nominal probability; read with coverage, not as a lower-is-better ranking. | Accuracy or calibration from narrow intervals alone. |

Summary `fit_status` is `complete` or `incomplete`; prediction/fold status is
`complete` or `failed`. If any fold fails, that profile's aggregate metrics are
withheld rather than computed from its successful subset. Successful predictions
remain inspectable, failed rows remain present with missing predictive values,
and other profiles continue. Nonfinite aggregate metrics also produce an
incomplete summary. Inspect warnings and failures even when some plots look good.
`complete` means computational completion, not scientific validation.
Summary `fit_message` is empty for complete profiles. For an incomplete profile,
it lists failed folds and their messages, or explains an aggregate numerical
failure even when every fold completed. Successful row predictions remain visible.
Scaled, equivalent reductions avoid unnecessary overflow; outcomes and variances
are never clipped or replaced by artificial floors or zero-valued failures.

Warning evidence is not an exhaustive optimizer history. BO Forge serializes
model construction and fitting together so their captured warnings belong to
the correct fit. Construction follows the active warning filters; fitting does
too, except for the `robust` profile's explicit capture policy. BoTorch may handle
retries internally. A zero `fit_warning_count`, empty
`fit_warnings`, or `fallback_status=not_needed` is therefore not proof that no
numerical warning or internal retry occurred. A posterior-prediction failure
retains the successful fit's captured warning evidence while marking the fold
failed and withholding aggregate metrics. Interpret these fields alongside
`fit_status` and `fit_message`, not as a convergence or calibration certificate.

## Read A Result

1. **Check completeness and warnings.** Read `fit_status`, `fit_message`, and
   fold evidence first. Do not compare withheld metrics with complete scores or
   replace missing values with zeros. A complete run is only a completed computation.
2. **Confirm comparison inputs.** Use the same observation IDs and values,
   objective units, held-out membership, and documented fitting conditions.
   Compare `metadata` source identities, `fold_membership`, software versions,
   and the per-fold fitting RNG fingerprints. Record intended differences such
   as profile settings; do not attribute every difference to the profile when
   fitting conditions also differ. Requested profile order is not a ranking.
3. **Inspect predictive errors.** Read RMSE and MAE in the objective's units,
   then inspect individual held-out predictions. Relate error sizes to the
   experiment's requirements; BO Forge supplies no universal acceptable threshold.
4. **Read uncertainty jointly.** Inspect standardized residuals, NLPD, coverage,
   and width together. A large standardized residual is an error large relative
   to its reported observation uncertainty, not an automatic outlier diagnosis.
5. **State the limits.** Distinguish computational completion, retrospective
   predictive error, uncertainty calibration, and future BO performance. These
   are different claims; none follows automatically from the previous one.

Gneiting and Raftery distinguish calibration from forecast concentration and
develop proper scores for distributions and intervals. Their interval-score
discussion considers both width and missed coverage; a log score also evaluates
the density assigned to the realized outcome. BO Forge reports the existing
metrics separately, not a new interval or composite score. Neither coverage nor
width alone tests calibration. See [Gneiting and Raftery (2007), sections 1, 4,
and 6](https://doi.org/10.1198/016214506000001437)
([author-hosted paper](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)).

| Pattern | What to inspect next, without an automatic diagnosis |
| --- | --- |
| Low RMSE with poor coverage | Check residuals relative to reported uncertainty, units, individual rows, and fold messages. Small absolute error can still be large relative to a narrow interval. |
| High coverage with wide intervals | Examine whether the intervals are informative at the experimental scale. Covering outcomes by broad intervals does not establish accurate or calibrated predictions. |
| A few large standardized residuals | Inspect row identity, recorded outcome, predicted mean/variance, and fold evidence. Data errors, model mismatch, and underestimated uncertainty are possibilities to investigate, not conclusions. Do not delete rows merely to improve metrics. |
| Incomplete profiles | Keep failed rows and messages; resolve the reported failure before interpreting that profile's aggregate metrics. Successful predictions remain diagnostic evidence, not a substitute aggregate score. |

## Manual Split Sensitivity

This is a manual reporting protocol, not an extra evaluation mode or notebook loop:

1. Before inspecting results, predeclare a small seed set, for example `{0, 7, 19}`,
   the profile order, fold count, fixed source snapshot, and intended fitting
   conditions. Keep observations, objective units, and software environment fixed.
2. Run each declared seed explicitly with the existing evaluator. Within each
   run, give all requested profiles the same held-out membership. Export each
   run to its own new directory; retain every result, warning, and failure.
3. Report per-run metrics, completion counts, and descriptive variability such
   as the range across complete runs. List excluded/incomplete runs and their
   reasons alongside that range; never select only the best seed or profile.
4. Record that the split seed controls membership, not all fitting randomness.
   Without separately controlled fitting conditions, variability includes both
   split and fitting effects. These runs reuse observations and are not independent
   replications: do not attach significance claims or independent-binomial
   confidence intervals to their correlated coverage outcomes.

Repeatedly trying profiles or seeds and reporting only the best observed score
selects on finite-sample noise as well as model quality. Cawley and Talbot show
how overfitting a model-selection criterion can bias performance evaluation;
retaining the search history does not make the chosen score unbiased. See
[Cawley and Talbot (2010)](https://www.jmlr.org/beta/papers/v11/cawley10a.html).

## Python And Export

The public signatures are
`model_predictive_evaluation(config, df, profiles=None, *, folds=5, seed=0)` and
`campaign.model_predictive_evaluation(profiles=None, *, folds=5, seed=0)`.
For an existing validated campaign with enough observations:

```python
from pathlib import Path
from bo_forge import CampaignSession

campaign = CampaignSession.from_files("campaign.yaml", "observed.csv")
result = campaign.model_predictive_evaluation(
    profiles=["default", "smooth"], folds=3, seed=0,
)
print(result.summary)
print(result.fold_outcomes)

output_dir = Path("reports/evaluation")  # Must not already exist.
result.export(output_dir)
result.plot_predictions(save_path=output_dir / "predictions.png")
result.plot_residuals(save_path=output_dir / "residuals.png")
```

`result.export(output_dir)` creates a new destination, refuses overwrite, and
returns its path. It writes exactly `summary.csv`, `predictions.csv`,
`fold_outcomes.csv`, and `metadata.json`. It does not export plots or mutate
campaign state. Use a different new directory for a later evaluation; do not
pre-create the export directory. Missing parent directories may be created.
The files are prepared in a temporary sibling directory and published together
with atomic no-overwrite directory publication on macOS/Linux. A serialization
or publication failure leaves the requested destination absent unless another
process created it; that other destination is never removed or overwritten.
Retry `result.export(output_dir)` from the retained in-memory result after fixing
the cause, without fitting again. A competing or existing destination raises
`FileExistsError`; choose a different destination. A killed process or failed
temporary-directory cleanup may leave an unpublished `.preparing-` sibling,
not a partial final bundle. Cleanup errors never mask the original failure.

Session-produced results retain campaign-source protection when exported or
plotted. CLI and Streamlit exports reject config, log, manifest, and referenced
archive aliases, including symlinks and hard links. The manifest name and its
descendants are reserved even for legacy campaigns; choose a separate output
directory. Plot destinations are checked before rendering and again before
writing. Ordinary figure files may still be overwritten; evaluation bundles may
not. Standalone evaluations supplied only config/data have no source-file paths,
so callers must keep their output paths separate from campaign files.

Each evaluation uses an isolated config/data snapshot, so predictions, fold
membership, and recorded identities describe the same inputs. Standalone calls
need no campaign files. File-loaded sessions (and the CLI/service wrappers)
check config, log, and manifest identity before and after fitting, including for
legacy campaigns. A changed source raises `LogConflictError`; reload and run
again. Existing provenance/recovery policies remain enforced. Returned results
are historical snapshots, not live views of a subsequently changed campaign.

`result.plot_predictions(save_path=None)` and
`result.plot_residuals(save_path=None)` use the stored result without refitting;
`save_path` is keyword-only. The first plots observed versus held-out mean. The
second plots standardized residuals with zero and nominal 95% reference lines.
Its row axis is display order, not chronological campaign time. Omit `save_path`
for an unsaved figure.

Fit evidence is separate from evaluation metadata. The top-level `FitMetadata`
is a frozen scalar record accepted by
`model_summary(config, df, *, metadata=None)`. Without an explicit matching record,
the standalone summary reports `not_recorded` for fit status rather than reading
ambient fit history. `CampaignSession.model_summary()` supplies its own record,
matched against configuration and current training tensors. Evaluation/comparison
fits do not become that session's suggestion-fit history. This state is not
serialized campaign provenance, and a fresh session has no prior fit record.

## CLI And Streamlit

```bash
bo-forge model-evaluate \
  --config campaign.yaml \
  --log observed.csv \
  --profile default \
  --profile smooth \
  --folds 3 \
  --seed 0 \
  --output-dir reports/evaluation
```

`--profile` is repeatable; omit it for the configured profile. `--folds` defaults
to `5`, `--seed` to `0`, and `--output-dir` is optional. Output-directory rules
are the same as Python export; the CLI does not automatically create plots.
Use [CLI](CLI.md) for command context and [Streamlit](STREAMLIT_APP.md) for
explicit evaluation controls in `Analyze`. Loading a campaign, rerunning the
page, or generating an ordinary report must not initiate evaluation.
The CLI prints fold outcomes, including failure messages, and returns exit code
`1` if any requested profile is incomplete (`0` only when all are complete).
An incomplete evaluation still exports its diagnostic tables when requested.
Existing output destinations are rejected before fitting and checked again
before export.
Streamlit shows incomplete-profile and captured-warning notices above the
result tables. Fold details remain available, and successful exports of incomplete
results are labeled incomplete. Export errors retain the result for retry without
refitting. Campaign/input/policy changes still invalidate the current-result cache.
The collapsed **Interpret these results** reference beside the tables is static;
it neither fits nor exports and does not modify evaluation or campaign state.

## Existing Notebook Walkthrough

Use [23_predictive_diagnostics.ipynb](../notebooks/23_predictive_diagnostics.ipynb)
with an installed BO Forge development/notebook environment. From the repository
root, in that environment, run:

```bash
jupyter notebook notebooks/23_predictive_diagnostics.ipynb
```

Run cells in order. The notebook builds its YAML through the existing config
parser and writes 20 synthetic, unique observations inside a `TemporaryDirectory`;
it needs no campaign fixture. The explicit evaluation cell requests `default`
and `smooth`, three folds, and seed zero: six model fits in total. Inspect
`summary`, `fold_outcomes`, predictions, and metadata before interpreting plots.

The export cell uses a new child directory, verifies the four exported files,
then explicitly saves the two plots. Input-byte assertions check that evaluation
and export leave YAML/CSV unchanged. The final cell removes the temporary files.
To repeat the whole walkthrough, rerun setup to obtain a fresh directory; rerunning
only export against the same destination is expected to fail. Keep committed
notebook outputs and execution counts cleared. These are instructions, not a
claim that this walkthrough or the release suite has passed in every environment.

## Acceptance Evidence

The v3.2.3 acceptance suite separates two evidence levels:

- **Known-distribution diagnostic tests:** 200 unique designs use deterministic
  midpoint normal quantiles, known means, and observation variance 4. A controlled
  fitter supplies matched, too-narrow, too-wide, and biased Gaussian predictions
  through the public evaluator. Scenario names do not identify BO Forge profiles.
  Independent standard-library calculations check every prediction quantity and
  pooled metric, directions, profile order, folds, and source metadata. The matched
  case covers exactly 190/200 outcomes at the nominal 95% level by construction;
  this is a metric/reporting correctness check, not learned-model calibration.
- **Bounded fitted-model integration:** a real GP evaluates five observations,
  one profile, and two folds. It must return complete outputs, finite means and
  metrics, and positive finite observation variance. No exact score or 95%
  coverage is required of the fitted model. Numerical CI runs this test alongside
  qMFKG; installed wheel/sdist probes also evaluate and export a five-row campaign.

Complete and incomplete workflow tests retain the same evidence through session,
service, CLI, Streamlit, plots, and CSV/JSON exports. Incomplete metrics stay
missing; stored-result plotting and interpretation help never refit or select a
profile. Existing snapshot, provenance, metadata-isolation, atomic-export, and
source-file-protection tests remain part of acceptance, including plot aliases,
reserved manifest directories, and retained-result export retries. Linux full-suite and
macOS selections include the fast synthetic and workflow tests.

The existing notebook remains a six-fit workflow demonstration, not a calibration
study. Passing these checks establishes the tested diagnostic and adapter
contracts, not empirical calibration of arbitrary learned campaign models,
prospective performance, or closed-loop BO performance. The broader workflow
benchmark remains in v3.3.x; exact-commit CI is a separate publication gate.

## Limits Of Interpretation

Random K-fold evaluation is retrospective, not a replay of sequential BO. In an
adaptive campaign, later design locations can depend on earlier observed outcomes;
a random training fold may contain such later designs. Removing an observation
from model fitting does not remove that acquisition-history dependence. The
result is not an unbiased estimate of future optimization performance, regret,
or performance at unexplored designs.

Twenty held-out observations yield coverage steps of 0.05, and their predictions
share overlapping training sets. Do not treat the rows as independent calibration
trials or infer reliable uncertainty calibration from 19/20 covered outcomes.
One split is sensitive to the data and seed; trying many seeds/profiles and then
reporting only the winner introduces selection bias. Lower RMSE alone also says
nothing about interval calibration.

Use these diagnostics to expose obvious errors and motivate further checks,
not to automate model selection. Prospective or suitably chronological validation,
empirical calibration studies, and closed-loop BO benchmarking remain separate
work. No automatic campaign execution, profile change, or selection recommendation
is part of this evaluator.
