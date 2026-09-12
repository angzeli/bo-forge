# Predictive Evaluation

This is the scientific and workflow reference for predictive diagnostics through
v3.2.1. It describes the implementation and how to inspect its results; it
does not certify that release gates, calibration studies, or v3.2.x acceptance
have passed. See the [release checklist](RELEASE_CHECKLIST.md) and
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

For a **hand-calculated illustration**, not an executed campaign result, take
`y = 12 U`, `mu = 10 U`, and observation-inclusive variance `4 U^2`. Then
`sigma = 2 U`, `r = 2 U`, `z = 1`, NLPD is approximately `2.1121`, and the interval
is approximately `[6.0801, 13.9199] U`; this observation is covered. Using latent
variance instead would answer a different question and typically narrow the
interval incorrectly for an observed outcome.

NLPD uses natural logarithms and the numerical objective units in the input.
Lower is better for the same outcomes and units, but NLPD is not unit-invariant:
rescaling the objective by a factor `a` shifts NLPD by `log(abs(a))`. Negative
NLPD is possible for a continuous density and is not itself an error.

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

| Summary metric | Definition | Interpretation |
| --- | --- | --- |
| `rmse` | `sqrt(sum(r_i^2) / n)` | Error magnitude in `U`, with greater weight on large errors. |
| `mae` | `sum(abs(r_i)) / n` | Mean absolute error in `U`. |
| `mean_nlpd` | `sum(NLPD_i) / n` | Gaussian predictive-density score; compare only on the same outcomes and units. |
| `interval_coverage` | `sum(abs(z_i) <= c) / n` | Fraction covered, between 0 and 1; nominal target is 0.95. |
| `mean_interval_width` | `sum(2 * c * sigma_i) / n` | Interval width in `U`; read together with coverage, not alone. |

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
broader synthetic calibration studies, and full v3.2.x acceptance remain separate
work. No automatic campaign execution, profile change, or selection recommendation
is part of this evaluator.
