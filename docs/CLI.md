# 💻 CLI Workflow

The `bo-forge` command wraps the same `CampaignSession` workflow used in notebooks.
It exposes the same BO behaviour as the package API; it makes validation, suggestions, reporting, and plotting usable from the terminal.

For a runnable notebook version of this workflow, open `notebooks/04_cli_four_variable_campaign.ipynb`.

## Versioned JSON Inspection

Starting in v3.4.0, put `--format json` (or `--format=json`) **after the subcommand**.
Text remains the default; `--format text` is equivalent to omitting the option.
The supported commands are `validate`, `summary`, `status`, `next-action`,
`cost-summary`, `replicate-summary`, `stage-summary`, `context-summary`,
`fidelity-summary`, `fidelity-coverage`, `qlog-nei-summary`, `model-summary`,
`model-compare`, `pareto-front`, `pareto-summary`, and `provenance`.

```bash
bo-forge validate --config configs/01_simple_2d_maximise_logei.yaml \
  --log examples/01_simple_2d_maximise_logei_campaign_log.csv --format json
```

```json
{"schema_version":1,"bo_forge_version":"3.4.0","command":"validate","ok":true,"data":{"valid":true},"error":null}
```

Every handled JSON request emits one newline-terminated object on stdout.
Diagnostics, warnings, and backend progress go to stderr. Help and version remain
textual. Exit codes remain `0` for success, `1` for handled operational failure,
and `2` for argument errors. Recognized JSON inspection requests also use the
envelope for missing arguments, invalid profiles, and unknown options:

```json
{"schema_version":1,"bo_forge_version":"3.4.0","command":"summary","ok":false,"data":null,"error":{"code":"argument_error","message":"the following arguments are required: --config, --log","hint":null,"details":{}}}
```

The last valid `--format` value selects the error format; an invalid or missing
value on a repeated flag does not erase an earlier JSON request. Unambiguous
option abbreviations follow the same parser rules, though full flags are
recommended for scripts. Options after `--` are not interpreted as format flags.

Tables have `columns` and `records`, including column names for empty results.
Row/profile order and native values are preserved. Missing and non-finite numbers
become `null`; strings such as `"001"` remain strings. Unsupported objects produce
`serialization_error`, not silently stringified data. `status` returns a `status`
string; `validate` returns `valid: true`. The
[machine-readable schema](../schemas/cli-inspection-v1.json) allows additive v1
fields. Removing fields or changing meanings/types requires an explicitly selected
new schema version, never a silent change to v1.
Each record has exactly the keys listed in `columns`; consumers should check
this dynamic correspondence separately from JSON Schema validation. Numeric
scalars support Python integers/floats and NumPy integers and floats up to
64-bit precision. Fractions, decimals, and extended-precision floating values
are rejected rather than silently rounded.

Operational error codes are `config_error`, `log_validation_error`,
`log_write_error`, `log_conflict_error`, `log_busy_error`, `suggestion_error`,
`provenance_error`, `provenance_recovery_required`, and the base `bo_forge_error`.
Each error contains `code`, `message`, nullable `hint`, and `details`; provenance
reason and recovery fields remain in `details` when available. Failed provenance
inspection retains its table when available. Inspection never repairs files.
Unexpected programming exceptions are not converted to success or generic JSON.

```python
import json
import subprocess

result = subprocess.run(
    ["bo-forge", "summary", "--config", "campaign.yaml", "--log", "campaign.csv",
     "--format", "json"], capture_output=True, text=True, check=False,
)
payload = json.loads(result.stdout)
if payload["schema_version"] != 1:
    raise RuntimeError("Unsupported inspection schema")
if result.returncode != 0 or not payload["ok"]:
    error = payload["error"]
    if error and error["code"] == "provenance_recovery_required":
        print(error["details"].get("recovery_action"))  # Explicit recovery is separate.
    raise RuntimeError(error)
rows = payload["data"]["records"]
```

Branch on codes/structured fields rather than prose. `ok` means the command
executed successfully, not that models are scientifically validated: `model-compare`
remains **in-sample** diagnostics and can successfully return failed-profile rows.
`next-action` is advice, not execution. Suggestions, mutations, reports/exports,
predictive evaluation, doctor, and launchers reject the format flag before execution.
JSON inspection adds no fitting to commands that did not already fit models.

## 🧰 Install

```bash
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev]"
```

Check the installed command:

```bash
bo-forge --version
bo-forge doctor
```

The equivalent module invocation is also supported:

```bash
python -m bo_forge --version
```

Use `bo-forge ...` in a normal terminal. Use `python -m bo_forge ...` when you want to guarantee that the command runs with a specific Python interpreter, such as inside notebooks or editable development environments.

## 📓 Using The CLI From Notebooks

Inside notebooks, prefer calling the CLI through the current notebook Python:

```python
subprocess.run(
    [sys.executable, "-m", "bo_forge", "next-action", *CAMPAIGN_ARGS],
    cwd=PROJECT_ROOT,
    check=True,
)
```

This is equivalent to running `bo-forge next-action ...` in a terminal, but it avoids relying on a shell alias or console script path. `check=True` makes the notebook stop clearly if the CLI command fails.

## 🔁 Basic Workflow

Create an empty canonical campaign log and provenance manifest from a config:

```bash
bo-forge init-log \
  --config configs/01_simple_2d_maximise_logei.yaml \
  --log examples/my_new_campaign_log.csv
```

The manifest is written beside the log as `<log>.manifest.json`. Existing
campaigns without a manifest remain legacy-compatible and are not adopted
automatically. Keep a managed CSV and sidecar together. Inspect either state with:

```bash
bo-forge provenance --config configs/01_simple_2d_maximise_logei.yaml \
  --log examples/my_new_campaign_log.csv
```

The command exits nonzero when the log is missing or a managed campaign reports
`mismatch` or `pending_recovery`; it remains a read-only diagnostic and does not repair
files.

Campaign-loading commands accept `--require-provenance`. Without it, a missing
manifest is treated as a legacy campaign for backward compatibility. With it, a
missing manifest fails; any present malformed or mismatched manifest fails under
either mode. When inspection reports `pending_previous_state` or
`pending_resulting_state`, recover explicitly and then reload:

```bash
bo-forge provenance-recover \
  --config configs/01_simple_2d_maximise_logei.yaml \
  --log examples/my_new_campaign_log.csv \
  --expected-log-fingerprint CURRENT_LOG_SHA256
```

The recovery command changes only the manifest. It never changes YAML or CSV bytes.

Validate a campaign log:

```bash
bo-forge validate \
  --config configs/01_simple_2d_maximise_logei.yaml \
  --log examples/01_simple_2d_maximise_logei_campaign_log.csv
```

Inspect state:

```bash
bo-forge status \
  --config configs/01_simple_2d_maximise_logei.yaml \
  --log examples/01_simple_2d_maximise_logei_campaign_log.csv

bo-forge summary \
  --config configs/01_simple_2d_maximise_logei.yaml \
  --log examples/01_simple_2d_maximise_logei_campaign_log.csv

bo-forge next-action \
  --config configs/01_simple_2d_maximise_logei.yaml \
  --log examples/01_simple_2d_maximise_logei_campaign_log.csv
```

Generate suggestions without changing the campaign log:

```bash
bo-forge suggest \
  --config configs/01_simple_2d_maximise_logei.yaml \
  --log examples/01_simple_2d_maximise_logei_campaign_log.csv \
  --batch-size 1
```

The same commands work for mixed-variable and constrained configs such as `configs/05_simple_mixed_logei.yaml` and `configs/06_mixed_constrained_logei.yaml`. Constraint violations fail during `validate`; generated suggestions are filtered to satisfy configured constraints.

For noisy or pending-aware single-objective configs, use `bo.acquisition:
qlog_nei`. Accepted review suggestions are treated as active pending
experiments and passed to qLogNEI as `X_pending`; review rows that are still
`pending` must be accepted, rejected, or deferred first:

```bash
bo-forge validate \
  --config configs/18_noisy_pending_qlognei.yaml \
  --log examples/18_noisy_pending_qlognei_campaign_log.csv

bo-forge qlog-nei-summary \
  --config configs/18_noisy_pending_qlognei.yaml \
  --log examples/18_noisy_pending_qlognei_campaign_log.csv

bo-forge suggest \
  --config configs/18_noisy_pending_qlognei.yaml \
  --log examples/18_noisy_pending_qlognei_campaign_log.csv \
  --batch-size 1

bo-forge plot \
  --config configs/18_noisy_pending_qlognei.yaml \
  --log examples/18_noisy_pending_qlognei_campaign_log.csv \
  --kind qlog-nei-diagnostics \
  --output /tmp/bo_forge_qlog_nei_diagnostics.png
```

For noisy or pending-aware coupled multi-objective configs, use
`bo.acquisition: qlog_nehvi`. Accepted review suggestions are treated as active
pending designs and passed to qLogNEHVI as `X_pending`; review rows that are
still `pending` must be resolved first:

```bash
bo-forge validate \
  --config configs/19_multi_objective_qlognehvi.yaml \
  --log examples/19_multi_objective_qlognehvi_campaign_log.csv

bo-forge suggest \
  --config configs/19_multi_objective_qlognehvi.yaml \
  --log examples/19_multi_objective_qlognehvi_campaign_log.csv \
  --batch-size 1
```

For contextual configs, context variables are normal CSV variable columns but
are fixed at suggestion time. Provide every context value with repeatable
`--context NAME=VALUE`, or define `context.default_values` in the YAML:

```bash
bo-forge validate \
  --config configs/16_contextual_logei.yaml \
  --log examples/16_contextual_logei_campaign_log.csv

bo-forge suggest \
  --config configs/16_contextual_logei.yaml \
  --log examples/16_contextual_logei_campaign_log.csv \
  --context feedstock_acidity=0.25 \
  --batch-size 1

bo-forge context-summary \
  --config configs/16_contextual_logei.yaml \
  --log examples/16_contextual_logei_campaign_log.csv

bo-forge plot \
  --config configs/16_contextual_logei.yaml \
  --log examples/16_contextual_logei_campaign_log.csv \
  --kind context-diagnostics \
  --output /tmp/bo_forge_context_diagnostics.png
```

Contextual LogEI campaigns may also use review metadata, deterministic cost,
replicates, or their combinations. Cost expressions are evaluated on the full
candidate, including fixed context values, and budget accounting is
campaign-global across all contexts. Active repeats only target replicate
groups matching the requested context:

```bash
bo-forge suggest \
  --config configs/20_contextual_cost_review_logei.yaml \
  --log examples/20_contextual_cost_review_campaign_log.csv \
  --context feedstock_acidity=0.5 \
  --batch-size 1

bo-forge context-summary \
  --config configs/20_contextual_cost_review_logei.yaml \
  --log examples/20_contextual_cost_review_campaign_log.csv

bo-forge cost-summary \
  --config configs/20_contextual_cost_review_logei.yaml \
  --log examples/20_contextual_cost_review_campaign_log.csv
```

The combined contextual replicate example uses the same interface:

```bash
bo-forge suggest \
  --config configs/21_contextual_replicate_logei.yaml \
  --log examples/21_contextual_replicate_campaign_log.csv \
  --context feedstock_acidity=0.25 \
  --batch-size 2
```

For single-objective model profiles, the CSV schema is unchanged. Inspect the
configured profile and fitting inputs, compare profiles read-only, then export
model diagnostics. Fit metadata belongs to the session, not ambient process
history; a fresh `model-summary` invocation shows `not_recorded` for
`last_fit_status` and `fallback_status`, with warning count `0` and empty warning
text. `model-compare` retains its columns and adds
`evaluation_scope=in_sample`. These are training-row metrics, not held-out
predictive evidence. `model-compare` is diagnostic only; it does
not change the configured profile or automatically select a model. Repeated
`--profile` flags must name distinct profiles. Failed or insufficient profile
fits are reported in `fit_status` with details in `fit_message`.

```bash
bo-forge model-summary \
  --config configs/17_model_profile_logei.yaml \
  --log examples/17_model_profile_campaign_log.csv

bo-forge model-compare \
  --config configs/17_model_profile_logei.yaml \
  --log examples/17_model_profile_campaign_log.csv \
  --profile default \
  --profile smooth

bo-forge plot \
  --config configs/17_model_profile_logei.yaml \
  --log examples/17_model_profile_campaign_log.csv \
  --kind model-diagnostics \
  --output /tmp/bo_forge_model_diagnostics.png

bo-forge plot \
  --config configs/17_model_profile_logei.yaml \
  --log examples/17_model_profile_campaign_log.csv \
  --kind model-comparison \
  --output /tmp/bo_forge_model_comparison.png
```

For explicit held-out evaluation, supply a standard single-objective
config and a log containing 5..200 observations with no duplicate designs:

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

`--profile` is repeatable. `--folds` defaults to `5` and accepts `2..5`;
`--seed` defaults to `0`. Every fold needs at least two training rows.
Context, replicates, stages, fidelity, and multi-objective campaigns are rejected.
The command reports profile summaries and fold outcomes; `--output-dir` requests
artifact export. It does not append observations, change the configured profile,
or run automatically as part of suggestion generation or normal reporting.
It exits `0` only when every requested profile is complete, and exits `1` if any
profile is incomplete. Failed-fold reasons remain visible without export;
requested diagnostic exports are retained even when evaluation exits `1`.
The appended summary `fit_message` distinguishes failed folds from aggregate
numerical failures. Export success messages identify incomplete results explicitly.
The export destination must not exist: export refuses overwrite and writes only
`summary.csv`, `predictions.csv`, `fold_outcomes.csv`, and `metadata.json`.
An existing destination is rejected before fitting; export checks again before
writing in case another process created that destination during evaluation.
The four files are prepared in a temporary sibling and published atomically
without overwrite on macOS/Linux. Failed serialization or publication leaves no
partial final bundle; a competing destination is never removed. The Python and
Streamlit result can retry export without refitting; rerunning the CLI starts a
new evaluation. File-loaded config/log/manifest changes during evaluation raise
`LogConflictError`; reload campaign state before retrying.
Plots require the explicit result plotting methods described in the Python API.
Campaign-aware plot and evaluation exports reject campaign-source aliases and
reserved manifest paths, including directories nested under a missing legacy
sidecar. Choose a separate artifact path; these errors leave the campaign intact.
Summary metrics are `rmse`, `mae`, `mean_nlpd`, `interval_coverage`, and
`mean_interval_width`, with `fit_status=complete` or `incomplete`.
Predictive variance includes observation noise in original objective units
squared. Held-out metrics on small adaptive datasets are not proof of calibration
or automatic model-selection evidence. See [Public API](PUBLIC_API.md) and the
[temporary-directory tutorial](../notebooks/23_predictive_diagnostics.ipynb).

For single-objective multi-fidelity configs, the fidelity variable is a normal
CSV variable column. Continuous fidelity and ordered numeric levels use the
same commands; qMFKG accepts batch sizes from one through four:

```bash
bo-forge validate \
  --config configs/22_discrete_multi_fidelity_qmfkg.yaml \
  --log examples/22_discrete_multi_fidelity_qmfkg_campaign_log.csv

bo-forge suggest \
  --config configs/22_discrete_multi_fidelity_qmfkg.yaml \
  --log examples/22_discrete_multi_fidelity_qmfkg_campaign_log.csv \
  --batch-size 2

bo-forge fidelity-summary \
  --config configs/22_discrete_multi_fidelity_qmfkg.yaml \
  --log examples/22_discrete_multi_fidelity_qmfkg_campaign_log.csv

bo-forge fidelity-coverage \
  --config configs/22_discrete_multi_fidelity_qmfkg.yaml \
  --log examples/22_discrete_multi_fidelity_qmfkg_campaign_log.csv

bo-forge plot \
  --config configs/22_discrete_multi_fidelity_qmfkg.yaml \
  --log examples/22_discrete_multi_fidelity_qmfkg_campaign_log.csv \
  --kind fidelity-diagnostics \
  --output reports/22_discrete_multi_fidelity_diagnostics.png

bo-forge plot \
  --config configs/22_discrete_multi_fidelity_qmfkg.yaml \
  --log examples/22_discrete_multi_fidelity_qmfkg_campaign_log.csv \
  --kind fidelity-progress \
  --output reports/22_discrete_multi_fidelity_progress.png
```

Continuous-fidelity batches use joint one-shot optimization. Ordered
discrete-fidelity batches are constructed greedily with BoTorch mixed
fixed-feature optimization, then assigned one joint post-selection acquisition
value.

Optional `fidelity.optimizer_maxiter` and
`fidelity.optimizer_timeout_seconds` settings control the existing optimizer;
they do not add CLI flags. The timeout starts after model fitting, spans
target-value optimization and candidate retries, and rejects candidate batches
returned after the shared deadline. BoTorch initial-condition generation and
an in-flight optimizer call cannot be cancelled immediately, so the command can
return later than the configured limit. The timeout is not a quality guarantee.

The generated model-based row uses `source=qmf_kg`. Use a copied working log
before `--append`, as with the other examples.

For structured campaigns, pass one configured stage name explicitly. Generated
rows populate the `stage` column, fill only variables active in that stage, and
leave inactive variables blank:

```bash
bo-forge init-log \
  --config configs/13_structured_campaign_core.yaml \
  --log examples/13_structured_campaign_core_working_log.csv

bo-forge suggest \
  --config configs/13_structured_campaign_core.yaml \
  --log examples/13_structured_campaign_core_working_log.csv \
  --stage screen
```

Use `--append` only when you want to append the generated stage-aware rows to
the canonical CSV log. Structured campaigns with multiple stages fail clearly
when `--stage` is omitted.

Inspect stage status without mutating the log:

```bash
bo-forge stage-summary \
  --config configs/13_structured_campaign_core.yaml \
  --log examples/13_structured_campaign_core_campaign_log.csv
```

Cost-aware and review-enabled campaigns use the same rhythm with extra review and cost commands. Inspect the current budget state:

```bash
bo-forge cost-summary \
  --config configs/07_cost_aware_human_review_logei.yaml \
  --log examples/07_cost_aware_human_review_working_log.csv
```

For replicate-aware campaigns, inspect group-level replicate statistics:

```bash
bo-forge replicate-summary \
  --config configs/08_replicate_aware_logei.yaml \
  --log examples/08_replicate_aware_working_log.csv
```

For deterministic multi-objective campaigns, `summary` includes Pareto and hypervolume fields, and `suggest` uses qLogEHVI after the initial design:

```bash
cp examples/10_multi_objective_mixed_constrained_campaign_log.csv \
  examples/10_multi_objective_mixed_constrained_working_log.csv

bo-forge suggest \
  --config configs/10_multi_objective_mixed_constrained_qlogehvi.yaml \
  --log examples/10_multi_objective_mixed_constrained_working_log.csv \
  --batch-size 2
```

Copy the seed log before mutating commands so the committed example CSV stays unchanged.

Generate suggestions, save a suggestions CSV, and append the same suggestions to the canonical log:

```bash
bo-forge suggest \
  --config configs/01_simple_2d_maximise_logei.yaml \
  --log examples/01_simple_2d_maximise_logei_working_log.csv \
  --output examples/01_simple_2d_maximise_logei_latest_suggestions.csv \
  --append
```

After running the experiment, record the result:

```bash
bo-forge mark-observed \
  --config configs/01_simple_2d_maximise_logei.yaml \
  --log examples/01_simple_2d_maximise_logei_working_log.csv \
  --row-id ROW_ID_FROM_SUGGESTIONS \
  --objective-value 1.95
```

For multi-objective campaigns, provide one `--objective name=value` argument per configured objective:

```bash
bo-forge mark-observed \
  --config configs/10_multi_objective_mixed_constrained_qlogehvi.yaml \
  --log examples/10_multi_objective_mixed_constrained_working_log.csv \
  --row-id ROW_ID_FROM_SUGGESTIONS \
  --objective yield_score=71.2 \
  --objective waste_score=13.4
```

The objective names must exactly match the YAML config, with no missing, duplicate, or unknown names. `--objective-value` is single-objective only.

For review-enabled campaigns, accept, reject, or defer suggestions before running them:

```bash
bo-forge review \
  --config configs/07_cost_aware_human_review_logei.yaml \
  --log examples/07_cost_aware_human_review_working_log.csv \
  --row-id ROW_ID_FROM_SUGGESTIONS \
  --decision accept \
  --note "run next"
```

Accepted suggestions can then be marked observed with an optional realised cost:

```bash
bo-forge mark-observed \
  --config configs/07_cost_aware_human_review_logei.yaml \
  --log examples/07_cost_aware_human_review_working_log.csv \
  --row-id ROW_ID_FROM_SUGGESTIONS \
  --objective-value 68.4 \
  --actual-cost 2.7
```

For multi-objective review-enabled campaigns, use the same `review` command, then mark accepted rows observed with repeated `--objective name=value` arguments. If the multi-objective config has a `cost:` section, add `--actual-cost VALUE` to record realised cost.

## 📄 Reports And Plots

Print a plain-text campaign report:

```bash
bo-forge report \
  --config configs/01_simple_2d_maximise_logei.yaml \
  --log examples/01_simple_2d_maximise_logei_working_log.csv
```

Export the same deterministic report format:

```bash
bo-forge report \
  --config configs/01_simple_2d_maximise_logei.yaml \
  --log examples/01_simple_2d_maximise_logei_working_log.csv \
  --output reports/latest_campaign_report.txt
```

Export one figure per command:

```bash
bo-forge plot \
  --config configs/01_simple_2d_maximise_logei.yaml \
  --log examples/01_simple_2d_maximise_logei_working_log.csv \
  --kind progress \
  --output reports/progress.png

bo-forge plot \
  --config configs/01_simple_2d_maximise_logei.yaml \
  --log examples/01_simple_2d_maximise_logei_working_log.csv \
  --kind diagnostics \
  --output reports/diagnostics.png

bo-forge plot \
  --config configs/07_cost_aware_human_review_logei.yaml \
  --log examples/07_cost_aware_human_review_working_log.csv \
  --kind cost-progress \
  --output reports/cost_progress.png

bo-forge plot \
  --config configs/08_replicate_aware_logei.yaml \
  --log examples/08_replicate_aware_working_log.csv \
  --kind replicates \
  --output reports/replicates.png

bo-forge plot \
  --config configs/10_multi_objective_mixed_constrained_qlogehvi.yaml \
  --log examples/10_multi_objective_mixed_constrained_working_log.csv \
  --kind pareto \
  --output reports/pareto.png

bo-forge plot \
  --config configs/10_multi_objective_mixed_constrained_qlogehvi.yaml \
  --log examples/10_multi_objective_mixed_constrained_working_log.csv \
  --kind hypervolume \
  --output reports/hypervolume.png

cp examples/11_four_objective_mixed_constrained_campaign_log.csv \
  examples/11_four_objective_mixed_constrained_working_log.csv

bo-forge plot \
  --config configs/11_four_objective_mixed_constrained_qlogehvi.yaml \
  --log examples/11_four_objective_mixed_constrained_working_log.csv \
  --kind pareto-parallel \
  --output reports/pareto_parallel.png
```

## 🧭 Command Reference

| Command | Description |
| --- | --- |
| `bo-forge --version` | Print the installed BO Forge version. |
| `bo-forge doctor` | Check the active BO Forge environment and key imports. |
| `python -m bo_forge --version` | Run the same CLI through a specific Python interpreter. |
| `bo-forge init-log --config PATH --log PATH` | Create an empty canonical campaign CSV log and versioned provenance manifest. |
| `bo-forge validate --config PATH --log PATH` | Validate a YAML config and CSV campaign log. |
| `bo-forge summary --config PATH --log PATH` | Print campaign counts, status, and best observation as readable text. |
| `bo-forge status --config PATH --log PATH` | Print exactly one campaign status line. |
| `bo-forge next-action --config PATH --log PATH` | Print the recommended next campaign action. |
| `bo-forge cost-summary --config PATH --log PATH` | Print cost, reserved-cost, budget, and either best-observed-objective or multi-objective hypervolume/Pareto fields. |
| `bo-forge replicate-summary --config PATH --log PATH` | Print group-level replicate counts, mean, std, SEM, min, and max. |
| `bo-forge stage-summary --config PATH --log PATH` | Print structured stage counts, active/inactive variables, warnings, and transition-readiness guidance. |
| `bo-forge fidelity-summary --config PATH --log PATH` | Print observed fidelity counts, target-fidelity coverage, pending qMFKG count, and direction-aware best rows. |
| `bo-forge fidelity-coverage --config PATH --log PATH` | Print per-fidelity modeled cost, observed statistics, active suggestion counts, and direction-aware best rows. |
| `bo-forge context-summary --config PATH --log PATH` | Print contextual observed counts, pending suggestions, and direction-aware best rows by context combination. |
| `bo-forge model-summary --config PATH --log PATH` | Print configured model profile, model class, covariance profile, fitting-row count, and train-Y variance use; a fresh CLI invocation has no session-owned fit evidence. |
| `bo-forge model-compare --config PATH --log PATH [--profile NAME ...]` | Compare model profiles on current observed fitting rows without changing CSV logs or the configured profile. |
| `bo-forge model-evaluate --config PATH --log PATH [--profile NAME ...] [--folds N] [--seed N] [--output-dir PATH]` | Explicit bounded held-out predictive evaluation; no automatic execution or model selection. |
| `bo-forge qlog-nei-summary --config PATH --log PATH` | Print qLogNEI observed baseline rows, active pending rows, review blockers, initial-design readiness, train-Y variance availability, and model profile. |
| `bo-forge pareto-front --config PATH --log PATH` | Print nondominated observed rows for a multi-objective campaign. |
| `bo-forge pareto-summary --config PATH --log PATH` | Print objective count, reference points, Pareto count, and hypervolume fields. |
| `bo-forge report --config PATH --log PATH [--output PATH]` | Print or export a deterministic campaign report. |
| `bo-forge provenance --config PATH --log PATH` | Print managed/legacy provenance status, identities, hashes, environment count, event count, and pending-transaction state. |
| `bo-forge provenance-adopt --config PATH --log PATH` | Preview legacy adoption; tracking starts now and earlier history is unknown. |
| `bo-forge provenance-migrate --config PATH --log PATH` | Preview explicit schema-v1 to schema-v2 migration. |
| `bo-forge provenance-accept-config --config PATH --log PATH` | Preview formatting-only config acceptance after explicit v2 migration. |
| `bo-forge provenance-fork --config PATH --log PATH --destination DIR [--config-changes JSON]` | Preview a child with inherited CSV bytes and restricted config changes. |
| `bo-forge provenance-recover --config PATH --log PATH [--expected-log-fingerprint SHA256]` | Explicitly finalize or cancel a recoverable pending manifest transaction without changing YAML or CSV bytes. |
| `bo-forge suggest --config PATH --log PATH [--batch-size N] [--stage STAGE_NAME] [--context NAME=VALUE ...] [--output PATH] [--append]` | Generate suggestions; append only when `--append` is passed. Structured campaigns use `--stage`; contextual campaigns use repeatable `--context`. |
| `bo-forge review --config PATH --log PATH --row-id ROW_ID --decision accept\|reject\|defer [--note TEXT]` | Record one human review decision. |
| `bo-forge mark-observed --config PATH --log PATH --row-id ROW_ID --objective-value VALUE [--actual-cost VALUE]` | Mark one pending suggestion as observed. |
| `bo-forge mark-observed --config PATH --log PATH --row-id ROW_ID --objective NAME=VALUE --objective NAME=VALUE [...] [--actual-cost VALUE]` | Mark a multi-objective pending suggestion observed, optionally with realised cost when cost is configured. |
| `bo-forge plot --config PATH --log PATH --kind progress\|diagnostics\|model-diagnostics\|model-comparison\|cost-progress\|replicates\|pareto\|pareto-parallel\|hypervolume\|stage-diagnostics\|fidelity-diagnostics\|fidelity-progress\|context-diagnostics\|qlog-nei-diagnostics --output PATH` | Export one supported campaign figure, including fidelity-distribution diagnostics or target-fidelity progress. |

## 🧯 CLI Error Output

Expected user-facing failures print `Error: ...` to stderr and exit with code `1`.
Most config, CSV, suggestion, and log-write errors also include a short `Hint: ...` line.

Missing required arguments use normal `argparse` behavior and exit with code `2`.

For detailed YAML and CSV fixes, see [COMMON_ERRORS.md](COMMON_ERRORS.md).
For copyable intentional failure examples, see [CLI_ERROR_EXAMPLES.md](CLI_ERROR_EXAMPLES.md).

## ⚠️ Mutation Rules

Most commands are read-only.

The commands that can change files are:

- `bo-forge init-log`: creates a new empty campaign log and provenance manifest and refuses to overwrite either file.
- `bo-forge provenance-recover`: updates only a managed manifest to resolve a recoverable interrupted transaction.

Lifecycle commands default to JSON previews. Redirect the preview to a file, inspect
it, then pass `--apply --reason "..." --preview FILE`. Apply rechecks all source
identities under lock; stale previews fail without modifying campaign data.

```bash
bo-forge provenance-adopt --config campaign.yaml --log campaign.csv > adoption-preview.json
bo-forge provenance-adopt --config campaign.yaml --log campaign.csv \
  --apply --reason "Start tracking validated legacy data" --preview adoption-preview.json
```

Fork `--config-changes` accepts a JSON mapping such as
`'{"campaign_name":"child","bo":{"random_seed":42}}'`. Resolve suggested rows first;
the destination must not exist. Back up referenced archives with managed campaign files.
- `bo-forge suggest --append`: appends generated suggestions as `status=suggested`.
- `bo-forge suggest --output`: writes a standalone suggestions CSV, even without `--append`.
- `bo-forge review`: updates `review_status` and `review_note` for one suggested row.
- `bo-forge mark-observed`: marks one existing pending row as `status=observed`.
- `bo-forge report --output`: writes a report file.
- `bo-forge plot --output`: writes a figure file.

Suggestion CSV and report exports reject destinations that alias the loaded
campaign's YAML, CSV, reserved manifest path, or referenced provenance archives.
The manifest filename is reserved case-insensitively, including for legacy logs.
Choose a separate artifact path; existing ordinary artifact files can still be
replaced. A rejected destination leaves campaign files unchanged, including when
`suggest --output` is combined with `--append`.

`bo-forge suggest --append` never marks suggestions observed. The explicit campaign rhythm remains:

> suggest → append → run experiment → mark-observed

For review-enabled campaigns, the explicit rhythm is:

> suggest → append → review → run accepted experiment → mark-observed

BO Forge serializes append, review, and observation mutations with the same
canonical-log lock used by sessions, Streamlit, and the API on one machine.
Each CLI mutation loads a current fingerprint before writing. A conflicting or
busy log fails clearly without overwriting newer rows; multi-host shared-file
coordination remains unsupported.

For provenance-managed campaigns, successful append, review, and observation
mutations also update the sidecar ledger under the same lock. Read-only commands
do not add events. Interrupted transactions block ordinary loading and mutation until
`bo-forge provenance-recover` is run explicitly. See
[PROVENANCE.md](PROVENANCE.md).

Conflict errors tell the user to reload and inspect the latest campaign before
retrying. Busy errors tell the user to wait briefly for the active local writer.
