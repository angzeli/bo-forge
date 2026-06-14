# 💻 CLI Workflow

The `bo-forge` command wraps the same `CampaignSession` workflow used in notebooks.
It exposes the same BO behaviour as the package API; it makes validation, suggestions, reporting, and plotting usable from the terminal.

For a runnable notebook version of this workflow, open `notebooks/04_cli_four_variable_campaign.ipynb`.

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

Create an empty canonical campaign log from a config:

```bash
bo-forge init-log \
  --config configs/01_simple_2d_maximise_logei.yaml \
  --log examples/my_new_campaign_log.csv
```

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

For single-objective multi-fidelity configs, the fidelity variable is a normal
CSV variable column. Once the initial design is complete, qMFKG model-based
suggestions are one-at-a-time:

```bash
bo-forge validate \
  --config configs/15_multi_fidelity_qmfkg.yaml \
  --log examples/15_multi_fidelity_qmfkg_campaign_log.csv

bo-forge suggest \
  --config configs/15_multi_fidelity_qmfkg.yaml \
  --log examples/15_multi_fidelity_qmfkg_campaign_log.csv \
  --batch-size 1

bo-forge fidelity-summary \
  --config configs/15_multi_fidelity_qmfkg.yaml \
  --log examples/15_multi_fidelity_qmfkg_campaign_log.csv

bo-forge plot \
  --config configs/15_multi_fidelity_qmfkg.yaml \
  --log examples/15_multi_fidelity_qmfkg_campaign_log.csv \
  --kind fidelity-diagnostics \
  --output reports/15_multi_fidelity_diagnostics.png
```

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

For multi-objective campaigns, `summary` includes Pareto and hypervolume fields, and `suggest` uses qLogEHVI after the initial design:

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
| `bo-forge init-log --config PATH --log PATH` | Create an empty canonical campaign CSV log. |
| `bo-forge validate --config PATH --log PATH` | Validate a YAML config and CSV campaign log. |
| `bo-forge summary --config PATH --log PATH` | Print campaign counts, status, and best observation as readable text. |
| `bo-forge status --config PATH --log PATH` | Print exactly one campaign status line. |
| `bo-forge next-action --config PATH --log PATH` | Print the recommended next campaign action. |
| `bo-forge cost-summary --config PATH --log PATH` | Print cost, reserved-cost, budget, and either best-observed-objective or multi-objective hypervolume/Pareto fields. |
| `bo-forge replicate-summary --config PATH --log PATH` | Print group-level replicate counts, mean, std, SEM, min, and max. |
| `bo-forge stage-summary --config PATH --log PATH` | Print structured stage counts, active/inactive variables, warnings, and transition-readiness guidance. |
| `bo-forge fidelity-summary --config PATH --log PATH` | Print observed fidelity counts, target-fidelity coverage, pending qMFKG count, and direction-aware best rows. |
| `bo-forge pareto-front --config PATH --log PATH` | Print nondominated observed rows for a multi-objective campaign. |
| `bo-forge pareto-summary --config PATH --log PATH` | Print objective count, reference points, Pareto count, and hypervolume fields. |
| `bo-forge report --config PATH --log PATH [--output PATH]` | Print or export a deterministic campaign report. |
| `bo-forge suggest --config PATH --log PATH [--batch-size N] [--output PATH] [--append]` | Generate suggestions; append only when `--append` is passed. |
| `bo-forge review --config PATH --log PATH --row-id ROW_ID --decision accept\|reject\|defer [--note TEXT]` | Record one human review decision. |
| `bo-forge mark-observed --config PATH --log PATH --row-id ROW_ID --objective-value VALUE [--actual-cost VALUE]` | Mark one pending suggestion as observed. |
| `bo-forge mark-observed --config PATH --log PATH --row-id ROW_ID --objective NAME=VALUE --objective NAME=VALUE [...] [--actual-cost VALUE]` | Mark a multi-objective pending suggestion observed, optionally with realised cost when cost is configured. |
| `bo-forge plot --config PATH --log PATH --kind progress\|diagnostics\|cost-progress\|replicates\|pareto\|pareto-parallel\|hypervolume\|stage-diagnostics\|fidelity-diagnostics --output PATH` | Export one progress, diagnostics, cost-progress, replicate-summary, Pareto, Pareto-parallel, hypervolume, structured stage-diagnostics, or fidelity-diagnostics figure. |

## 🧯 CLI Error Output

Expected user-facing failures print `Error: ...` to stderr and exit with code `1`.
Most config, CSV, suggestion, and log-write errors also include a short `Hint: ...` line.

Missing required arguments use normal `argparse` behavior and exit with code `2`.

For detailed YAML and CSV fixes, see [COMMON_ERRORS.md](COMMON_ERRORS.md).
For copyable intentional failure examples, see [CLI_ERROR_EXAMPLES.md](CLI_ERROR_EXAMPLES.md).

## ⚠️ Mutation Rules

Most commands are read-only.

The commands that can change files are:

- `bo-forge init-log`: creates a new empty campaign log and refuses to overwrite existing files.
- `bo-forge suggest --append`: appends generated suggestions as `status=suggested`.
- `bo-forge review`: updates `review_status` and `review_note` for one suggested row.
- `bo-forge mark-observed`: marks one existing pending row as `status=observed`.
- `bo-forge report --output`: writes a report file.
- `bo-forge plot --output`: writes a figure file.

`bo-forge suggest --append` never marks suggestions observed. The explicit campaign rhythm remains:

> suggest → append → run experiment → mark-observed

For review-enabled campaigns, the explicit rhythm is:

> suggest → append → review → run accepted experiment → mark-observed
