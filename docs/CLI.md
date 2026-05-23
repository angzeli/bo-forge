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

Cost-aware and review-enabled campaigns use the same rhythm with extra review and cost commands. Inspect the current budget state:

```bash
bo-forge cost-summary \
  --config configs/07_cost_aware_human_review_logei.yaml \
  --log examples/07_cost_aware_human_review_working_log.csv
```

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
| `bo-forge cost-summary --config PATH --log PATH` | Print cost, reserved-cost, budget, and best-observed-objective fields. |
| `bo-forge report --config PATH --log PATH [--output PATH]` | Print or export a deterministic campaign report. |
| `bo-forge suggest --config PATH --log PATH [--batch-size N] [--output PATH] [--append]` | Generate suggestions; append only when `--append` is passed. |
| `bo-forge review --config PATH --log PATH --row-id ROW_ID --decision accept\|reject\|defer [--note TEXT]` | Record one human review decision. |
| `bo-forge mark-observed --config PATH --log PATH --row-id ROW_ID --objective-value VALUE [--actual-cost VALUE]` | Mark one pending suggestion as observed. |
| `bo-forge plot --config PATH --log PATH --kind progress\|diagnostics\|cost-progress --output PATH` | Export one progress, diagnostics, or cost-progress figure. |

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
