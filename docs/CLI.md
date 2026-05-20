# 💻 CLI Workflow

The `bo-forge` command wraps the same `CampaignSession` workflow used in notebooks.
It does not add new BO behaviour; it makes validation, suggestions, reporting, and plotting usable from the terminal.

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
  --config configs/simple_2d_maximise_logei.yaml \
  --log examples/my_new_campaign_log.csv
```

Validate a campaign log:

```bash
bo-forge validate \
  --config configs/simple_2d_maximise_logei.yaml \
  --log examples/simple_2d_maximise_logei_campaign_log.csv
```

Inspect state:

```bash
bo-forge status \
  --config configs/simple_2d_maximise_logei.yaml \
  --log examples/simple_2d_maximise_logei_campaign_log.csv

bo-forge summary \
  --config configs/simple_2d_maximise_logei.yaml \
  --log examples/simple_2d_maximise_logei_campaign_log.csv

bo-forge next-action \
  --config configs/simple_2d_maximise_logei.yaml \
  --log examples/simple_2d_maximise_logei_campaign_log.csv
```

Generate suggestions without changing the campaign log:

```bash
bo-forge suggest \
  --config configs/simple_2d_maximise_logei.yaml \
  --log examples/simple_2d_maximise_logei_campaign_log.csv \
  --batch-size 1
```

Generate suggestions, save a suggestions CSV, and append the same suggestions to the canonical log:

```bash
bo-forge suggest \
  --config configs/simple_2d_maximise_logei.yaml \
  --log examples/simple_2d_maximise_logei_working_log.csv \
  --output examples/simple_2d_maximise_logei_latest_suggestions.csv \
  --append
```

After running the experiment, record the result:

```bash
bo-forge mark-observed \
  --config configs/simple_2d_maximise_logei.yaml \
  --log examples/simple_2d_maximise_logei_working_log.csv \
  --row-id ROW_ID_FROM_SUGGESTIONS \
  --objective-value 1.95
```

## 📄 Reports And Plots

Print a plain-text campaign report:

```bash
bo-forge report \
  --config configs/simple_2d_maximise_logei.yaml \
  --log examples/simple_2d_maximise_logei_working_log.csv
```

Export the same deterministic report format:

```bash
bo-forge report \
  --config configs/simple_2d_maximise_logei.yaml \
  --log examples/simple_2d_maximise_logei_working_log.csv \
  --output reports/latest_campaign_report.txt
```

Export one figure per command:

```bash
bo-forge plot \
  --config configs/simple_2d_maximise_logei.yaml \
  --log examples/simple_2d_maximise_logei_working_log.csv \
  --kind progress \
  --output reports/progress.png

bo-forge plot \
  --config configs/simple_2d_maximise_logei.yaml \
  --log examples/simple_2d_maximise_logei_working_log.csv \
  --kind diagnostics \
  --output reports/diagnostics.png
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
| `bo-forge report --config PATH --log PATH [--output PATH]` | Print or export a deterministic campaign report. |
| `bo-forge suggest --config PATH --log PATH [--batch-size N] [--output PATH] [--append]` | Generate suggestions; append only when `--append` is passed. |
| `bo-forge mark-observed --config PATH --log PATH --row-id ROW_ID --objective-value VALUE` | Mark one pending suggestion as observed. |
| `bo-forge plot --config PATH --log PATH --kind progress\|diagnostics --output PATH` | Export one progress or diagnostics figure. |

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
- `bo-forge mark-observed`: marks one existing pending row as `status=observed`.
- `bo-forge report --output`: writes a report file.
- `bo-forge plot --output`: writes a figure file.

`bo-forge suggest --append` never marks suggestions observed. The explicit campaign rhythm remains:

> suggest → append → run experiment → mark-observed
