# 🧯 CLI Error Examples

This page shows intentional CLI failures and the kind of error output BO Forge should produce.

Use these examples as a quick way to understand `Error: ...` and `Hint: ...` messages. The commands use `python -m bo_forge` so they run through the active Python environment; `bo-forge` is equivalent in a normal terminal.

## 🧪 Setup A Temporary Demo Copy

Run examples against temporary files, not committed seed logs:

```bash
mkdir -p /tmp/bo_forge_error_demo
cp configs/01_simple_2d_maximise_logei.yaml /tmp/bo_forge_error_demo/config.yaml
cp examples/01_simple_2d_maximise_logei_campaign_log.csv /tmp/bo_forge_error_demo/campaign.csv
```

## ⚙️ Missing Config Path

Command:

```bash
python -m bo_forge validate \
  --config /tmp/bo_forge_error_demo/missing.yaml \
  --log /tmp/bo_forge_error_demo/campaign.csv
```

Expected shape:

```text
Error: Could not read config file ...
Hint: Check the YAML config path and campaign settings.
```

Fix: check the config path, or create the YAML file before running the command.

## 🧾 CSV Value Outside Bounds

Create an invalid temporary CSV:

```bash
python - <<'PY'
from pathlib import Path

import pandas as pd

path = Path("/tmp/bo_forge_error_demo/campaign.csv")
df = pd.read_csv(path)
df.loc[0, "precursor_ratio"] = 2.0
df.to_csv(path, index=False)
PY
```

Command:

```bash
python -m bo_forge validate \
  --config /tmp/bo_forge_error_demo/config.yaml \
  --log /tmp/bo_forge_error_demo/campaign.csv
```

Expected shape:

```text
Error: ... outside bounds ...
Hint: Check the CSV schema, statuses, objective values, and variable bounds.
```

Fix: correct the CSV value or update the YAML bounds if the campaign definition was wrong.

## 🧭 Pending Suggestions Block New Suggestions

Reset the temporary CSV, append one suggestion, then ask for another:

```bash
cp examples/01_simple_2d_maximise_logei_campaign_log.csv /tmp/bo_forge_error_demo/campaign.csv

python -m bo_forge suggest \
  --config /tmp/bo_forge_error_demo/config.yaml \
  --log /tmp/bo_forge_error_demo/campaign.csv \
  --append

python -m bo_forge suggest \
  --config /tmp/bo_forge_error_demo/config.yaml \
  --log /tmp/bo_forge_error_demo/campaign.csv
```

Expected shape:

```text
Error: Cannot generate new suggestions while unresolved status='suggested' rows exist ...
Hint: Resolve pending suggestions or review the campaign state before requesting new suggestions.
```

Fix: run the suggested experiment and mark that row observed before requesting more suggestions.

## 🧬 Unknown Row ID In `mark-observed`

Command:

```bash
python -m bo_forge mark-observed \
  --config /tmp/bo_forge_error_demo/config.yaml \
  --log /tmp/bo_forge_error_demo/campaign.csv \
  --row-id not_a_real_row \
  --objective-value 1.23
```

Expected shape:

```text
Error: Cannot mark row 'not_a_real_row' observed because row_id was not found.
Hint: Check the row_id, pending status, campaign log path, and file permissions.
```

Fix: get the exact `row_id` from `python -m bo_forge next-action`, `summary`, or the campaign CSV.

## 📄 Invalid Output Path

Create a file where a directory is expected:

```bash
printf "not a directory" > /tmp/bo_forge_error_demo/not_a_dir
```

Command:

```bash
python -m bo_forge report \
  --config /tmp/bo_forge_error_demo/config.yaml \
  --log /tmp/bo_forge_error_demo/campaign.csv \
  --output /tmp/bo_forge_error_demo/not_a_dir/report.txt
```

Expected shape:

```text
Error: Could not write campaign report '/tmp/bo_forge_error_demo/not_a_dir/report.txt': ...
```

Fix: choose an output path whose parent is a directory, for example `reports/latest_campaign_report.txt`.

## 🛠️ More Fixes

For detailed YAML and CSV troubleshooting, see [COMMON_ERRORS.md](COMMON_ERRORS.md).
