"""Bounded JSON previews against a clean installed artifact, outside the checkout."""

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def _preview(entry, campaign, source, environment):
    protected = {p: p.read_bytes() for p in campaign.log_path.parent.iterdir() if p.is_file()}
    result = subprocess.run(
        [*entry, "suggest", "--config", str(campaign.config_path),
         "--log", str(campaign.log_path), "--batch-size", "1",
         "--require-provenance", "--format=json"],
        env=environment, text=True, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.endswith("\n") and len(result.stdout.splitlines()) == 1

    def invalid_constant(value):
        raise AssertionError(f"Non-JSON constant: {value}")

    payload = json.loads(result.stdout, parse_constant=invalid_constant)
    import bo_forge

    assert payload["bo_forge_version"] == bo_forge.__version__
    assert payload["schema_version"] == 1 and payload["ok"] is True
    assert payload["command"] == "suggest" and payload["error"] is None
    assert payload["data"]["columns"] == list(campaign.df.columns)
    row, = payload["data"]["records"]
    assert row["status"] == "suggested" and row["source"] == source
    assert row["score"] == "" and 0 <= row["x"] <= 1
    for field in ("predicted_mean", "predicted_std", "acquisition"):
        if source == "sobol":
            assert row[field] == ""
        else:
            assert math.isfinite(row[field]), row
    assert protected == {
        p: p.read_bytes() for p in campaign.log_path.parent.iterdir() if p.is_file()
    }
    print(f"Installed JSON {source} preview passed: {' '.join(entry)}")


def main():
    import bo_forge
    from bo_forge import CampaignSession

    source_root = Path(os.environ["SOURCE_ROOT"]).resolve()
    assert not Path.cwd().resolve().is_relative_to(source_root)
    assert not Path(bo_forge.__file__).resolve().is_relative_to(source_root)
    assert Path(bo_forge.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert "include-system-site-packages = false" in (
        Path(sys.prefix) / "pyvenv.cfg"
    ).read_text().lower()
    assert importlib.util.find_spec("streamlit") is None
    assert importlib.util.find_spec("fastapi") is None
    environment = {
        **os.environ, "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    entries = ([sys.executable, "-m", "bo_forge"],
               [str(Path(sys.executable).parent / "bo-forge")])
    with TemporaryDirectory(prefix="json-preview-", dir=Path.cwd()) as directory:
        config, log = Path(directory) / "campaign.yaml", Path(directory) / "campaign.csv"
        config.write_text(
            "campaign_name: installed_json_preview\n"
            "objective: {name: score, direction: maximize}\n"
            "variables: [{name: x, type: continuous, lower: 0, upper: 1}]\n"
            "bo: {acquisition: log_ei, initial_design_size: 4, batch_size: 1, "
            "random_seed: 0, raw_samples: 16, num_restarts: 1, mc_samples: 16}\n",
            encoding="utf-8",
        )
        campaign = CampaignSession.initialize(config, log)
        for entry in entries:
            _preview(entry, campaign, "sobol", environment)
        initial = campaign.suggest_next(batch_size=4)
        campaign.append_suggestions(initial)
        for row in initial.itertuples():
            campaign.mark_observed(row.row_id, 2 * row.x + 1)
        for entry in entries:
            _preview(entry, campaign, "log_ei", environment)


if __name__ == "__main__":
    main()
