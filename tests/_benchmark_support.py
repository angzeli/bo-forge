"""Small controlled strategies for repository benchmark acceptance."""

import copy
from pathlib import Path

import yaml

from benchmarks.runner import _finalize_status
from benchmarks.spec import load_spec, schedule
from benchmarks.storage import write_json
from benchmarks.trial import _external_row, run_trial

ROOT = Path(__file__).resolve().parents[1]


def smoke_spec():
    return copy.deepcopy(load_spec(ROOT / "benchmarks/specs/smoke.yaml"))


def spec_file(tmp_path, spec=None):
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(spec or smoke_spec()))
    return path


def prepare_trial(tmp_path, strategy="bo", mode="deterministic"):
    trial = next(item for item in schedule(smoke_spec())
                 if item["strategy"] == strategy and item["mode"] == mode)
    directory = tmp_path / trial["trial_id"]
    directory.mkdir()
    write_json(directory / "trial.json", trial)
    write_json(directory / "status.json", {"status": "pending", "completed_evaluations": 0})
    return directory, trial


def controlled_suggest(campaign):
    fraction = .12 + .03 * len(campaign.df)
    point = [variable.lower + fraction * (variable.upper - variable.lower)
             for variable in campaign.config.variables]
    return _external_row(campaign, point, len(campaign.df), campaign.config.bo.acquisition)


def controlled_execute(directory, timeout):
    (directory / "worker.log").write_text("Controlled fitter; no BoTorch fit executed.\n")
    try:
        run_trial(directory, suggest=controlled_suggest)
    except Exception as exc:
        # Test process boundary mirrors the runner's preservation of failed trial evidence.
        _finalize_status(directory, "failed", str(exc), 1.0)
    else:
        _finalize_status(directory, None, "", 1.0)
