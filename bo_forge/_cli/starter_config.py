"""Starter campaign config written by ``bo-forge init-config``.

The wheel does not ship the repository's example ``configs/``, so this text is the
starting point for an installed user. It is a string rather than package data so the
wheel's file list is unchanged.
"""

from __future__ import annotations

EXAMPLE_CONFIGS_URL = "https://github.com/angzeli/bo-forge/tree/main/configs"

STARTER_CONFIG_YAML = f"""\
# BO Forge starter campaign config.
# Edit the names, bounds, and choices to match your experiment, then create the
# campaign log with: bo-forge init-log --config THIS_FILE --log NEW_LOG.csv
# More examples (multi-objective, cost, context, fidelity): {EXAMPLE_CONFIGS_URL}

campaign_name: my_campaign

# The measured result to optimise; direction is maximize or minimize.
objective:
  name: yield_score
  direction: maximize

# What you set in each experiment.
variables:
  - name: temperature
    type: continuous
    lower: 20.0
    upper: 80.0

  - name: reaction_time
    type: integer
    lower: 10
    upper: 60

  - name: solvent
    type: categorical
    values: [MeCN, EtOH, Water]

# Optional: exclude combinations you cannot or will not run.
# constraints:
#   - name: water_needs_longer_time
#     expression: "solvent != 'Water' or reaction_time >= 35"

bo:
  batch_size: 2            # suggestions per round
  initial_design_size: 6   # space-filling experiments before the model is used
  acquisition: log_ei
  random_seed: 0
"""
