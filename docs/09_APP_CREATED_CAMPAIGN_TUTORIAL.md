# 🧭 09 App-Created Campaign Tutorial

This tutorial shows how to create a new BO Forge campaign from inside the Streamlit app.

The example campaign is named `09_app_created_practical_catalyst`. It deliberately combines ideas from the previous example campaigns:

- mixed variables from the mixed-variable examples;
- feasibility constraints from the constrained campaign;
- deterministic cost and human review from the cost-aware campaign;
- explicit replicate metadata from the replicate-aware campaign.

The app writes both files:

- config: `configs/09_app_created_practical_catalyst.yaml`
- log: `examples/09_app_created_practical_catalyst_campaign_log.csv`

## ▶️ Start The App

From the repository root:

```bash
bo-forge-app
```

From a source checkout, `./.venv/bin/python -m streamlit run bo_forge_app/streamlit_app.py` also works.

In the compact campaign source bar, set `Campaign file action` to `Create Campaign`.

## 🧾 Fill The Structured Fields

Use these values in the structured form:

| Field | Value |
| --- | --- |
| New campaign name | `09_app_created_practical_catalyst` |
| New YAML config output path | `configs/09_app_created_practical_catalyst.yaml` |
| New CSV log output path | `examples/09_app_created_practical_catalyst_campaign_log.csv` |
| Campaign kind | `Single-objective` |
| Objective name | `yield_score` |
| Objective direction | `maximize` |
| Batch size | `2` |
| Initial design size | `6` |
| Initial design method | `sobol` |
| Random seed | `29` |
| Number of variables | `4` |

Define the variables as:

| Variable | Type | Values |
| --- | --- | --- |
| `catalyst_loading` | `continuous` | lower `0.02`, upper `0.20` |
| `reaction_time` | `integer` | lower `10`, upper `60` |
| `base_equivalents` | `discrete` | `0.1, 0.2, 0.5, 1.0` |
| `solvent` | `categorical` | `MeCN, EtOH, Water` |

For discrete values, use comma-separated numbers. For categorical values, use comma-separated labels.

## ✏️ Edit The YAML Preview

After filling the structured fields, click `Update YAML preview from form`.

Then replace the YAML preview with this full config:

```yaml
campaign_name: 09_app_created_practical_catalyst
objective:
  name: yield_score
  direction: maximize

variables:
  - name: catalyst_loading
    type: continuous
    lower: 0.02
    upper: 0.20

  - name: reaction_time
    type: integer
    lower: 10
    upper: 60

  - name: base_equivalents
    type: discrete
    values: [0.1, 0.2, 0.5, 1.0]

  - name: solvent
    type: categorical
    values: [MeCN, EtOH, Water]

constraints:
  - name: no_water_high_base
    expression: "not (solvent == 'Water' and base_equivalents >= 0.5)"

  - name: water_needs_longer_time
    expression: "solvent != 'Water' or reaction_time >= 35"

cost:
  expression: "1.0 + 0.04 * reaction_time + 2.0 * (solvent == 'Water')"
  weight: 0.5
  budget: 60.0
  candidate_pool_size: 128
  top_k: 24

review:
  enabled: true

replicates:
  enabled: true

bo:
  batch_size: 2
  initial_design_size: 6
  acquisition: log_ei
  initial_design_method: sobol
  random_seed: 29
  raw_samples: 32
  num_restarts: 3
  mc_samples: 32
  min_normalized_distance: 0.03
```

This config must pass BO Forge validation before any files are written.

## 🧪 Create And Validate

Click `Create campaign`.

Expected result:

- the YAML config file is created;
- the empty canonical CSV log is created;
- the campaign is loaded immediately;
- the source bar and `Overview` panel show the campaign as valid;
- the log contains headers only and no observed rows yet.

The canonical CSV columns will include review, replicate, cost, and utility fields because those sections are enabled in the YAML.

## 🔁 Run The First Campaign Loop

Use the normal app workflow:

1. Open `Suggest`.
2. Click `Generate suggestions (dry run)`.
3. Review the staged suggestions and `Suggestion Quality`.
4. Click `Append staged suggestions`.
5. Open `Resolve`.
6. Review pending suggestions.
7. Accept one suggested row.
8. Run that experiment outside BO Forge.
9. Mark the accepted row observed and enter `yield_score`.
10. Reload and repeat.

Because review is enabled, only accepted suggestions should be marked observed. Because cost is enabled, you can optionally record actual cost when marking a row observed. Because replicates are enabled, generated exploration suggestions start as replicate index `0` for a new replicate group.

## 📌 What This Example Demonstrates

This `09` campaign is useful as a compact practical app demo:

- the app can create a campaign from scratch;
- `Campaign kind` selects single-objective, multi-objective, multi-fidelity
  qMFKG, or Contextual LogEI creation paths;
- structured fields handle the basic config;
- advanced YAML adds constraints, cost, review, and replicates;
- BO Forge still writes a strict canonical CSV log;
- the app remains a wrapper around the same backend validation and session workflow.
