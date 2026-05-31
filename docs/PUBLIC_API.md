# 📦 BO Forge Public API

This page lists the stable imports supported from the top-level `bo_forge` package in v1.1.1.

Implementation modules such as `bo_forge.transforms`, `bo_forge.models`, and `bo_forge.diagnostics` remain importable for development, but their private helpers are not part of the stable public surface.

## ✅ Public Package Exports

These names are supported imports from `bo_forge`:

- `BOConfig`
- `BOForgeError`
- `CampaignConfig`
- `CampaignSession`
- `ConfigError`
- `ConstraintConfig`
- `CostConfig`
- `LogValidationError`
- `LogWriteError`
- `ObjectiveConfig`
- `ReplicateConfig`
- `ReviewConfig`
- `SuggestionError`
- `VariableConfig`
- `__version__`
- `append_suggestions`
- `aggregate_observed_replicates`
- `best_replicate_group`
- `evaluate_cost`
- `get_observed_data`
- `hypervolume`
- `hypervolume_progress`
- `load_campaign_log`
- `mark_observed`
- `pareto_front`
- `pareto_summary`
- `review_suggestion`
- `replicate_summary`
- `suggest_next`
- `suggestion_quality_summary`
- `validate_campaign_data`

## 🧪 Example

```python
from bo_forge import CampaignConfig, CampaignSession, suggest_next

config = CampaignConfig.from_yaml("configs/01_simple_2d_maximise_logei.yaml")
campaign = CampaignSession.from_files(
    config_path="configs/01_simple_2d_maximise_logei.yaml",
    log_path="examples/01_simple_2d_maximise_logei_campaign_log.csv",
)
suggestions = suggest_next(config, campaign.df)
```

## 🚧 Not Public API

The following are intentionally not guaranteed as stable v1.0 public APIs:

- private functions beginning with `_`;
- Streamlit app helper internals;
- matplotlib styling internals;
- exact text formatting of reports beyond documented sections;
- implementation details of latent transforms and acquisition optimisation.
