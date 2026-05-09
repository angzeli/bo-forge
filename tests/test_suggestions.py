import pandas as pd
import pytest

from bo_forge.config import BOConfig, CampaignConfig, ObjectiveConfig, VariableConfig
from bo_forge.errors import SuggestionError
from bo_forge.io import empty_campaign_log
from bo_forge.logs import append_suggestions, load_campaign_log, mark_observed
from bo_forge.suggestions import suggest_next


def config(batch_size: int = 2, initial_design_size: int = 3) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="test",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("temperature", "continuous", 300.0, 800.0),
        ),
        bo=BOConfig(
            batch_size=batch_size,
            initial_design_size=initial_design_size,
            random_seed=3,
            raw_samples=16,
            num_restarts=2,
            mc_samples=16,
        ),
    )


def observed_log(cfg: CampaignConfig) -> pd.DataFrame:
    rows = []
    for index, (x_value, temperature, activity) in enumerate(
        [
            (0.1, 350.0, 0.5),
            (0.3, 500.0, 1.1),
            (0.6, 650.0, 1.8),
            (0.9, 780.0, 1.2),
        ]
    ):
        rows.append(
            {
                "row_id": f"obs_{index}",
                "iteration": index,
                "status": "observed",
                "source": "manual",
                "x": x_value,
                "temperature": temperature,
                "activity": activity,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        )
    return pd.DataFrame(rows)


def test_suggest_next_returns_sobol_initial_suggestions() -> None:
    cfg = config(batch_size=2, initial_design_size=3)
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 2
    assert set(suggestions["status"]) == {"suggested"}
    assert set(suggestions["source"]) == {"sobol"}
    assert suggestions["x"].astype(float).between(0.0, 1.0).all()
    assert suggestions["temperature"].astype(float).between(300.0, 800.0).all()
    assert suggestions["activity"].astype(str).eq("").all()


def test_suggest_next_refuses_pending_suggestions() -> None:
    cfg = config(batch_size=1, initial_design_size=3)
    df = empty_campaign_log(cfg)
    pending = suggest_next(cfg, df)

    with pytest.raises(SuggestionError, match="unresolved status='suggested'"):
        suggest_next(cfg, pending)


def test_suggest_next_returns_model_based_single_suggestion() -> None:
    cfg = config(batch_size=1, initial_design_size=3)
    df = observed_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 1
    assert suggestions.loc[0, "status"] == "suggested"
    assert suggestions.loc[0, "source"] == "log_ei"
    assert float(suggestions.loc[0, "predicted_std"]) >= 0.0
    assert float(suggestions.loc[0, "x"]) >= 0.0
    assert float(suggestions.loc[0, "x"]) <= 1.0


def test_suggest_next_returns_model_based_batch_suggestions() -> None:
    cfg = config(batch_size=2, initial_design_size=3)
    df = observed_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 2
    assert set(suggestions["source"]) == {"qlog_ei"}
    assert suggestions["temperature"].astype(float).between(300.0, 800.0).all()


def test_one_by_one_sobol_suggestions_do_not_repeat_after_csv_round_trip(tmp_path) -> None:
    cfg = config(batch_size=1, initial_design_size=4)
    log_path = tmp_path / "campaign.csv"
    df = empty_campaign_log(cfg)
    seen: set[tuple[float, float]] = set()

    for index in range(4):
        suggestions = suggest_next(cfg, df, batch_size=1)
        candidate = (
            float(suggestions.loc[0, "x"]),
            float(suggestions.loc[0, "temperature"]),
        )
        assert candidate not in seen
        seen.add(candidate)

        append_suggestions(log_path, suggestions)
        mark_observed(log_path, str(suggestions.loc[0, "row_id"]), float(index))
        df = load_campaign_log(log_path, cfg)
