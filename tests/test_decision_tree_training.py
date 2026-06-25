import pandas as pd
import pytest

from src.training.train_decision_tree import (
    TRAINING_BOT_NAMES,
    filter_training_telemetry,
)


def test_filter_training_telemetry_excludes_decision_tree_rows():
    df = pd.DataFrame(
        [
            {"bot_name": "baseline_dfs", "value": 1},
            {"bot_name": "reward_aware", "value": 2},
            {"bot_name": "decision_tree", "value": 3},
        ]
    )

    filtered_df = filter_training_telemetry(df)

    assert set(filtered_df["bot_name"].unique()) == set(TRAINING_BOT_NAMES)
    assert "decision_tree" not in set(filtered_df["bot_name"].unique())
    assert len(filtered_df) == 2


def test_filter_training_telemetry_raises_when_no_supported_training_rows_exist():
    df = pd.DataFrame(
        [
            {"bot_name": "decision_tree", "value": 1},
        ]
    )

    with pytest.raises(ValueError, match="No usable training telemetry found"):
        filter_training_telemetry(df)


def test_filter_training_telemetry_requires_bot_name_column():
    df = pd.DataFrame(
        [
            {"value": 1},
        ]
    )

    with pytest.raises(ValueError, match="bot_name"):
        filter_training_telemetry(df)