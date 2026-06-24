import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.features.candidate_action_features import FEATURE_COLUMNS
from src.models import MazeState
from src.policies.decision_tree_policy import DecisionTreePolicy
from src.policies.navigation_policy import NavigationContext


def make_context() -> NavigationContext:
    return NavigationContext(
        maze_name="Test",
        step=1,
        path_depth=1,
        has_known_exit=False,
        has_known_collection_point=False,
    )


def make_state() -> MazeState:
    return MazeState.from_api(
        {
            "possibleMoveActions": [
                {
                    "direction": "Left",
                    "isStart": False,
                    "allowsExit": False,
                    "allowsScoreCollection": False,
                    "hasBeenVisited": False,
                    "numberOfVisitsToTile": 0,
                    "rewardOnDestination": 1,
                    "tagOnTile": None,
                },
                {
                    "direction": "Right",
                    "isStart": False,
                    "allowsExit": False,
                    "allowsScoreCollection": False,
                    "hasBeenVisited": False,
                    "numberOfVisitsToTile": 0,
                    "rewardOnDestination": 20,
                    "tagOnTile": None,
                },
            ],
            "canCollectScoreHere": False,
            "canExitMazeHere": False,
            "currentScoreInHand": 0,
            "currentScoreInBag": 0,
            "tagOnCurrentTile": None,
            "numberOfVisitsToTile": 1,
        }
    )


def make_training_row(reward: int) -> dict[str, float]:
    row = {column: 0.0 for column in FEATURE_COLUMNS}
    row["available_action_count"] = 2.0
    row["candidate_reward_on_destination"] = float(reward)
    return row


def test_decision_tree_policy_prefers_action_with_higher_learned_score(tmp_path):
    rows = [
        make_training_row(1),
        make_training_row(2),
        make_training_row(20),
        make_training_row(25),
    ]

    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    y = [0, 0, 1, 1]

    model = DecisionTreeClassifier(
        max_depth=2,
        random_state=42,
    )
    model.fit(X, y)

    model_path = tmp_path / "decision_tree_policy.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
        },
        model_path,
    )

    policy = DecisionTreePolicy(model_path=model_path)

    action = policy.choose_action(
        state=make_state(),
        context=make_context(),
    )

    assert action is not None
    assert action.direction == "Right"