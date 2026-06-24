from pathlib import Path

import joblib
import pandas as pd

from src.features.candidate_action_features import (
    FEATURE_COLUMNS,
    candidate_action_to_feature_row,
)
from src.models import MazeState, MoveAction
from src.policies.navigation_policy import NavigationContext


class DecisionTreePolicy:
    """
    Lightweight trained ML navigation policy.

    The policy loads a Decision Tree model trained on candidate-action telemetry.
    At runtime it scores each unvisited candidate action and chooses the action
    with the highest predicted probability of being preferred.
    """

    name = "decision_tree"

    def __init__(
        self,
        model_path: str | Path = "models/decision_tree_policy.joblib",
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Decision Tree model not found: {self.model_path}. "
                "Train it first with `python -m src.training.train_decision_tree`."
            )

        artifact = joblib.load(self.model_path)

        self.model = artifact["model"]
        self.feature_columns = artifact.get("feature_columns", FEATURE_COLUMNS)

    def choose_action(
        self,
        state: MazeState,
        context: NavigationContext,
    ) -> MoveAction | None:
        candidate_actions = [
            action
            for action in state.possible_move_actions
            if not action.has_been_visited
        ]

        if not candidate_actions:
            return None

        feature_rows = [
            candidate_action_to_feature_row(
                state=state,
                action=action,
                context=context,
            )
            for action in candidate_actions
        ]

        X = pd.DataFrame(feature_rows)
        X = X.reindex(columns=self.feature_columns, fill_value=0.0)

        probabilities = self._predict_preferred_probability(X)

        best_index = max(
            range(len(candidate_actions)),
            key=lambda index: (
                probabilities[index],
                candidate_actions[index].reward_on_destination,
                -candidate_actions[index].number_of_visits_to_tile,
            ),
        )

        return candidate_actions[best_index]

    def _predict_preferred_probability(self, X: pd.DataFrame) -> list[float]:
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)

            if 1 in self.model.classes_:
                preferred_index = list(self.model.classes_).index(1)
                return probabilities[:, preferred_index].tolist()

        predictions = self.model.predict(X)
        return [float(prediction) for prediction in predictions]