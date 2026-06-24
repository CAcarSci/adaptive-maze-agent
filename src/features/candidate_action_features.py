from src.models import MazeState, MoveAction
from src.policies.navigation_policy import NavigationContext


FEATURE_COLUMNS = [
    "current_score_in_hand",
    "current_score_in_bag",
    "can_collect_score_here",
    "can_exit_maze_here",
    "path_depth",
    "available_action_count",
    "candidate_reward_on_destination",
    "candidate_has_been_visited",
    "candidate_visit_count",
    "candidate_allows_exit",
    "candidate_allows_score_collection",
    "candidate_is_start",
]


def bool_to_int(value: object) -> int:
    if value in {True, "True", "true", "1", 1}:
        return 1

    return 0


def candidate_action_to_feature_row(
    *,
    state: MazeState,
    action: MoveAction,
    context: NavigationContext,
) -> dict[str, float]:
    """
    Converts one candidate action into a model-ready feature row.

    The same feature definition is used during training and inference so the
    Decision Tree policy receives the same input shape in both cases.
    """

    return {
        "current_score_in_hand": float(state.current_score_in_hand),
        "current_score_in_bag": float(state.current_score_in_bag),
        "can_collect_score_here": float(bool_to_int(state.can_collect_score_here)),
        "can_exit_maze_here": float(bool_to_int(state.can_exit_maze_here)),
        "path_depth": float(context.path_depth),
        "available_action_count": float(len(state.possible_move_actions)),
        "candidate_reward_on_destination": float(action.reward_on_destination),
        "candidate_has_been_visited": float(bool_to_int(action.has_been_visited)),
        "candidate_visit_count": float(action.number_of_visits_to_tile),
        "candidate_allows_exit": float(bool_to_int(action.allows_exit)),
        "candidate_allows_score_collection": float(
            bool_to_int(action.allows_score_collection)
        ),
        "candidate_is_start": float(bool_to_int(action.is_start)),
    }


def prepare_training_dataframe(df):
    """
    Creates a supervised learning dataset from telemetry.

    The telemetry does not directly contain human labels. Therefore we create a
    weakly supervised target using a transparent preference score.

    For each decision step, the candidate action with the highest preference
    score becomes the positive example.
    """

    candidate_df = df[df["candidate_direction"].notna()].copy()

    for column in FEATURE_COLUMNS:
        if column not in candidate_df.columns:
            raise ValueError(f"Missing required feature column: {column}")

    candidate_df["decision_id"] = (
        candidate_df["run_id"].astype(str)
        + "::"
        + candidate_df["step"].astype(str)
    )

    candidate_df["candidate_count_in_decision"] = (
        candidate_df.groupby("decision_id")["candidate_direction"]
        .transform("count")
    )

    candidate_df = candidate_df[
        candidate_df["candidate_count_in_decision"] >= 2
    ].copy()

    for column in FEATURE_COLUMNS:
        candidate_df[column] = candidate_df[column].apply(bool_to_int)

    numeric_columns = [
        "current_score_in_hand",
        "current_score_in_bag",
        "path_depth",
        "available_action_count",
        "candidate_reward_on_destination",
        "candidate_visit_count",
    ]

    for column in numeric_columns:
        candidate_df[column] = candidate_df[column].astype(float)

    candidate_df["weak_preference_score"] = (
        (1 - candidate_df["candidate_has_been_visited"]) * 100.0
        + candidate_df["candidate_reward_on_destination"] * 1.0
        + candidate_df["candidate_allows_score_collection"] * 35.0
        + (
            candidate_df["candidate_allows_score_collection"]
            * (candidate_df["current_score_in_hand"] > 0).astype(int)
            * 35.0
        )
        + candidate_df["candidate_allows_exit"] * 10.0
        - candidate_df["candidate_has_been_visited"]
        * candidate_df["candidate_visit_count"]
        * 25.0
        - candidate_df["candidate_is_start"] * 5.0
    )

    max_score_per_decision = candidate_df.groupby("decision_id")[
        "weak_preference_score"
    ].transform("max")

    candidate_df["target_preferred"] = (
        candidate_df["weak_preference_score"] == max_score_per_decision
    ).astype(int)

    return candidate_df