import csv

from src.data.telemetry_logger import TelemetryLogger
from src.models import MazeState


def test_telemetry_logger_writes_candidate_action_rows(tmp_path):
    output_path = tmp_path / "action_logs.csv"

    state = MazeState.from_api(
        {
            "possibleMoveActions": [
                {
                    "direction": "Right",
                    "isStart": False,
                    "allowsExit": False,
                    "allowsScoreCollection": False,
                    "hasBeenVisited": False,
                    "numberOfVisitsToTile": 0,
                    "rewardOnDestination": 10,
                    "tagOnTile": None,
                },
                {
                    "direction": "Left",
                    "isStart": True,
                    "allowsExit": True,
                    "allowsScoreCollection": False,
                    "hasBeenVisited": True,
                    "numberOfVisitsToTile": 1,
                    "rewardOnDestination": 0,
                    "tagOnTile": None,
                },
            ],
            "canCollectScoreHere": False,
            "canExitMazeHere": False,
            "currentScoreInHand": 10,
            "currentScoreInBag": 0,
            "tagOnCurrentTile": None,
            "numberOfVisitsToTile": 1,
        }
    )

    logger = TelemetryLogger(
        output_path=str(output_path),
        run_id="test-run-id",
    )

    logger.log_decision(
        maze_name="Test",
        bot_name="baseline_dfs",
        phase="exploration",
        step=1,
        decision_type="explore",
        state=state,
        chosen_direction="Right",
        path_depth=2,
    )

    with output_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2

    right_row = rows[0]
    left_row = rows[1]

    assert right_row["run_id"] == "test-run-id"
    assert right_row["maze_name"] == "Test"
    assert right_row["bot_name"] == "baseline_dfs"
    assert right_row["step"] == "1"
    assert right_row["decision_type"] == "explore"
    assert right_row["chosen_direction"] == "Right"
    assert right_row["candidate_direction"] == "Right"
    assert right_row["is_chosen"] == "True"
    assert right_row["candidate_reward_on_destination"] == "10"

    assert left_row["candidate_direction"] == "Left"
    assert left_row["is_chosen"] == "False"
    assert left_row["candidate_allows_exit"] == "True"