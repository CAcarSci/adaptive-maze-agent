from src.models import (
    DIRECTION_ORDER,
    OPPOSITE_DIRECTION,
    MazeState,
    MoveAction,
)


def test_move_action_from_api_maps_expected_fields():
    raw_action = {
        "direction": "Right",
        "isStart": False,
        "allowsExit": False,
        "allowsScoreCollection": True,
        "hasBeenVisited": False,
        "numberOfVisitsToTile": 0,
        "rewardOnDestination": 10,
        "tagOnTile": 123,
    }

    action = MoveAction.from_api(raw_action)

    assert action.direction == "Right"
    assert action.is_start is False
    assert action.allows_exit is False
    assert action.allows_score_collection is True
    assert action.has_been_visited is False
    assert action.number_of_visits_to_tile == 0
    assert action.reward_on_destination == 10
    assert action.tag_on_tile == 123


def test_maze_state_from_api_maps_possible_move_actions():
    raw_state = {
        "possibleMoveActions": [
            {
                "direction": "Right",
                "isStart": False,
                "allowsExit": False,
                "allowsScoreCollection": False,
                "hasBeenVisited": False,
                "numberOfVisitsToTile": 0,
                "rewardOnDestination": 1,
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
        "currentScoreInHand": 0,
        "currentScoreInBag": 0,
        "tagOnCurrentTile": None,
        "numberOfVisitsToTile": 1,
    }

    state = MazeState.from_api(raw_state)

    assert state.can_collect_score_here is False
    assert state.can_exit_maze_here is False
    assert state.current_score_in_hand == 0
    assert state.current_score_in_bag == 0
    assert state.tag_on_current_tile is None
    assert state.number_of_visits_to_tile == 1

    assert len(state.possible_move_actions) == 2
    assert state.possible_move_actions[0].direction == "Right"
    assert state.possible_move_actions[0].reward_on_destination == 1
    assert state.possible_move_actions[1].direction == "Left"
    assert state.possible_move_actions[1].allows_exit is True


def test_direction_order_is_stable_for_baseline_policy():
    assert DIRECTION_ORDER == ["Up", "Right", "Down", "Left"]


def test_opposite_direction_mapping_is_symmetric():
    for direction, opposite in OPPOSITE_DIRECTION.items():
        assert OPPOSITE_DIRECTION[opposite] == direction