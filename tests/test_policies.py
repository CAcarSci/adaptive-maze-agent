from src.models import MazeState
from src.policies.baseline_policy import BaselineDfsPolicy
from src.policies.navigation_policy import NavigationContext
from src.policies.reward_aware_policy import RewardAwarePolicy


def make_context() -> NavigationContext:
    return NavigationContext(
        maze_name="Test",
        step=1,
        path_depth=1,
        has_known_exit=False,
        has_known_collection_point=False,
    )


def make_state(actions: list[dict]) -> MazeState:
    return MazeState.from_api(
        {
            "possibleMoveActions": actions,
            "canCollectScoreHere": False,
            "canExitMazeHere": False,
            "currentScoreInHand": 10,
            "currentScoreInBag": 0,
            "tagOnCurrentTile": None,
            "numberOfVisitsToTile": 1,
        }
    )


def test_baseline_policy_uses_stable_direction_order():
    state = make_state(
        [
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
                "direction": "Up",
                "isStart": False,
                "allowsExit": False,
                "allowsScoreCollection": False,
                "hasBeenVisited": False,
                "numberOfVisitsToTile": 0,
                "rewardOnDestination": 1,
                "tagOnTile": None,
            },
        ]
    )

    action = BaselineDfsPolicy().choose_action(
        state=state,
        context=make_context(),
    )

    assert action is not None
    assert action.direction == "Up"


def test_reward_aware_policy_prefers_higher_reward_unvisited_action():
    state = make_state(
        [
            {
                "direction": "Right",
                "isStart": False,
                "allowsExit": False,
                "allowsScoreCollection": False,
                "hasBeenVisited": False,
                "numberOfVisitsToTile": 0,
                "rewardOnDestination": 5,
                "tagOnTile": None,
            },
            {
                "direction": "Left",
                "isStart": False,
                "allowsExit": False,
                "allowsScoreCollection": False,
                "hasBeenVisited": False,
                "numberOfVisitsToTile": 0,
                "rewardOnDestination": 20,
                "tagOnTile": None,
            },
        ]
    )

    action = RewardAwarePolicy().choose_action(
        state=state,
        context=make_context(),
    )

    assert action is not None
    assert action.direction == "Left"


def test_reward_aware_policy_ignores_visited_high_reward_action():
    state = make_state(
        [
            {
                "direction": "Right",
                "isStart": False,
                "allowsExit": False,
                "allowsScoreCollection": False,
                "hasBeenVisited": True,
                "numberOfVisitsToTile": 3,
                "rewardOnDestination": 100,
                "tagOnTile": None,
            },
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
        ]
    )

    action = RewardAwarePolicy().choose_action(
        state=state,
        context=make_context(),
    )

    assert action is not None
    assert action.direction == "Left"


def test_reward_aware_policy_returns_none_when_no_unvisited_actions_exist():
    state = make_state(
        [
            {
                "direction": "Right",
                "isStart": False,
                "allowsExit": False,
                "allowsScoreCollection": False,
                "hasBeenVisited": True,
                "numberOfVisitsToTile": 1,
                "rewardOnDestination": 10,
                "tagOnTile": None,
            }
        ]
    )

    action = RewardAwarePolicy().choose_action(
        state=state,
        context=make_context(),
    )

    assert action is None