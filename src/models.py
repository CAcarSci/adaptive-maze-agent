from dataclasses import dataclass
from typing import Any


DIRECTION_ORDER = ["Up", "Right", "Down", "Left"]

OPPOSITE_DIRECTION = {
    "Up": "Down",
    "Right": "Left",
    "Down": "Up",
    "Left": "Right",
}


@dataclass(frozen=True)
class MoveAction:
    direction: str
    is_start: bool
    allows_exit: bool
    allows_score_collection: bool
    has_been_visited: bool
    number_of_visits_to_tile: int
    reward_on_destination: int
    tag_on_tile: int | None

    @staticmethod
    def from_api(data: dict[str, Any]) -> "MoveAction":
        return MoveAction(
            direction=data["direction"],
            is_start=data["isStart"],
            allows_exit=data["allowsExit"],
            allows_score_collection=data["allowsScoreCollection"],
            has_been_visited=data["hasBeenVisited"],
            number_of_visits_to_tile=data["numberOfVisitsToTile"],
            reward_on_destination=data["rewardOnDestination"],
            tag_on_tile=data.get("tagOnTile"),
        )


@dataclass(frozen=True)
class MazeState:
    possible_move_actions: list[MoveAction]
    can_collect_score_here: bool
    can_exit_maze_here: bool
    current_score_in_hand: int
    current_score_in_bag: int
    tag_on_current_tile: int | None
    number_of_visits_to_tile: int

    @staticmethod
    def from_api(data: dict[str, Any]) -> "MazeState":
        return MazeState(
            possible_move_actions=[
                MoveAction.from_api(action)
                for action in data.get("possibleMoveActions", [])
            ],
            can_collect_score_here=data["canCollectScoreHere"],
            can_exit_maze_here=data["canExitMazeHere"],
            current_score_in_hand=data["currentScoreInHand"],
            current_score_in_bag=data["currentScoreInBag"],
            tag_on_current_tile=data.get("tagOnCurrentTile"),
            number_of_visits_to_tile=data.get("numberOfVisitsToTile", 0),
        )