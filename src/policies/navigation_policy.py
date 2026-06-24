from dataclasses import dataclass
from typing import Protocol

from src.models import MazeState, MoveAction


@dataclass(frozen=True)
class NavigationContext:
    """
    Runtime context provided to navigation policies.

    This keeps policy decisions independent from the bot orchestration.
    Later we can add more context here without changing every policy.
    """

    maze_name: str
    step: int
    path_depth: int
    has_known_exit: bool
    has_known_collection_point: bool


class NavigationPolicy(Protocol):
    """
    Contract for navigation policies.

    A policy receives the current maze state and returns the next forward
    action to take. If no forward action should be taken, it returns None
    and the bot can decide to backtrack.
    """

    name: str

    def choose_action(
        self,
        state: MazeState,
        context: NavigationContext,
    ) -> MoveAction | None:
        ...