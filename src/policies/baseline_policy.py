from src.models import DIRECTION_ORDER, MazeState, MoveAction
from src.policies.navigation_policy import NavigationContext


class BaselineDfsPolicy:
    """
    Deterministic baseline policy.

    This policy selects the first unvisited action based on a stable direction
    order. It is intentionally simple and used as the reference strategy.
    """

    name = "baseline_dfs"

    def choose_action(
        self,
        state: MazeState,
        context: NavigationContext,
    ) -> MoveAction | None:
        actions_by_direction = {
            action.direction: action
            for action in state.possible_move_actions
        }

        for direction in DIRECTION_ORDER:
            action = actions_by_direction.get(direction)

            if action is not None and not action.has_been_visited:
                return action

        return None