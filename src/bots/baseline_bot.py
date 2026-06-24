from src.data.telemetry_logger import TelemetryLogger
from src.maze_client import MazeClient
from src.models import (
    DIRECTION_ORDER,
    OPPOSITE_DIRECTION,
    MazeState,
    MoveAction,
)


class BaselineMazeBot:
    """
    Step 1 baseline bot.

    This bot intentionally uses a simple DFS-like exploration strategy:
    - explore unvisited tiles first
    - backtrack when no unvisited move is available
    - collect score whenever possible
    - remember the first exit found
    - remember the first collection point found
    - collect remaining score before exiting if possible

    DFS (Depth-First Search) is a graph traversal algorithm that explores as far as
    possible along each branch before backtracking. It uses a stack to keep track of
    the path, visiting unvisited neighbors and returning to previous nodes when no
    unvisited neighbors remain.

    The goal of this step is not to be optimal yet, but to create a reliable
    baseline for later data collection, smarter policy development and evaluation.
    """

    BOT_NAME = "baseline_dfs"

    def __init__(
        self,
        client: MazeClient,
        max_steps: int = 2_000,
        telemetry_logger: TelemetryLogger | None = None,
    ) -> None:
        self.client = client
        self.max_steps = max_steps
        self.telemetry_logger = telemetry_logger

        self.backtrack_stack: list[str] = []
        self.path_to_current_tile: list[str] = []

        self.path_to_exit: list[str] | None = None
        self.path_to_collection: list[str] | None = None

    def solve(self, maze_name: str) -> None:
        print(f"Entering maze: {maze_name}")
        state = self.client.enter_maze(maze_name)

        for step in range(self.max_steps):
            self._print_step(step, state)

            self._remember_exit_if_possible(state)
            self._remember_collection_if_possible(state)
            state = self._collect_if_possible(state)

            next_action = self._choose_unvisited_action(state)

            if next_action is not None:
                self._log_decision(
                    maze_name=maze_name,
                    step=step,
                    state=state,
                    decision_type="explore",
                    chosen_direction=next_action.direction,
                )
                state = self._move_forward(next_action)
                continue

            if self.backtrack_stack:
                backtrack_direction = self.backtrack_stack[-1]
                self._log_decision(
                    maze_name=maze_name,
                    step=step,
                    state=state,
                    decision_type="backtrack",
                    chosen_direction=backtrack_direction,
                )
                state = self._backtrack()
                continue

            self._log_decision(
                maze_name=maze_name,
                step=step,
                state=state,
                decision_type="stop",
                chosen_direction=None,
            )
            print("No unvisited actions and no backtracking left. Exploration complete.")
            break

        if self.path_to_exit is None:
            print("No exit discovered. Cannot exit maze safely.")
            return

        state = self._collect_remaining_score_before_exit(state)
        state = self._navigate_to_exit(state)

        if state.can_exit_maze_here:
            print("Exit tile reached. Exiting maze.")
            self.client.exit_maze()
        else:
            print("Expected to be on an exit tile, but can_exit_maze_here is False.")

    def _print_step(self, step: int, state: MazeState) -> None:
        print(
            f"Step={step} | "
            f"hand={state.current_score_in_hand} | "
            f"bag={state.current_score_in_bag} | "
            f"can_collect={state.can_collect_score_here} | "
            f"can_exit={state.can_exit_maze_here} | "
            f"available_moves={[a.direction for a in state.possible_move_actions]}"
        )

    def _remember_exit_if_possible(self, state: MazeState) -> None:
        if state.can_exit_maze_here and self.path_to_exit is None:
            self.path_to_exit = list(self.path_to_current_tile)
            print(f"Exit discovered. Path to exit stored: {self.path_to_exit}")

    def _remember_collection_if_possible(self, state: MazeState) -> None:
        if state.can_collect_score_here and self.path_to_collection is None:
            self.path_to_collection = list(self.path_to_current_tile)
            print(
                "Collection point discovered. "
                f"Path stored: {self.path_to_collection}"
            )

    def _collect_if_possible(self, state: MazeState) -> MazeState:
        if state.can_collect_score_here and state.current_score_in_hand > 0:
            print(f"Collecting score: {state.current_score_in_hand}")
            return self.client.collect_score()

        return state

    def _choose_unvisited_action(self, state: MazeState) -> MoveAction | None:
        actions_by_direction = {
            action.direction: action
            for action in state.possible_move_actions
        }

        for direction in DIRECTION_ORDER:
            action = actions_by_direction.get(direction)

            if action is not None and not action.has_been_visited:
                return action

        return None

    def _move_forward(self, action: MoveAction) -> MazeState:
        print(
            f"Moving {action.direction} "
            f"(destination_reward={action.reward_on_destination})"
        )

        self.backtrack_stack.append(OPPOSITE_DIRECTION[action.direction])
        self.path_to_current_tile.append(action.direction)

        return self.client.move(action.direction)

    def _backtrack(self) -> MazeState:
        direction = self.backtrack_stack.pop()
        print(f"Backtracking {direction}")

        if self.path_to_current_tile:
            self.path_to_current_tile.pop()

        return self.client.move(direction)

    def _collect_remaining_score_before_exit(self, current_state: MazeState) -> MazeState:
        if current_state.current_score_in_hand <= 0:
            return current_state

        if self.path_to_collection is None:
            print(
                "Score still in hand, but no collection point is known. "
                "Proceeding to exit."
            )
            return current_state

        print(
            "Score still in hand after exploration. "
            f"Navigating to collection point via path: {self.path_to_collection}"
        )

        if self.path_to_collection == []:
            state = current_state
        else:
            state = self._follow_path(self.path_to_collection)

        state = self._collect_if_possible(state)

        if self.path_to_collection == []:
            return state

        path_back_to_start = self._reverse_path(self.path_to_collection)
        print(f"Returning to start via path: {path_back_to_start}")

        return self._follow_path(path_back_to_start)

    def _navigate_to_exit(self, current_state: MazeState) -> MazeState:
        if self.path_to_exit is None:
            raise RuntimeError("Cannot navigate to exit because no exit is known.")

        if self.path_to_exit == []:
            print("Exit is located on the start tile.")
            return current_state

        print(f"Navigating to known exit via path: {self.path_to_exit}")
        return self._follow_path(self.path_to_exit)

    def _follow_path(self, path: list[str]) -> MazeState:
        state: MazeState | None = None

        for direction in path:
            print(f"Following path direction: {direction}")
            state = self.client.move(direction)

        if state is None:
            raise RuntimeError("Cannot follow an empty path without current state.")

        return state

    def _reverse_path(self, path: list[str]) -> list[str]:
        return [
            OPPOSITE_DIRECTION[direction]
            for direction in reversed(path)
        ]

    def _log_decision(
        self,
        *,
        maze_name: str,
        step: int,
        state: MazeState,
        decision_type: str,
        chosen_direction: str | None,
    ) -> None:
        if self.telemetry_logger is None:
            return

        self.telemetry_logger.log_decision(
            maze_name=maze_name,
            bot_name=self.BOT_NAME,
            phase="exploration",
            step=step,
            decision_type=decision_type,
            state=state,
            chosen_direction=chosen_direction,
            path_depth=len(self.path_to_current_tile),
        )