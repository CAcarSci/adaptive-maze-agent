import csv
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.models import MazeState, MoveAction


class TelemetryLogger:
    """
    Logs structured decision telemetry for maze navigation.

    The logger writes one row per candidate action. This makes it possible
    to analyze not only what the bot chose, but also what alternatives were
    available at that decision point.
    """

    FIELD_NAMES = [
        "run_id",
        "timestamp_utc",
        "maze_name",
        "bot_name",
        "phase",
        "step",
        "decision_type",
        "chosen_direction",
        "candidate_direction",
        "is_chosen",
        "current_score_in_hand",
        "current_score_in_bag",
        "can_collect_score_here",
        "can_exit_maze_here",
        "current_tile_tag",
        "current_tile_visit_count",
        "path_depth",
        "available_action_count",
        "candidate_reward_on_destination",
        "candidate_has_been_visited",
        "candidate_visit_count",
        "candidate_allows_exit",
        "candidate_allows_score_collection",
        "candidate_is_start",
        "candidate_tag_on_tile",
    ]

    def __init__(self, output_path: str, run_id: str | None = None) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or str(uuid.uuid4())

        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.output_path.exists() and self.output_path.stat().st_size > 0:
            return

        with self.output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.FIELD_NAMES)
            writer.writeheader()

    def log_decision(
        self,
        *,
        maze_name: str,
        bot_name: str,
        phase: str,
        step: int,
        decision_type: str,
        state: MazeState,
        chosen_direction: str | None,
        path_depth: int,
    ) -> None:
        """
        Logs all candidate move actions for a single bot decision.

        decision_type examples:
        - explore
        - backtrack
        - stop
        """

        if not state.possible_move_actions:
            self._write_row(
                maze_name=maze_name,
                bot_name=bot_name,
                phase=phase,
                step=step,
                decision_type=decision_type,
                state=state,
                chosen_direction=chosen_direction,
                candidate_action=None,
                path_depth=path_depth,
            )
            return

        for candidate_action in state.possible_move_actions:
            self._write_row(
                maze_name=maze_name,
                bot_name=bot_name,
                phase=phase,
                step=step,
                decision_type=decision_type,
                state=state,
                chosen_direction=chosen_direction,
                candidate_action=candidate_action,
                path_depth=path_depth,
            )

    def _write_row(
        self,
        *,
        maze_name: str,
        bot_name: str,
        phase: str,
        step: int,
        decision_type: str,
        state: MazeState,
        chosen_direction: str | None,
        candidate_action: MoveAction | None,
        path_depth: int,
    ) -> None:
        row = {
            "run_id": self.run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "maze_name": maze_name,
            "bot_name": bot_name,
            "phase": phase,
            "step": step,
            "decision_type": decision_type,
            "chosen_direction": chosen_direction,
            "candidate_direction": self._get_candidate_value(
                candidate_action, "direction"
            ),
            "is_chosen": (
                candidate_action is not None
                and candidate_action.direction == chosen_direction
            ),
            "current_score_in_hand": state.current_score_in_hand,
            "current_score_in_bag": state.current_score_in_bag,
            "can_collect_score_here": state.can_collect_score_here,
            "can_exit_maze_here": state.can_exit_maze_here,
            "current_tile_tag": state.tag_on_current_tile,
            "current_tile_visit_count": state.number_of_visits_to_tile,
            "path_depth": path_depth,
            "available_action_count": len(state.possible_move_actions),
            "candidate_reward_on_destination": self._get_candidate_value(
                candidate_action, "reward_on_destination"
            ),
            "candidate_has_been_visited": self._get_candidate_value(
                candidate_action, "has_been_visited"
            ),
            "candidate_visit_count": self._get_candidate_value(
                candidate_action, "number_of_visits_to_tile"
            ),
            "candidate_allows_exit": self._get_candidate_value(
                candidate_action, "allows_exit"
            ),
            "candidate_allows_score_collection": self._get_candidate_value(
                candidate_action, "allows_score_collection"
            ),
            "candidate_is_start": self._get_candidate_value(
                candidate_action, "is_start"
            ),
            "candidate_tag_on_tile": self._get_candidate_value(
                candidate_action, "tag_on_tile"
            ),
        }

        with self.output_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.FIELD_NAMES)
            writer.writerow(row)

    @staticmethod
    def _get_candidate_value(
        candidate_action: MoveAction | None,
        attribute_name: str,
    ) -> object:
        if candidate_action is None:
            return None

        return getattr(candidate_action, attribute_name)