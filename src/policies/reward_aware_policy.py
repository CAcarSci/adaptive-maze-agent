from dataclasses import dataclass

from src.models import MazeState, MoveAction
from src.policies.navigation_policy import NavigationContext


@dataclass(frozen=True)
class RewardAwarePolicyConfig:
    """
    Tunable weights for the reward-aware policy.

    These weights are intentionally simple and explainable. They can later be
    tuned based on telemetry analysis or replaced by a trained policy.
    """

    novelty_bonus: float = 100.0
    reward_weight: float = 1.0
    collection_bonus: float = 35.0
    exit_discovery_bonus: float = 10.0
    revisit_penalty: float = 25.0
    start_tile_penalty: float = 5.0


class RewardAwarePolicy:
    """
    First smarter navigation policy.

    The policy still only chooses from unvisited actions to avoid loops and keep
    exploration complete. Among unvisited actions it prioritizes:

    - new tiles
    - higher immediate reward
    - score collection opportunities
    - exit discovery
    - lower revisit counts

    This makes the strategy data-driven and explainable without overengineering
    the challenge into a full reinforcement learning problem.
    """

    name = "reward_aware"

    def __init__(self, config: RewardAwarePolicyConfig | None = None) -> None:
        self.config = config or RewardAwarePolicyConfig()

    def choose_action(
        self,
        state: MazeState,
        context: NavigationContext,
    ) -> MoveAction | None:
        unvisited_actions = [
            action
            for action in state.possible_move_actions
            if not action.has_been_visited
        ]

        if not unvisited_actions:
            return None

        return max(
            unvisited_actions,
            key=lambda action: self.score_action(
                action=action,
                state=state,
                context=context,
            ),
        )

    def score_action(
        self,
        *,
        action: MoveAction,
        state: MazeState,
        context: NavigationContext,
    ) -> float:
        score = 0.0

        if not action.has_been_visited:
            score += self.config.novelty_bonus
        else:
            score -= self.config.revisit_penalty * action.number_of_visits_to_tile

        score += self.config.reward_weight * action.reward_on_destination

        if action.allows_score_collection:
            score += self.config.collection_bonus

            if state.current_score_in_hand > 0:
                score += self.config.collection_bonus

        if action.allows_exit:
            score += self.config.exit_discovery_bonus

        if action.is_start:
            score -= self.config.start_tile_penalty

        return score