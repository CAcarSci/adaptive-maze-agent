from src.bots.baseline_bot import BaselineMazeBot
from src.data.telemetry_logger import TelemetryLogger
from src.maze_client import MazeClient
from src.policies.reward_aware_policy import RewardAwarePolicy


class SmartMazeBot(BaselineMazeBot):
    """
    Smart bot implementation for Step 3.

    This bot reuses the same robust maze orchestration as the baseline bot,
    but injects a smarter reward-aware navigation policy.
    """

    def __init__(
        self,
        client: MazeClient,
        max_steps: int = 2_000,
        telemetry_logger: TelemetryLogger | None = None,
    ) -> None:
        super().__init__(
            client=client,
            max_steps=max_steps,
            telemetry_logger=telemetry_logger,
            policy=RewardAwarePolicy(),
        )