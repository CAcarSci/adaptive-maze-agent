from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.bots.baseline_bot import BaselineMazeBot
from src.bots.decision_tree_bot import DecisionTreeMazeBot
from src.bots.smart_bot import SmartMazeBot
from src.config import load_settings
from src.data.telemetry_logger import TelemetryLogger
from src.maze_client import MazeApiError, MazeClient


EVALUATION_MAZES = [
    "Example Maze",
    "Gradius Pathways",
    "Hello Maze",
]

BOT_TYPES = [
    "baseline",
    "smart",
    "decision_tree",
]

TELEMETRY_OUTPUT_PATH = Path("experiments/evaluation_action_logs.csv")
RESULTS_OUTPUT_PATH = Path("reports/evaluation_results.csv")
REPORT_OUTPUT_PATH = Path("reports/evaluation_report.md")


@dataclass(frozen=True)
class EvaluationResult:
    bot_type: str
    bot_name: str
    maze_name: str
    run_id: str
    exit_found: bool
    final_score_delta: int
    final_player_score: int
    steps_logged: int
    chosen_actions: int
    explore_decisions: int
    backtrack_decisions: int
    stop_decisions: int
    avg_chosen_reward: float
    revisit_ratio: float
    max_score_in_hand: int
    max_score_in_bag: int


def create_client() -> MazeClient:
    settings = load_settings()

    return MazeClient(
        base_url=settings.base_url,
        authorization_header=settings.authorization_header,
    )


def reset_player(client: MazeClient, player_name: str) -> None:
    try:
        client.forget_player()
    except MazeApiError:
        pass

    try:
        client.register_player(player_name)
    except MazeApiError as error:
        error_message = str(error)

        if "already" not in error_message.lower() and "409" not in error_message:
            raise


def create_bot(
    *,
    bot_type: str,
    client: MazeClient,
    telemetry_logger: TelemetryLogger,
) -> BaselineMazeBot:
    if bot_type == "baseline":
        return BaselineMazeBot(
            client=client,
            telemetry_logger=telemetry_logger,
        )

    if bot_type == "smart":
        return SmartMazeBot(
            client=client,
            telemetry_logger=telemetry_logger,
        )

    if bot_type == "decision_tree":
        return DecisionTreeMazeBot(
            client=client,
            telemetry_logger=telemetry_logger,
        )

    raise ValueError(f"Unsupported bot_type: {bot_type}")


def summarize_run_from_telemetry(
    *,
    telemetry_path: Path,
    run_id: str,
) -> dict[str, float | int]:
    df = pd.read_csv(telemetry_path)
    run_df = df[df["run_id"] == run_id].copy()

    if run_df.empty:
        return {
            "steps_logged": 0,
            "chosen_actions": 0,
            "explore_decisions": 0,
            "backtrack_decisions": 0,
            "stop_decisions": 0,
            "avg_chosen_reward": 0.0,
            "revisit_ratio": 0.0,
            "max_score_in_hand": 0,
            "max_score_in_bag": 0,
        }

    run_df["is_chosen"] = run_df["is_chosen"].map(
        {
            True: True,
            False: False,
            "True": True,
            "False": False,
            "true": True,
            "false": False,
        }
    )

    numeric_columns = [
        "candidate_reward_on_destination",
        "candidate_has_been_visited",
        "current_score_in_hand",
        "current_score_in_bag",
    ]

    for column in numeric_columns:
        if column in run_df.columns:
            run_df[column] = pd.to_numeric(run_df[column], errors="coerce")

    chosen_df = run_df[
        (run_df["candidate_direction"].notna())
        & (run_df["is_chosen"] == True)
    ].copy()

    decision_df = run_df[["step", "decision_type"]].drop_duplicates()

    if chosen_df.empty:
        revisit_ratio = 0.0
        avg_chosen_reward = 0.0
    else:
        revisit_ratio = float(chosen_df["candidate_has_been_visited"].mean())
        avg_chosen_reward = float(
            chosen_df["candidate_reward_on_destination"].mean()
        )

    return {
        "steps_logged": int(decision_df["step"].nunique()),
        "chosen_actions": int(len(chosen_df)),
        "explore_decisions": int((decision_df["decision_type"] == "explore").sum()),
        "backtrack_decisions": int((decision_df["decision_type"] == "backtrack").sum()),
        "stop_decisions": int((decision_df["decision_type"] == "stop").sum()),
        "avg_chosen_reward": avg_chosen_reward,
        "revisit_ratio": revisit_ratio,
        "max_score_in_hand": int(run_df["current_score_in_hand"].max()),
        "max_score_in_bag": int(run_df["current_score_in_bag"].max()),
    }


def evaluate_single_run(
    *,
    bot_type: str,
    maze_name: str,
) -> EvaluationResult:
    settings = load_settings()
    client = create_client()

    reset_player(
        client=client,
        player_name=settings.player_name,
    )

    player_before = client.get_player()
    score_before = int(player_before.get("playerScore", 0))

    telemetry_logger = TelemetryLogger(
        output_path=str(TELEMETRY_OUTPUT_PATH),
    )

    bot = create_bot(
        bot_type=bot_type,
        client=client,
        telemetry_logger=telemetry_logger,
    )

    exit_found = True

    try:
        bot.solve(maze_name)
    except Exception:
        exit_found = False
        raise

    player_after = client.get_player()
    score_after = int(player_after.get("playerScore", 0))

    telemetry_summary = summarize_run_from_telemetry(
        telemetry_path=TELEMETRY_OUTPUT_PATH,
        run_id=telemetry_logger.run_id,
    )

    return EvaluationResult(
        bot_type=bot_type,
        bot_name=bot.policy.name,
        maze_name=maze_name,
        run_id=telemetry_logger.run_id,
        exit_found=exit_found and not player_after.get("isInMaze", False),
        final_score_delta=score_after - score_before,
        final_player_score=score_after,
        steps_logged=int(telemetry_summary["steps_logged"]),
        chosen_actions=int(telemetry_summary["chosen_actions"]),
        explore_decisions=int(telemetry_summary["explore_decisions"]),
        backtrack_decisions=int(telemetry_summary["backtrack_decisions"]),
        stop_decisions=int(telemetry_summary["stop_decisions"]),
        avg_chosen_reward=float(telemetry_summary["avg_chosen_reward"]),
        revisit_ratio=float(telemetry_summary["revisit_ratio"]),
        max_score_in_hand=int(telemetry_summary["max_score_in_hand"]),
        max_score_in_bag=int(telemetry_summary["max_score_in_bag"]),
    )


def write_results_csv(results: list[EvaluationResult]) -> pd.DataFrame:
    RESULTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([asdict(result) for result in results])
    df.to_csv(RESULTS_OUTPUT_PATH, index=False)

    return df


def format_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"

    formatted_df = df.copy()

    for column in formatted_df.select_dtypes(include=["float"]).columns:
        formatted_df[column] = formatted_df[column].round(3)

    return formatted_df.to_markdown(index=False)


def write_evaluation_report(results_df: pd.DataFrame) -> None:
    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary_by_bot = (
        results_df.groupby("bot_name", dropna=False)
        .agg(
            runs=("run_id", "count"),
            avg_score=("final_score_delta", "mean"),
            total_score=("final_score_delta", "sum"),
            avg_steps=("steps_logged", "mean"),
            avg_chosen_reward=("avg_chosen_reward", "mean"),
            avg_revisit_ratio=("revisit_ratio", "mean"),
            exit_success_rate=("exit_found", "mean"),
        )
        .reset_index()
        .sort_values("avg_score", ascending=False)
    )

    summary_by_maze_and_bot = (
        results_df[
            [
                "maze_name",
                "bot_name",
                "final_score_delta",
                "steps_logged",
                "avg_chosen_reward",
                "revisit_ratio",
                "exit_found",
            ]
        ]
        .sort_values(["maze_name", "bot_name"])
        .reset_index(drop=True)
    )

    lines = [
        "# Bot Evaluation Report",
        "",
        "This report compares all implemented navigation policies on the same maze set.",
        "",
        "The goal is to evaluate whether the smarter policies improve behavior compared with the deterministic DFS baseline.",
        "",
        "## Evaluated Policies",
        "",
        "- `baseline_dfs`: deterministic DFS-style baseline policy",
        "- `reward_aware`: explainable heuristic policy based on telemetry insights",
        "- `decision_tree`: lightweight ML policy trained from telemetry-derived labels",
        "",
        "## Evaluation Mazes",
        "",
        *[f"- `{maze_name}`" for maze_name in EVALUATION_MAZES],
        "",
        "## Metric Definitions",
        "",
        "| Metric | Meaning |",
        "|:-------|:--------|",
        "| final_score_delta | Score gained during the evaluated run. Calculated from player score before and after the run. |",
        "| steps_logged | Number of decision steps logged during exploration. |",
        "| avg_chosen_reward | Average immediate reward on candidate actions selected by the policy. |",
        "| revisit_ratio | Share of chosen actions that moved to already visited destination tiles. Lower is usually better. |",
        "| exit_success_rate | Fraction of runs where the bot successfully exited the maze. |",
        "",
        "## Summary by Bot Policy",
        "",
        format_markdown_table(summary_by_bot),
        "",
        "## Results by Maze and Bot",
        "",
        format_markdown_table(summary_by_maze_and_bot),
        "",
        "## Preliminary Interpretation",
        "",
        "This evaluation is intentionally lightweight. It focuses on run-level behavior rather than only individual decision rows.",
        "",
        "The most important comparison is whether the smarter policies achieve similar or higher score with fewer revisits and reasonable step counts compared with the baseline.",
        "",
        "The Decision Tree policy should be interpreted carefully because it is trained from telemetry-derived labels, not from human labels or final maze outcomes. Its value is primarily explainability and testing whether structured candidate-action features can learn a useful preference policy.",
        "",
    ]

    REPORT_OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    results: list[EvaluationResult] = []

    if TELEMETRY_OUTPUT_PATH.exists():
        TELEMETRY_OUTPUT_PATH.unlink()

    for maze_name in EVALUATION_MAZES:
        for bot_type in BOT_TYPES:
            print(f"Evaluating bot_type={bot_type} on maze='{maze_name}'")

            result = evaluate_single_run(
                bot_type=bot_type,
                maze_name=maze_name,
            )

            results.append(result)

            print(
                f"Completed: bot={result.bot_name}, "
                f"maze={result.maze_name}, "
                f"score_delta={result.final_score_delta}, "
                f"steps={result.steps_logged}, "
                f"exit_found={result.exit_found}"
            )

    results_df = write_results_csv(results)
    write_evaluation_report(results_df)

    print(f"Evaluation results written to: {RESULTS_OUTPUT_PATH}")
    print(f"Evaluation report written to: {REPORT_OUTPUT_PATH}")
    print(f"Evaluation telemetry written to: {TELEMETRY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()