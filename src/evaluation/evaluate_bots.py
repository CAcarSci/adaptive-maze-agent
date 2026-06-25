from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.bots.baseline_bot import BaselineMazeBot
from src.bots.decision_tree_bot import DecisionTreeMazeBot
from src.bots.smart_bot import SmartMazeBot
from src.config import load_settings
from src.data.telemetry_logger import TelemetryLogger
from src.maze_client import MazeApiError, MazeClient
from src.tracking.mlflow_tracker import (
    log_artifacts,
    log_metrics,
    log_params,
    mlflow_run,
    sanitize_mlflow_key,
)


@dataclass(frozen=True)
class EvaluationMaze:
    name: str
    group: str


EVALUATION_MAZES = [
    EvaluationMaze(name="Example Maze", group="seen"),
    EvaluationMaze(name="Gradius Pathways", group="seen"),
    EvaluationMaze(name="Hello Maze", group="seen"),
    EvaluationMaze(name="Exit", group="unseen"),
    EvaluationMaze(name="O Contra", group="unseen"),
    EvaluationMaze(name="Dig Down", group="unseen"),
    EvaluationMaze(name="Glasses", group="unseen"),
    EvaluationMaze(name="Reverse", group="unseen"),
    EvaluationMaze(name="Loops", group="unseen"),
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
    maze_group: str
    run_id: str
    exit_found: bool
    final_score_delta: int
    final_player_score: int
    steps_logged: int
    chosen_actions: int
    explore_decisions: int
    backtrack_decisions: int
    stop_decisions: int
    backtrack_ratio: float
    avg_chosen_reward: float
    explore_avg_chosen_reward: float
    revisit_ratio: float
    explore_revisit_ratio: float
    score_per_step: float
    first_exit_step: int | None
    first_collection_step: int | None
    max_score_in_hand: int
    max_score_in_bag: int
    score_in_bag_at_step_10: int
    score_in_bag_at_step_25: int
    score_in_bag_at_step_50: int
    score_progress_at_step_10: int
    score_progress_at_step_25: int
    score_progress_at_step_50: int


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


def to_bool(value: Any) -> bool:
    return value in {True, "True", "true", "1", 1}


def prepare_run_dataframe(
    *,
    telemetry_path: Path,
    run_id: str,
) -> pd.DataFrame:
    df = pd.read_csv(telemetry_path)
    run_df = df[df["run_id"] == run_id].copy()

    if run_df.empty:
        return run_df

    boolean_columns = [
        "is_chosen",
        "can_collect_score_here",
        "can_exit_maze_here",
        "candidate_has_been_visited",
        "candidate_allows_exit",
        "candidate_allows_score_collection",
        "candidate_is_start",
    ]

    for column in boolean_columns:
        if column in run_df.columns:
            run_df[column] = run_df[column].apply(to_bool)

    numeric_columns = [
        "step",
        "candidate_reward_on_destination",
        "candidate_visit_count",
        "current_score_in_hand",
        "current_score_in_bag",
        "path_depth",
        "available_action_count",
    ]

    for column in numeric_columns:
        if column in run_df.columns:
            run_df[column] = pd.to_numeric(run_df[column], errors="coerce")

    return run_df


def get_step_level_dataframe(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return run_df

    return (
        run_df.sort_values("step")
        .drop_duplicates(subset=["step"])
        .reset_index(drop=True)
    )


def get_first_step_where(
    *,
    step_df: pd.DataFrame,
    column: str,
) -> int | None:
    if step_df.empty or column not in step_df.columns:
        return None

    matching_rows = step_df[step_df[column].fillna(False).astype(bool)]

    if matching_rows.empty:
        return None

    return int(matching_rows["step"].min())


def get_checkpoint_scores(
    *,
    step_df: pd.DataFrame,
    checkpoint: int,
) -> tuple[int, int]:
    """
    Returns:
    - score_in_bag: secured score by the checkpoint
    - score_progress: score in bag + score in hand by the checkpoint
    """

    if step_df.empty:
        return 0, 0

    checkpoint_df = step_df[step_df["step"] <= checkpoint].copy()

    if checkpoint_df.empty:
        return 0, 0

    score_in_bag = int(
        checkpoint_df["current_score_in_bag"]
        .fillna(0)
        .max()
    )

    score_progress = int(
        (
            checkpoint_df["current_score_in_bag"].fillna(0)
            + checkpoint_df["current_score_in_hand"].fillna(0)
        ).max()
    )

    return score_in_bag, score_progress


def summarize_run_from_telemetry(
    *,
    telemetry_path: Path,
    run_id: str,
) -> dict[str, Any]:
    run_df = prepare_run_dataframe(
        telemetry_path=telemetry_path,
        run_id=run_id,
    )

    if run_df.empty:
        return {
            "steps_logged": 0,
            "chosen_actions": 0,
            "explore_decisions": 0,
            "backtrack_decisions": 0,
            "stop_decisions": 0,
            "backtrack_ratio": 0.0,
            "avg_chosen_reward": 0.0,
            "explore_avg_chosen_reward": 0.0,
            "revisit_ratio": 0.0,
            "explore_revisit_ratio": 0.0,
            "first_exit_step": None,
            "first_collection_step": None,
            "max_score_in_hand": 0,
            "max_score_in_bag": 0,
            "score_in_bag_at_step_10": 0,
            "score_in_bag_at_step_25": 0,
            "score_in_bag_at_step_50": 0,
            "score_progress_at_step_10": 0,
            "score_progress_at_step_25": 0,
            "score_progress_at_step_50": 0,
        }

    step_df = get_step_level_dataframe(run_df)

    chosen_df = run_df[
        (run_df["candidate_direction"].notna())
        & run_df["is_chosen"].fillna(False).astype(bool)
    ].copy()

    explore_chosen_df = chosen_df[chosen_df["decision_type"] == "explore"].copy()

    decision_df = run_df[["step", "decision_type"]].drop_duplicates()

    steps_logged = int(decision_df["step"].nunique())
    explore_decisions = int((decision_df["decision_type"] == "explore").sum())
    backtrack_decisions = int((decision_df["decision_type"] == "backtrack").sum())
    stop_decisions = int((decision_df["decision_type"] == "stop").sum())

    if chosen_df.empty:
        revisit_ratio = 0.0
        avg_chosen_reward = 0.0
    else:
        revisit_ratio = float(chosen_df["candidate_has_been_visited"].mean())
        avg_chosen_reward = float(
            chosen_df["candidate_reward_on_destination"].mean()
        )

    if explore_chosen_df.empty:
        explore_revisit_ratio = 0.0
        explore_avg_chosen_reward = 0.0
    else:
        explore_revisit_ratio = float(
            explore_chosen_df["candidate_has_been_visited"].mean()
        )
        explore_avg_chosen_reward = float(
            explore_chosen_df["candidate_reward_on_destination"].mean()
        )

    score_in_bag_at_step_10, score_progress_at_step_10 = get_checkpoint_scores(
        step_df=step_df,
        checkpoint=10,
    )
    score_in_bag_at_step_25, score_progress_at_step_25 = get_checkpoint_scores(
        step_df=step_df,
        checkpoint=25,
    )
    score_in_bag_at_step_50, score_progress_at_step_50 = get_checkpoint_scores(
        step_df=step_df,
        checkpoint=50,
    )

    return {
        "steps_logged": steps_logged,
        "chosen_actions": int(len(chosen_df)),
        "explore_decisions": explore_decisions,
        "backtrack_decisions": backtrack_decisions,
        "stop_decisions": stop_decisions,
        "backtrack_ratio": (
            float(backtrack_decisions / steps_logged)
            if steps_logged > 0
            else 0.0
        ),
        "avg_chosen_reward": avg_chosen_reward,
        "explore_avg_chosen_reward": explore_avg_chosen_reward,
        "revisit_ratio": revisit_ratio,
        "explore_revisit_ratio": explore_revisit_ratio,
        "first_exit_step": get_first_step_where(
            step_df=step_df,
            column="can_exit_maze_here",
        ),
        "first_collection_step": get_first_step_where(
            step_df=step_df,
            column="can_collect_score_here",
        ),
        "max_score_in_hand": int(step_df["current_score_in_hand"].fillna(0).max()),
        "max_score_in_bag": int(step_df["current_score_in_bag"].fillna(0).max()),
        "score_in_bag_at_step_10": score_in_bag_at_step_10,
        "score_in_bag_at_step_25": score_in_bag_at_step_25,
        "score_in_bag_at_step_50": score_in_bag_at_step_50,
        "score_progress_at_step_10": score_progress_at_step_10,
        "score_progress_at_step_25": score_progress_at_step_25,
        "score_progress_at_step_50": score_progress_at_step_50,
    }


def evaluate_single_run(
    *,
    bot_type: str,
    evaluation_maze: EvaluationMaze,
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

    try:
        bot.solve(evaluation_maze.name)
    except Exception as error:
        raise RuntimeError(
            f"Evaluation failed for bot_type={bot_type}, "
            f"maze={evaluation_maze.name}: {error}"
        ) from error

    player_after = client.get_player()
    score_after = int(player_after.get("playerScore", 0))
    final_score_delta = score_after - score_before

    telemetry_summary = summarize_run_from_telemetry(
        telemetry_path=TELEMETRY_OUTPUT_PATH,
        run_id=telemetry_logger.run_id,
    )

    steps_logged = int(telemetry_summary["steps_logged"])

    score_per_step = (
        float(final_score_delta / steps_logged)
        if steps_logged > 0
        else 0.0
    )

    return EvaluationResult(
        bot_type=bot_type,
        bot_name=bot.policy.name,
        maze_name=evaluation_maze.name,
        maze_group=evaluation_maze.group,
        run_id=telemetry_logger.run_id,
        exit_found=not player_after.get("isInMaze", False),
        final_score_delta=final_score_delta,
        final_player_score=score_after,
        steps_logged=steps_logged,
        chosen_actions=int(telemetry_summary["chosen_actions"]),
        explore_decisions=int(telemetry_summary["explore_decisions"]),
        backtrack_decisions=int(telemetry_summary["backtrack_decisions"]),
        stop_decisions=int(telemetry_summary["stop_decisions"]),
        backtrack_ratio=float(telemetry_summary["backtrack_ratio"]),
        avg_chosen_reward=float(telemetry_summary["avg_chosen_reward"]),
        explore_avg_chosen_reward=float(
            telemetry_summary["explore_avg_chosen_reward"]
        ),
        revisit_ratio=float(telemetry_summary["revisit_ratio"]),
        explore_revisit_ratio=float(telemetry_summary["explore_revisit_ratio"]),
        score_per_step=score_per_step,
        first_exit_step=telemetry_summary["first_exit_step"],
        first_collection_step=telemetry_summary["first_collection_step"],
        max_score_in_hand=int(telemetry_summary["max_score_in_hand"]),
        max_score_in_bag=int(telemetry_summary["max_score_in_bag"]),
        score_in_bag_at_step_10=int(
            telemetry_summary["score_in_bag_at_step_10"]
        ),
        score_in_bag_at_step_25=int(
            telemetry_summary["score_in_bag_at_step_25"]
        ),
        score_in_bag_at_step_50=int(
            telemetry_summary["score_in_bag_at_step_50"]
        ),
        score_progress_at_step_10=int(
            telemetry_summary["score_progress_at_step_10"]
        ),
        score_progress_at_step_25=int(
            telemetry_summary["score_progress_at_step_25"]
        ),
        score_progress_at_step_50=int(
            telemetry_summary["score_progress_at_step_50"]
        ),
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


def format_metric_value(value: float) -> str:
    if pd.isna(value):
        return "n/a"

    numeric_value = float(value)

    if numeric_value.is_integer():
        return str(int(numeric_value))

    return f"{numeric_value:.3f}"


def format_bot_names(bot_names: list[str]) -> str:
    if not bot_names:
        return "n/a"

    if len(bot_names) == 1:
        return f"`{bot_names[0]}`"

    quoted_names = [f"`{bot_name}`" for bot_name in bot_names]

    if len(quoted_names) == 2:
        return " and ".join(quoted_names)

    return ", ".join(quoted_names[:-1]) + f" and {quoted_names[-1]}"


def get_metric_leaders(
    *,
    results_df: pd.DataFrame,
    metric: str,
    higher_is_better: bool,
    tolerance: float = 1e-9,
) -> tuple[list[str], float] | None:
    if results_df.empty or metric not in results_df.columns:
        return None

    metric_by_bot = results_df.groupby("bot_name")[metric].mean().dropna()

    if metric_by_bot.empty:
        return None

    best_value = (
        metric_by_bot.max()
        if higher_is_better
        else metric_by_bot.min()
    )

    leaders = metric_by_bot[
        (metric_by_bot - best_value).abs() <= tolerance
    ].index.tolist()

    return leaders, float(best_value)


def create_final_score_interpretation(results_df: pd.DataFrame) -> str:
    if results_df.empty:
        return "No evaluation results are available."

    final_scores_identical_per_maze = (
        results_df.groupby("maze_name")["final_score_delta"]
        .nunique()
        .eq(1)
        .all()
    )

    if final_scores_identical_per_maze:
        return (
            "Observed result: all policies achieved the same final score on each "
            "evaluated maze."
        )

    leader_result = get_metric_leaders(
        results_df=results_df,
        metric="final_score_delta",
        higher_is_better=True,
    )

    if leader_result is None:
        return "Final score comparison could not be calculated."

    leaders, best_value = leader_result

    return (
        "Observed result: "
        f"{format_bot_names(leaders)} achieved the highest average final score "
        f"with `{format_metric_value(best_value)}`."
    )


def create_data_driven_findings(results_df: pd.DataFrame) -> list[str]:
    if results_df.empty:
        return ["No evaluation results are available."]

    findings: list[str] = []

    findings.append(create_final_score_interpretation(results_df))

    metric_configs = [
        (
            "score_per_step",
            True,
            "Highest average score per logged step",
        ),
        (
            "score_progress_at_step_10",
            True,
            "Highest average reward progress by step 10",
        ),
        (
            "score_progress_at_step_25",
            True,
            "Highest average reward progress by step 25",
        ),
        (
            "score_progress_at_step_50",
            True,
            "Highest average reward progress by step 50",
        ),
        (
            "first_collection_step",
            False,
            "Lowest average first collection step",
        ),
        (
            "first_exit_step",
            False,
            "Lowest average first exit-capable tile step",
        ),
        (
            "backtrack_ratio",
            False,
            "Lowest average backtrack ratio",
        ),
    ]

    for metric, higher_is_better, label in metric_configs:
        leader_result = get_metric_leaders(
            results_df=results_df,
            metric=metric,
            higher_is_better=higher_is_better,
        )

        if leader_result is None:
            continue

        leaders, best_value = leader_result

        findings.append(
            f"{label}: {format_bot_names(leaders)} "
            f"(`{format_metric_value(best_value)}`)."
        )

    exit_success_result = get_metric_leaders(
        results_df=results_df,
        metric="exit_found",
        higher_is_better=True,
    )

    if exit_success_result is not None:
        leaders, best_value = exit_success_result

        findings.append(
            f"Highest average exit success rate: {format_bot_names(leaders)} "
            f"(`{format_metric_value(best_value)}`)."
        )

    return findings


def write_evaluation_report(results_df: pd.DataFrame) -> None:
    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    data_driven_findings = create_data_driven_findings(results_df)

    summary_by_bot = (
        results_df.groupby("bot_name", dropna=False)
        .agg(
            runs=("run_id", "count"),
            avg_score=("final_score_delta", "mean"),
            total_score=("final_score_delta", "sum"),
            avg_steps=("steps_logged", "mean"),
            avg_score_per_step=("score_per_step", "mean"),
            avg_explore_reward=("explore_avg_chosen_reward", "mean"),
            avg_explore_revisit_ratio=("explore_revisit_ratio", "mean"),
            avg_backtrack_ratio=("backtrack_ratio", "mean"),
            avg_first_exit_step=("first_exit_step", "mean"),
            avg_first_collection_step=("first_collection_step", "mean"),
            exit_success_rate=("exit_found", "mean"),
        )
        .reset_index()
        .sort_values(["avg_score", "avg_score_per_step"], ascending=False)
    )

    summary_by_group_and_bot = (
        results_df.groupby(["maze_group", "bot_name"], dropna=False)
        .agg(
            runs=("run_id", "count"),
            avg_score=("final_score_delta", "mean"),
            avg_steps=("steps_logged", "mean"),
            avg_score_per_step=("score_per_step", "mean"),
            avg_explore_reward=("explore_avg_chosen_reward", "mean"),
            avg_explore_revisit_ratio=("explore_revisit_ratio", "mean"),
            exit_success_rate=("exit_found", "mean"),
        )
        .reset_index()
        .sort_values(["maze_group", "avg_score"], ascending=[True, False])
    )

    checkpoint_summary = (
        results_df.groupby("bot_name", dropna=False)
        .agg(
            avg_score_in_bag_at_step_10=("score_in_bag_at_step_10", "mean"),
            avg_score_in_bag_at_step_25=("score_in_bag_at_step_25", "mean"),
            avg_score_in_bag_at_step_50=("score_in_bag_at_step_50", "mean"),
            avg_score_progress_at_step_10=("score_progress_at_step_10", "mean"),
            avg_score_progress_at_step_25=("score_progress_at_step_25", "mean"),
            avg_score_progress_at_step_50=("score_progress_at_step_50", "mean"),
        )
        .reset_index()
    )

    results_by_maze_and_bot = (
        results_df[
            [
                "maze_group",
                "maze_name",
                "bot_name",
                "final_score_delta",
                "steps_logged",
                "score_per_step",
                "explore_avg_chosen_reward",
                "explore_revisit_ratio",
                "backtrack_ratio",
                "first_exit_step",
                "first_collection_step",
                "score_progress_at_step_10",
                "score_progress_at_step_25",
                "score_progress_at_step_50",
                "exit_found",
            ]
        ]
        .sort_values(["maze_group", "maze_name", "bot_name"])
        .reset_index(drop=True)
    )

    lines = [
        "# Bot Evaluation Report",
        "",
        "This report compares all implemented navigation policies on the same maze set.",
        "",
        "## Evaluated Policies",
        "",
        "- `baseline_dfs`: deterministic DFS-style baseline policy",
        "- `reward_aware`: explainable heuristic policy based on telemetry insights",
        "- `decision_tree`: lightweight ML policy trained from telemetry-derived labels",
        "",
        "## Evaluation Mazes",
        "",
        *[
            f"- `{maze.name}` ({maze.group})"
            for maze in EVALUATION_MAZES
        ],
        "",
        "## Metric Definitions",
        "",
        "| Metric | Meaning |",
        "|:-------|:--------|",
        "| final_score_delta | Score gained during the evaluated run. Calculated from player score before and after the run. |",
        "| steps_logged | Number of decision steps logged during exploration. |",
        "| score_per_step | Final score divided by the number of logged decision steps. |",
        "| explore_avg_chosen_reward | Average immediate reward selected during forward exploration decisions only. |",
        "| explore_revisit_ratio | Share of forward exploration decisions that selected an already visited destination tile. |",
        "| backtrack_ratio | Share of decision steps that were backtracking decisions. |",
        "| first_exit_step | First logged step where the bot was standing on an exit-capable tile. |",
        "| first_collection_step | First logged step where the bot was standing on a score collection tile. |",
        "| score_in_bag_at_step_N | Score already secured in the bag by step N. |",
        "| score_progress_at_step_N | Score in bag plus score in hand by step N. |",
        "| exit_success_rate | Fraction of runs where the bot successfully exited the maze. |",
        "",
        "## Final Score Finding",
        "",
        create_final_score_interpretation(results_df),
        "",
        "## Data-Driven Findings",
        "",
        *[f"- {finding}" for finding in data_driven_findings],
        "",
        "## Summary by Bot Policy",
        "",
        format_markdown_table(summary_by_bot),
        "",
        "## Summary by Maze Group and Bot Policy",
        "",
        format_markdown_table(summary_by_group_and_bot),
        "",
        "## Early Reward Checkpoints",
        "",
        "This table shows whether a policy finds or secures reward earlier during exploration.",
        "",
        format_markdown_table(checkpoint_summary),
        "",
        "## Results by Maze and Bot",
        "",
        format_markdown_table(results_by_maze_and_bot),
        "",
        "## Report Scope",
        "",
        "This report is generated deterministically from evaluation results.",
        "",
    ]

    REPORT_OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")

def log_evaluation_to_mlflow(results_df: pd.DataFrame) -> None:
    if results_df.empty:
        return

    with mlflow_run(
        run_name="bot-policy-evaluation",
        tags={
            "stage": "evaluation",
            "evaluation_type": "policy_comparison",
        },
    ) as mlflow:
        if mlflow is None:
            return

        evaluated_policies = sorted(
            results_df["bot_name"].dropna().unique().tolist()
        )
        evaluated_mazes = sorted(
            results_df["maze_name"].dropna().unique().tolist()
        )
        evaluated_maze_groups = sorted(
            results_df["maze_group"].dropna().unique().tolist()
        )

        log_params(
            mlflow=mlflow,
            params={
                "evaluated_policies": ", ".join(evaluated_policies),
                "evaluated_mazes": ", ".join(evaluated_mazes),
                "evaluated_maze_groups": ", ".join(evaluated_maze_groups),
                "run_count": len(results_df),
                "policy_count": len(evaluated_policies),
                "maze_count": len(evaluated_mazes),
            },
        )

        log_metrics(
            mlflow=mlflow,
            metrics={
                "total_runs": len(results_df),
                "unique_policies": len(evaluated_policies),
                "unique_mazes": len(evaluated_mazes),
                "overall_avg_score": results_df["final_score_delta"].mean(),
                "overall_avg_score_per_step": results_df["score_per_step"].mean(),
                "overall_exit_success_rate": results_df["exit_found"].mean(),
            },
        )

        summary_by_bot = (
            results_df.groupby("bot_name", dropna=False)
            .agg(
                avg_score=("final_score_delta", "mean"),
                total_score=("final_score_delta", "sum"),
                avg_steps=("steps_logged", "mean"),
                avg_score_per_step=("score_per_step", "mean"),
                avg_explore_reward=("explore_avg_chosen_reward", "mean"),
                avg_backtrack_ratio=("backtrack_ratio", "mean"),
                avg_first_exit_step=("first_exit_step", "mean"),
                avg_first_collection_step=("first_collection_step", "mean"),
                exit_success_rate=("exit_found", "mean"),
            )
            .reset_index()
        )

        for _, row in summary_by_bot.iterrows():
            bot_name = sanitize_mlflow_key(str(row["bot_name"]))

            log_metrics(
                mlflow=mlflow,
                metrics={
                    f"{bot_name}_avg_score": row["avg_score"],
                    f"{bot_name}_total_score": row["total_score"],
                    f"{bot_name}_avg_steps": row["avg_steps"],
                    f"{bot_name}_avg_score_per_step": row["avg_score_per_step"],
                    f"{bot_name}_avg_explore_reward": row["avg_explore_reward"],
                    f"{bot_name}_avg_backtrack_ratio": row["avg_backtrack_ratio"],
                    f"{bot_name}_avg_first_exit_step": row["avg_first_exit_step"],
                    f"{bot_name}_avg_first_collection_step": row[
                        "avg_first_collection_step"
                    ],
                    f"{bot_name}_exit_success_rate": row["exit_success_rate"],
                },
            )

        log_artifacts(
            mlflow=mlflow,
            artifact_paths=[
                TELEMETRY_OUTPUT_PATH,
                RESULTS_OUTPUT_PATH,
                REPORT_OUTPUT_PATH,
            ],
        )


def main() -> None:
    results: list[EvaluationResult] = []

    if TELEMETRY_OUTPUT_PATH.exists():
        TELEMETRY_OUTPUT_PATH.unlink()

    for evaluation_maze in EVALUATION_MAZES:
        for bot_type in BOT_TYPES:
            print(
                f"Evaluating bot_type={bot_type} "
                f"on maze='{evaluation_maze.name}' "
                f"({evaluation_maze.group})"
            )

            result = evaluate_single_run(
                bot_type=bot_type,
                evaluation_maze=evaluation_maze,
            )

            results.append(result)

            print(
                f"Completed: bot={result.bot_name}, "
                f"maze={result.maze_name}, "
                f"group={result.maze_group}, "
                f"score_delta={result.final_score_delta}, "
                f"steps={result.steps_logged}, "
                f"score_per_step={result.score_per_step:.3f}, "
                f"exit_found={result.exit_found}"
            )

    results_df = write_results_csv(results)
    write_evaluation_report(results_df)
    log_evaluation_to_mlflow(results_df)

    print(f"Evaluation results written to: {RESULTS_OUTPUT_PATH}")
    print(f"Evaluation report written to: {REPORT_OUTPUT_PATH}")
    print(f"Evaluation telemetry written to: {TELEMETRY_OUTPUT_PATH}")
    print("MLflow tracking completed if ENABLE_MLFLOW=true.")


if __name__ == "__main__":
    main()