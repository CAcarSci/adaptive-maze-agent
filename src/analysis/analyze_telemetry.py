from pathlib import Path

import pandas as pd


INPUT_PATH = Path("experiments/action_logs.csv")
OUTPUT_PATH = Path("reports/telemetry_analysis.md")


BOOLEAN_COLUMNS = [
    "is_chosen",
    "can_collect_score_here",
    "can_exit_maze_here",
    "candidate_has_been_visited",
    "candidate_allows_exit",
    "candidate_allows_score_collection",
    "candidate_is_start",
]

NUMERIC_COLUMNS = [
    "step",
    "current_score_in_hand",
    "current_score_in_bag",
    "current_tile_visit_count",
    "path_depth",
    "available_action_count",
    "candidate_reward_on_destination",
    "candidate_visit_count",
]


def load_telemetry(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Telemetry file not found: {path}. "
            "Run the bot first with `python -m src.main`."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"Telemetry file is empty: {path}. "
            "Run the bot first with `python -m src.main`."
        )

    for column in BOOLEAN_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(
                {
                    True: True,
                    False: False,
                    "True": True,
                    "False": False,
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                    1: True,
                    0: False,
                }
            )

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def format_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"

    formatted_df = df.copy()

    for column in formatted_df.select_dtypes(include=["float"]).columns:
        formatted_df[column] = formatted_df[column].round(3)

    return formatted_df.to_markdown(index=False)


def get_candidate_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["candidate_direction"].notna()].copy()


def get_chosen_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["candidate_direction"].notna())
        & (df["is_chosen"] == True)
    ].copy()


def summarize_overall(df: pd.DataFrame) -> list[str]:
    chosen_df = get_chosen_rows(df)

    lines = [
        "## Overall Summary",
        "",
        f"- Rows logged: `{len(df)}`",
        f"- Unique runs: `{df['run_id'].nunique()}`",
        f"- Mazes observed: `{', '.join(sorted(df['maze_name'].dropna().unique()))}`",
        f"- Bot policies observed: `{', '.join(sorted(df['bot_name'].dropna().unique()))}`",
        f"- Decision steps logged: `{df[['run_id', 'step']].drop_duplicates().shape[0]}`",
        f"- Chosen actions logged: `{len(chosen_df)}`",
        "",
    ]

    return lines


def summarize_runs_by_bot_and_maze(df: pd.DataFrame) -> list[str]:
    grouped = (
        df.groupby(["bot_name", "maze_name"], dropna=False)
        .agg(
            rows=("run_id", "count"),
            unique_runs=("run_id", "nunique"),
            decision_steps=("step", "nunique"),
            max_step=("step", "max"),
            max_score_in_hand=("current_score_in_hand", "max"),
            max_score_in_bag=("current_score_in_bag", "max"),
        )
        .reset_index()
        .sort_values(["bot_name", "maze_name"])
    )

    lines = [
        "## Runs by Bot Policy and Maze",
        "",
        "This section shows which bot policies and mazes are represented in the telemetry dataset.",
        "",
        format_markdown_table(grouped),
        "",
    ]

    return lines


def summarize_reward_distribution(df: pd.DataFrame) -> list[str]:
    candidate_df = get_candidate_rows(df)

    reward_summary = (
        candidate_df["candidate_reward_on_destination"]
        .describe()
        .reset_index()
        .rename(
            columns={
                "index": "metric",
                "candidate_reward_on_destination": "value",
            }
        )
    )

    lines = [
        "## Reward Distribution",
        "",
        "This section looks at the immediate reward available on candidate destination tiles.",
        "",
        format_markdown_table(reward_summary),
        "",
    ]

    return lines


def summarize_policy_reward_comparison(df: pd.DataFrame) -> list[str]:
    candidate_df = get_candidate_rows(df)
    chosen_df = get_chosen_rows(df)

    candidate_grouped = (
        candidate_df.groupby("bot_name", dropna=False)
        .agg(
            candidate_rows=("candidate_direction", "count"),
            avg_available_reward=("candidate_reward_on_destination", "mean"),
            median_available_reward=("candidate_reward_on_destination", "median"),
            max_available_reward=("candidate_reward_on_destination", "max"),
            avg_available_actions=("available_action_count", "mean"),
        )
        .reset_index()
    )

    chosen_grouped = (
        chosen_df.groupby("bot_name", dropna=False)
        .agg(
            chosen_rows=("candidate_direction", "count"),
            avg_chosen_reward=("candidate_reward_on_destination", "mean"),
            median_chosen_reward=("candidate_reward_on_destination", "median"),
            max_chosen_reward=("candidate_reward_on_destination", "max"),
            avg_chosen_visit_count=("candidate_visit_count", "mean"),
        )
        .reset_index()
    )

    comparison = candidate_grouped.merge(
        chosen_grouped,
        on="bot_name",
        how="left",
    )

    comparison["chosen_reward_lift_vs_available_avg"] = (
        comparison["avg_chosen_reward"]
        - comparison["avg_available_reward"]
    )

    lines = [
        "## Policy Reward Comparison",
        "",
        "This compares the reward profile of available candidate actions with the actions actually selected by each bot policy.",
        "",
        format_markdown_table(comparison),
        "",
    ]

    return lines


def summarize_chosen_vs_not_chosen(df: pd.DataFrame) -> list[str]:
    candidate_df = get_candidate_rows(df)

    grouped = (
        candidate_df.groupby(["bot_name", "is_chosen"], dropna=False)
        .agg(
            rows=("candidate_direction", "count"),
            avg_reward=("candidate_reward_on_destination", "mean"),
            median_reward=("candidate_reward_on_destination", "median"),
            max_reward=("candidate_reward_on_destination", "max"),
            avg_candidate_visit_count=("candidate_visit_count", "mean"),
        )
        .reset_index()
        .sort_values(["bot_name", "is_chosen"])
    )

    lines = [
        "## Chosen vs Non-Chosen Candidate Actions",
        "",
        "This compares selected candidate actions with the alternatives that were available at the same decision point, grouped by bot policy.",
        "",
        format_markdown_table(grouped),
        "",
    ]

    return lines


def summarize_by_decision_type(df: pd.DataFrame) -> list[str]:
    candidate_df = get_candidate_rows(df)

    grouped = (
        candidate_df.groupby(["bot_name", "decision_type"], dropna=False)
        .agg(
            rows=("candidate_direction", "count"),
            chosen_rows=("is_chosen", "sum"),
            avg_reward=("candidate_reward_on_destination", "mean"),
            avg_available_actions=("available_action_count", "mean"),
            avg_path_depth=("path_depth", "mean"),
        )
        .reset_index()
        .sort_values(["bot_name", "decision_type"])
    )

    lines = [
        "## Decision Type Summary",
        "",
        "This section summarizes exploration and backtracking behavior by bot policy.",
        "",
        format_markdown_table(grouped),
        "",
    ]

    return lines


def summarize_by_candidate_flags(df: pd.DataFrame) -> list[str]:
    candidate_df = get_candidate_rows(df)

    summaries = []

    for flag in [
        "candidate_has_been_visited",
        "candidate_allows_exit",
        "candidate_allows_score_collection",
        "candidate_is_start",
    ]:
        grouped = (
            candidate_df.groupby(["bot_name", flag], dropna=False)
            .agg(
                rows=("candidate_direction", "count"),
                avg_reward=("candidate_reward_on_destination", "mean"),
                median_reward=("candidate_reward_on_destination", "median"),
                max_reward=("candidate_reward_on_destination", "max"),
            )
            .reset_index()
            .sort_values(["bot_name", flag])
        )

        summaries.extend(
            [
                f"### Reward by `{flag}`",
                "",
                format_markdown_table(grouped),
                "",
            ]
        )

    lines = [
        "## Reward Patterns by Candidate Flags",
        "",
        "This section checks whether immediate rewards differ across candidate tile properties exposed by the API.",
        "",
        *summaries,
    ]

    return lines


def summarize_by_branching_factor(df: pd.DataFrame) -> list[str]:
    candidate_df = get_candidate_rows(df)

    grouped = (
        candidate_df.groupby(["bot_name", "available_action_count"], dropna=False)
        .agg(
            rows=("candidate_direction", "count"),
            avg_reward=("candidate_reward_on_destination", "mean"),
            median_reward=("candidate_reward_on_destination", "median"),
            max_reward=("candidate_reward_on_destination", "max"),
        )
        .reset_index()
        .rename(columns={"available_action_count": "current_tile_available_actions"})
        .sort_values(["bot_name", "current_tile_available_actions"])
    )

    lines = [
        "## Reward by Current Tile Branching Factor",
        "",
        "This is an initial approximation for checking whether rewards differ when the bot is standing on a dead-end, corridor or junction-like tile.",
        "",
        "Important note: `candidate_reward_on_destination` describes the reward on the destination tile, while `available_action_count` describes the current tile. A more precise dead-end/junction analysis will require reconstructing tile-level graph features in a later iteration.",
        "",
        format_markdown_table(grouped),
        "",
    ]

    return lines


def summarize_feature_candidates(df: pd.DataFrame) -> list[str]:
    candidate_df = get_candidate_rows(df)

    feature_columns = [
        "candidate_reward_on_destination",
        "candidate_has_been_visited",
        "candidate_visit_count",
        "candidate_allows_exit",
        "candidate_allows_score_collection",
        "candidate_is_start",
        "available_action_count",
        "path_depth",
    ]

    available_columns = [
        column
        for column in feature_columns
        if column in candidate_df.columns
    ]

    correlations = (
        candidate_df[available_columns]
        .astype(float)
        .corr(numeric_only=True)["candidate_reward_on_destination"]
        .reset_index()
        .rename(
            columns={
                "index": "feature",
                "candidate_reward_on_destination": "correlation_with_reward",
            }
        )
        .sort_values("correlation_with_reward", ascending=False)
    )

    lines = [
        "## Initial Feature Signals",
        "",
        "This table shows simple correlations with immediate destination reward. It is not a final model, but it helps identify candidate features for smarter policies.",
        "",
        format_markdown_table(correlations),
        "",
    ]

    return lines


def summarize_feature_candidates_by_policy(df: pd.DataFrame) -> list[str]:
    candidate_df = get_candidate_rows(df)

    feature_columns = [
        "candidate_reward_on_destination",
        "candidate_has_been_visited",
        "candidate_visit_count",
        "candidate_allows_exit",
        "candidate_allows_score_collection",
        "candidate_is_start",
        "available_action_count",
        "path_depth",
    ]

    policy_summaries = []

    for bot_name, bot_df in candidate_df.groupby("bot_name", dropna=False):
        if len(bot_df) < 2:
            continue

        correlations = (
            bot_df[feature_columns]
            .astype(float)
            .corr(numeric_only=True)["candidate_reward_on_destination"]
            .reset_index()
            .rename(
                columns={
                    "index": "feature",
                    "candidate_reward_on_destination": "correlation_with_reward",
                }
            )
            .sort_values("correlation_with_reward", ascending=False)
        )

        policy_summaries.extend(
            [
                f"### Feature Signals for `{bot_name}`",
                "",
                format_markdown_table(correlations),
                "",
            ]
        )

    lines = [
        "## Initial Feature Signals by Bot Policy",
        "",
        "This section repeats the feature correlation analysis per bot policy. This becomes more useful once telemetry contains both baseline and smart bot runs.",
        "",
        *policy_summaries,
    ]

    return lines


def write_report(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    observed_bots = set(df["bot_name"].dropna().unique())

    if len(observed_bots) > 1:
        conclusion = (
            "The dataset now contains multiple bot policies. This makes it possible "
            "to start comparing policy behavior, especially selected reward profiles, "
            "decision types and revisit-related signals. The next step is to formalize "
            "this into a Step 4 evaluation workflow with consistent run-level metrics."
        )
    else:
        conclusion = (
            "The dataset currently contains telemetry for a single bot policy. "
            "The next step is to run both the baseline and smart bot on comparable "
            "mazes, then use the resulting telemetry for a formal evaluation."
        )

    lines = [
        "# Telemetry Analysis",
        "",
        "This report is generated from `experiments/action_logs.csv`.",
        "",
        "The goal of this analysis is to inspect navigation telemetry and compare behavior between available bot policies.",
        "",
        *summarize_overall(df),
        *summarize_runs_by_bot_and_maze(df),
        *summarize_reward_distribution(df),
        *summarize_policy_reward_comparison(df),
        *summarize_chosen_vs_not_chosen(df),
        *summarize_by_decision_type(df),
        *summarize_by_candidate_flags(df),
        *summarize_by_branching_factor(df),
        *summarize_feature_candidates(df),
        *summarize_feature_candidates_by_policy(df),
        "## Preliminary Conclusion",
        "",
        conclusion,
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_telemetry(INPUT_PATH)
    write_report(df, OUTPUT_PATH)

    print(f"Telemetry rows analyzed: {len(df)}")
    print(f"Bot policies analyzed: {', '.join(sorted(df['bot_name'].dropna().unique()))}")
    print(f"Report written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()