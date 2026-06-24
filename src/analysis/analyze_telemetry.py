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
                }
            )

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def format_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"

    return df.to_markdown(index=False)


def summarize_overall(df: pd.DataFrame) -> list[str]:
    chosen_df = df[df["is_chosen"] == True].copy()

    lines = [
        "## Overall Summary",
        "",
        f"- Rows logged: `{len(df)}`",
        f"- Unique runs: `{df['run_id'].nunique()}`",
        f"- Mazes observed: `{', '.join(sorted(df['maze_name'].dropna().unique()))}`",
        f"- Bot types observed: `{', '.join(sorted(df['bot_name'].dropna().unique()))}`",
        f"- Decision steps logged: `{df[['run_id', 'step']].drop_duplicates().shape[0]}`",
        f"- Chosen actions logged: `{len(chosen_df)}`",
        "",
    ]

    return lines


def summarize_reward_distribution(df: pd.DataFrame) -> list[str]:
    candidate_df = df[df["candidate_direction"].notna()].copy()

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


def summarize_chosen_vs_not_chosen(df: pd.DataFrame) -> list[str]:
    candidate_df = df[df["candidate_direction"].notna()].copy()

    grouped = (
        candidate_df.groupby("is_chosen", dropna=False)
        .agg(
            rows=("candidate_direction", "count"),
            avg_reward=("candidate_reward_on_destination", "mean"),
            median_reward=("candidate_reward_on_destination", "median"),
            max_reward=("candidate_reward_on_destination", "max"),
            avg_candidate_visit_count=("candidate_visit_count", "mean"),
        )
        .reset_index()
    )

    lines = [
        "## Chosen vs Non-Chosen Candidate Actions",
        "",
        "This compares the candidate actions selected by the baseline bot with the alternatives that were available at the same decision point.",
        "",
        format_markdown_table(grouped),
        "",
    ]

    return lines


def summarize_by_decision_type(df: pd.DataFrame) -> list[str]:
    candidate_df = df[df["candidate_direction"].notna()].copy()

    grouped = (
        candidate_df.groupby("decision_type", dropna=False)
        .agg(
            rows=("candidate_direction", "count"),
            chosen_rows=("is_chosen", "sum"),
            avg_reward=("candidate_reward_on_destination", "mean"),
            avg_available_actions=("available_action_count", "mean"),
            avg_path_depth=("path_depth", "mean"),
        )
        .reset_index()
    )

    lines = [
        "## Decision Type Summary",
        "",
        "The baseline bot currently makes three types of decisions: `explore`, `backtrack` and `stop`.",
        "",
        format_markdown_table(grouped),
        "",
    ]

    return lines


def summarize_by_candidate_flags(df: pd.DataFrame) -> list[str]:
    candidate_df = df[df["candidate_direction"].notna()].copy()

    summaries = []

    for flag in [
        "candidate_has_been_visited",
        "candidate_allows_exit",
        "candidate_allows_score_collection",
        "candidate_is_start",
    ]:
        grouped = (
            candidate_df.groupby(flag, dropna=False)
            .agg(
                rows=("candidate_direction", "count"),
                avg_reward=("candidate_reward_on_destination", "mean"),
                median_reward=("candidate_reward_on_destination", "median"),
                max_reward=("candidate_reward_on_destination", "max"),
            )
            .reset_index()
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
    candidate_df = df[df["candidate_direction"].notna()].copy()

    grouped = (
        candidate_df.groupby("available_action_count", dropna=False)
        .agg(
            rows=("candidate_direction", "count"),
            avg_reward=("candidate_reward_on_destination", "mean"),
            median_reward=("candidate_reward_on_destination", "median"),
            max_reward=("candidate_reward_on_destination", "max"),
        )
        .reset_index()
        .rename(columns={"available_action_count": "current_tile_available_actions"})
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
    candidate_df = df[df["candidate_direction"].notna()].copy()

    correlations = (
        candidate_df[
            [
                "candidate_reward_on_destination",
                "candidate_has_been_visited",
                "candidate_visit_count",
                "candidate_allows_exit",
                "candidate_allows_score_collection",
                "candidate_is_start",
                "available_action_count",
                "path_depth",
            ]
        ]
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
        "This table shows simple correlations with immediate destination reward. It is not a final model, but it helps identify candidate features for a smarter policy.",
        "",
        format_markdown_table(correlations),
        "",
    ]

    return lines


def write_report(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Telemetry Analysis",
        "",
        "This report is generated from `experiments/action_logs.csv`.",
        "",
        "The goal of this analysis is to inspect the data collected during baseline maze navigation before implementing a smarter bot.",
        "",
        *summarize_overall(df),
        *summarize_reward_distribution(df),
        *summarize_chosen_vs_not_chosen(df),
        *summarize_by_decision_type(df),
        *summarize_by_candidate_flags(df),
        *summarize_by_branching_factor(df),
        *summarize_feature_candidates(df),
        "## Preliminary Conclusion",
        "",
        "At this stage, the dataset is still small and collected only from baseline runs. Therefore, conclusions should be treated as exploratory.",
        "",
        "The next step is to run the baseline bot on several mazes, collect more telemetry, and then use the observed reward and navigation patterns to design a simple smarter policy.",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_telemetry(INPUT_PATH)
    write_report(df, OUTPUT_PATH)

    print(f"Telemetry rows analyzed: {len(df)}")
    print(f"Report written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()