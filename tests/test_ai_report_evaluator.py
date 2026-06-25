import pandas as pd

from src.evaluation.ai_report_evaluator import (
    build_llama_context,
    build_policy_summary,
    is_local_ollama_endpoint,
    normalize_two_paragraph_summary,
)


def make_results_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": "baseline-Example Maze",
                "bot_name": "baseline_dfs",
                "maze_name": "Example Maze",
                "maze_group": "seen",
                "final_score_delta": 100,
                "score_per_step": 5.0,
                "steps_logged": 20,
                "backtrack_ratio": 0.5,
                "first_exit_step": 10,
                "first_collection_step": 8,
                "exit_found": True,
            },
            {
                "run_id": "smart-Example Maze",
                "bot_name": "reward_aware",
                "maze_name": "Example Maze",
                "maze_group": "seen",
                "final_score_delta": 110,
                "score_per_step": 7.0,
                "steps_logged": 16,
                "backtrack_ratio": 0.3,
                "first_exit_step": 7,
                "first_collection_step": 4,
                "exit_found": True,
            },
        ]
    )


def test_build_policy_summary_contains_expected_columns():
    results_df = make_results_df()

    summary = build_policy_summary(results_df)

    assert "bot_name" in summary.columns
    assert "avg_final_score" in summary.columns
    assert "avg_score_per_step" in summary.columns
    assert "exit_success_rate" in summary.columns


def test_build_llama_context_contains_policy_names():
    results_df = make_results_df()

    context = build_llama_context(results_df)

    assert "baseline_dfs" in context
    assert "reward_aware" in context
    assert "Policy summary" in context


def test_normalize_two_paragraph_summary_keeps_only_two_paragraphs():
    summary = "Paragraph one.\n\nParagraph two.\n\nParagraph three."

    normalized = normalize_two_paragraph_summary(summary)

    assert normalized == "Paragraph one.\n\nParagraph two."


def test_normalize_two_paragraph_summary_adds_second_paragraph_when_missing():
    summary = "Only one paragraph."

    normalized = normalize_two_paragraph_summary(summary)

    assert "Only one paragraph." in normalized
    assert "\n\n" in normalized
    assert "deterministic evaluation metrics" in normalized


def test_is_local_ollama_endpoint_detects_localhost():
    assert is_local_ollama_endpoint("http://localhost:11434/v1") is True
    assert is_local_ollama_endpoint("http://127.0.0.1:11434/v1") is True


def test_is_local_ollama_endpoint_detects_remote_host():
    assert is_local_ollama_endpoint("https://example.com/v1") is False