from dataclasses import asdict

import pandas as pd

from src.evaluation import evaluate_bots as evaluation


def make_result(
    *,
    bot_type: str,
    bot_name: str,
    maze_name: str,
    final_score_delta: int,
    score_per_step: float | None = None,
    score_progress_at_step_10: int | None = None,
    score_progress_at_step_25: int | None = None,
    score_progress_at_step_50: int | None = None,
    first_exit_step: int | None = 5,
    first_collection_step: int | None = 4,
    backtrack_ratio: float = 0.4,
    exit_found: bool = True,
) -> evaluation.EvaluationResult:
    resolved_score_per_step = (
        score_per_step
        if score_per_step is not None
        else final_score_delta / 10
    )

    return evaluation.EvaluationResult(
        bot_type=bot_type,
        bot_name=bot_name,
        maze_name=maze_name,
        maze_group="seen",
        run_id=f"{bot_name}-{maze_name}",
        exit_found=exit_found,
        final_score_delta=final_score_delta,
        final_player_score=final_score_delta,
        steps_logged=10,
        chosen_actions=9,
        explore_decisions=5,
        backtrack_decisions=4,
        stop_decisions=1,
        backtrack_ratio=backtrack_ratio,
        avg_chosen_reward=3.0,
        explore_avg_chosen_reward=6.0,
        revisit_ratio=0.4,
        explore_revisit_ratio=0.0,
        score_per_step=resolved_score_per_step,
        first_exit_step=first_exit_step,
        first_collection_step=first_collection_step,
        max_score_in_hand=20,
        max_score_in_bag=final_score_delta,
        score_in_bag_at_step_10=final_score_delta,
        score_in_bag_at_step_25=final_score_delta,
        score_in_bag_at_step_50=final_score_delta,
        score_progress_at_step_10=(
            score_progress_at_step_10
            if score_progress_at_step_10 is not None
            else final_score_delta
        ),
        score_progress_at_step_25=(
            score_progress_at_step_25
            if score_progress_at_step_25 is not None
            else final_score_delta
        ),
        score_progress_at_step_50=(
            score_progress_at_step_50
            if score_progress_at_step_50 is not None
            else final_score_delta
        ),
    )


def test_checkpoint_scores_use_bag_and_progress():
    step_df = pd.DataFrame(
        [
            {
                "step": 0,
                "current_score_in_bag": 0,
                "current_score_in_hand": 0,
            },
            {
                "step": 5,
                "current_score_in_bag": 10,
                "current_score_in_hand": 20,
            },
            {
                "step": 15,
                "current_score_in_bag": 30,
                "current_score_in_hand": 5,
            },
        ]
    )

    score_in_bag, score_progress = evaluation.get_checkpoint_scores(
        step_df=step_df,
        checkpoint=10,
    )

    assert score_in_bag == 10
    assert score_progress == 30


def test_final_score_finding_detects_identical_scores():
    results_df = pd.DataFrame(
        [
            asdict(
                make_result(
                    bot_type="baseline",
                    bot_name="baseline_dfs",
                    maze_name="Example Maze",
                    final_score_delta=104,
                )
            ),
            asdict(
                make_result(
                    bot_type="smart",
                    bot_name="reward_aware",
                    maze_name="Example Maze",
                    final_score_delta=104,
                )
            ),
            asdict(
                make_result(
                    bot_type="decision_tree",
                    bot_name="decision_tree",
                    maze_name="Example Maze",
                    final_score_delta=104,
                )
            ),
        ]
    )

    finding = evaluation.create_final_score_interpretation(results_df)

    assert "Observed result" in finding
    assert "same final score" in finding


def test_final_score_finding_detects_best_average_score():
    results_df = pd.DataFrame(
        [
            asdict(
                make_result(
                    bot_type="baseline",
                    bot_name="baseline_dfs",
                    maze_name="Example Maze",
                    final_score_delta=100,
                )
            ),
            asdict(
                make_result(
                    bot_type="smart",
                    bot_name="reward_aware",
                    maze_name="Example Maze",
                    final_score_delta=110,
                )
            ),
        ]
    )

    finding = evaluation.create_final_score_interpretation(results_df)

    assert "Observed result" in finding
    assert "reward_aware" in finding
    assert "highest average final score" in finding
    assert "`110`" in finding


def test_create_data_driven_findings_uses_metric_leaders():
    results_df = pd.DataFrame(
        [
            asdict(
                make_result(
                    bot_type="baseline",
                    bot_name="baseline_dfs",
                    maze_name="Example Maze",
                    final_score_delta=100,
                    score_per_step=10.0,
                    score_progress_at_step_10=40,
                    score_progress_at_step_25=70,
                    score_progress_at_step_50=100,
                    first_collection_step=7,
                    first_exit_step=6,
                    backtrack_ratio=0.5,
                )
            ),
            asdict(
                make_result(
                    bot_type="smart",
                    bot_name="reward_aware",
                    maze_name="Example Maze",
                    final_score_delta=110,
                    score_per_step=11.0,
                    score_progress_at_step_10=50,
                    score_progress_at_step_25=90,
                    score_progress_at_step_50=110,
                    first_collection_step=3,
                    first_exit_step=4,
                    backtrack_ratio=0.3,
                )
            ),
            asdict(
                make_result(
                    bot_type="decision_tree",
                    bot_name="decision_tree",
                    maze_name="Example Maze",
                    final_score_delta=105,
                    score_per_step=10.5,
                    score_progress_at_step_10=45,
                    score_progress_at_step_25=80,
                    score_progress_at_step_50=105,
                    first_collection_step=5,
                    first_exit_step=5,
                    backtrack_ratio=0.4,
                )
            ),
        ]
    )

    findings = evaluation.create_data_driven_findings(results_df)
    joined_findings = "\n".join(findings)

    assert "reward_aware" in joined_findings
    assert "Highest average score per logged step" in joined_findings
    assert "Highest average reward progress by step 25" in joined_findings
    assert "Lowest average first collection step" in joined_findings
    assert "Lowest average backtrack ratio" in joined_findings


def test_write_evaluation_report_creates_markdown_file(tmp_path, monkeypatch):
    report_path = tmp_path / "evaluation_report.md"

    monkeypatch.setattr(evaluation, "REPORT_OUTPUT_PATH", report_path)

    results = [
        make_result(
            bot_type="baseline",
            bot_name="baseline_dfs",
            maze_name="Example Maze",
            final_score_delta=104,
        ),
        make_result(
            bot_type="smart",
            bot_name="reward_aware",
            maze_name="Example Maze",
            final_score_delta=104,
        ),
        make_result(
            bot_type="decision_tree",
            bot_name="decision_tree",
            maze_name="Example Maze",
            final_score_delta=104,
        ),
    ]

    results_df = pd.DataFrame([asdict(result) for result in results])

    evaluation.write_evaluation_report(results_df)

    report = report_path.read_text(encoding="utf-8")

    assert "# Bot Evaluation Report" in report
    assert "## Final Score Finding" in report
    assert "## Data-Driven Findings" in report
    assert "## Early Reward Checkpoints" in report
    assert "## Report Scope" in report
    assert "This report is generated deterministically from evaluation results" in report
    assert "baseline_dfs" in report
    assert "reward_aware" in report
    assert "decision_tree" in report