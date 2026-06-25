# Changelog

All notable changes to this project are documented here.

The project follows an incremental AI engineering workflow for the HTI Adaptive Maze Agent challenge.

---

## [Latest] - AI Evaluation Summary

### Added

- Added AI Evaluation Summary powered by a local Llama model through Ollama.
- Added automatic local Llama model check and pull support.
- Added `reports/evaluation_ai_summary.md` as a generated summary artifact.
- Appended the AI Evaluation Summary to `reports/evaluation_report.md` before the `Report Scope` section.
- Added tests for local Llama summary helper logic.
- Updated README with Ollama installation instructions for Windows, macOS and Linux.

### Changed

- Updated evaluation report generation so `Report Scope` remains the final explanatory section.
- Updated displayed report paths in `src/main.py`.

---

## [Step 4] - Policy Evaluation and Tracking

### Added

- Added evaluation workflow for comparing all implemented bot policies:
  - `baseline_dfs`
  - `reward_aware`
  - `decision_tree`
- Added deterministic evaluation report generation.
- Added structured evaluation results output:
  - `reports/evaluation_results.csv`
  - `reports/evaluation_report.md`
- Added separate evaluation telemetry output:
  - `experiments/evaluation_action_logs.csv`
- Added order-sensitive evaluation metrics:
  - score per step
  - early reward progress
  - first collection step
  - first exit-capable tile step
  - backtracking ratio
  - exit success rate
- Added lightweight local MLflow tracking with SQLite backend.
- Added interactive application flow in `src/main.py`.
- Added full pipeline execution from the application entry point.

### Changed

- Improved `src/main.py` from a single-run script into an interactive CLI and pipeline runner.
- Updated README with Step 4 evaluation, MLflow tracking and generated artifacts.
- Updated `.env.example` with MLflow configuration.
- Updated `.gitignore` for local generated tracking files.

---

## [Step 3] - Smart Bot Policies

### Added

- Added policy-based navigation abstraction.
- Added `BaselineDfsPolicy`.
- Added `RewardAwarePolicy`.
- Added `DecisionTreePolicy`.
- Added `SmartMazeBot`.
- Added `DecisionTreeMazeBot`.
- Added candidate-action feature preparation.
- Added Decision Tree training workflow.
- Added Decision Tree explanation artifacts:
  - `reports/decision_tree_policy.txt`
  - `reports/decision_tree_policy.png`
  - `reports/decision_tree_training_report.md`

### Changed

- Refactored bot decision-making so the solving orchestration is separated from the navigation policy.
- Reused the same maze-solving flow across baseline, reward-aware and Decision Tree bots.

---

## [Step 2] - Telemetry and Analysis

### Added

- Added structured telemetry logging for every candidate action at each decision point.
- Added telemetry output:
  - `experiments/action_logs.csv`
- Added telemetry analysis workflow.
- Added telemetry analysis report:
  - `reports/telemetry_analysis.md`

### Changed

- Extended bot execution to record selected and non-selected candidate actions for later analysis.

---

## [Step 1] - Baseline Bot

### Added

- Added Maze API client.
- Added player registration.
- Added maze listing.
- Added maze entry.
- Added deterministic DFS-like baseline bot.
- Added movement, backtracking, score collection and exit handling.
- Added domain models:
  - `MazeState`
  - `MoveAction`
- Added initial unit tests.

### Notes

- The baseline bot successfully solved initial mazes including `Test`, `Easy deal` and `Hello Maze`.