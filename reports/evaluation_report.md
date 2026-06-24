# Bot Evaluation Report

This report compares all implemented navigation policies on the same maze set.

The goal is to evaluate whether the smarter policies improve behavior compared with the deterministic DFS baseline.

## Evaluated Policies

- `baseline_dfs`: deterministic DFS-style baseline policy
- `reward_aware`: explainable heuristic policy based on telemetry insights
- `decision_tree`: lightweight ML policy trained from telemetry-derived labels

## Evaluation Mazes

- `Example Maze`
- `Gradius Pathways`
- `Hello Maze`

## Metric Definitions

| Metric | Meaning |
|:-------|:--------|
| final_score_delta | Score gained during the evaluated run. Calculated from player score before and after the run. |
| steps_logged | Number of decision steps logged during exploration. |
| avg_chosen_reward | Average immediate reward on candidate actions selected by the policy. |
| revisit_ratio | Share of chosen actions that moved to already visited destination tiles. Lower is usually better. |
| exit_success_rate | Fraction of runs where the bot successfully exited the maze. |

## Summary by Bot Policy

| bot_name      |   runs |   avg_score |   total_score |   avg_steps |   avg_chosen_reward |   avg_revisit_ratio |   exit_success_rate |
|:--------------|-------:|------------:|--------------:|------------:|--------------------:|--------------------:|--------------------:|
| baseline_dfs  |      3 |     129.333 |           388 |          33 |               3.868 |                 0.5 |                   1 |
| decision_tree |      3 |     129.333 |           388 |          33 |               3.868 |                 0.5 |                   1 |
| reward_aware  |      3 |     129.333 |           388 |          33 |               3.868 |                 0.5 |                   1 |

## Results by Maze and Bot

| maze_name        | bot_name      |   final_score_delta |   steps_logged |   avg_chosen_reward |   revisit_ratio | exit_found   |
|:-----------------|:--------------|--------------------:|---------------:|--------------------:|----------------:|:-------------|
| Example Maze     | baseline_dfs  |                 104 |             33 |               3.25  |             0.5 | True         |
| Example Maze     | decision_tree |                 104 |             33 |               3.25  |             0.5 | True         |
| Example Maze     | reward_aware  |                 104 |             33 |               3.25  |             0.5 | True         |
| Gradius Pathways | baseline_dfs  |                 232 |             51 |               4.64  |             0.5 | True         |
| Gradius Pathways | decision_tree |                 232 |             51 |               4.64  |             0.5 | True         |
| Gradius Pathways | reward_aware  |                 232 |             51 |               4.64  |             0.5 | True         |
| Hello Maze       | baseline_dfs  |                  52 |             15 |               3.714 |             0.5 | True         |
| Hello Maze       | decision_tree |                  52 |             15 |               3.714 |             0.5 | True         |
| Hello Maze       | reward_aware  |                  52 |             15 |               3.714 |             0.5 | True         |

## Preliminary Interpretation

This evaluation is intentionally lightweight. It focuses on run-level behavior rather than only individual decision rows.

The most important comparison is whether the smarter policies achieve similar or higher score with fewer revisits and reasonable step counts compared with the baseline.

The Decision Tree policy should be interpreted carefully because it is trained from telemetry-derived labels, not from human labels or final maze outcomes. Its value is primarily explainability and testing whether structured candidate-action features can learn a useful preference policy.
