# Bot Evaluation Report

This report compares all implemented navigation policies on the same maze set.

## Evaluated Policies

- `baseline_dfs`: deterministic DFS-style baseline policy
- `reward_aware`: explainable heuristic policy based on telemetry insights
- `decision_tree`: lightweight ML policy trained from telemetry-derived labels

## Evaluation Mazes

- `Example Maze` (seen)
- `Gradius Pathways` (seen)
- `Hello Maze` (seen)
- `Exit` (unseen)
- `O Contra` (unseen)
- `Dig Down` (unseen)
- `Glasses` (unseen)
- `Reverse` (unseen)
- `Loops` (unseen)

## Metric Definitions

| Metric | Meaning |
|:-------|:--------|
| final_score_delta | Score gained during the evaluated run. Calculated from player score before and after the run. |
| steps_logged | Number of decision steps logged during exploration. |
| score_per_step | Final score divided by the number of logged decision steps. |
| explore_avg_chosen_reward | Average immediate reward selected during forward exploration decisions only. |
| explore_revisit_ratio | Share of forward exploration decisions that selected an already visited destination tile. |
| backtrack_ratio | Share of decision steps that were backtracking decisions. |
| first_exit_step | First logged step where the bot was standing on an exit-capable tile. |
| first_collection_step | First logged step where the bot was standing on a score collection tile. |
| score_in_bag_at_step_N | Score already secured in the bag by step N. |
| score_progress_at_step_N | Score in bag plus score in hand by step N. |
| exit_success_rate | Fraction of runs where the bot successfully exited the maze. |

## Final Score Finding

Observed result: all policies achieved the same final score on each evaluated maze.

## Data-Driven Findings

- Observed result: all policies achieved the same final score on each evaluated maze.
- Highest average score per logged step: `baseline_dfs`, `decision_tree` and `reward_aware` (`4.031`).
- Highest average reward progress by step 10: `decision_tree` (`81.111`).
- Highest average reward progress by step 25: `decision_tree` (`144.444`).
- Highest average reward progress by step 50: `baseline_dfs`, `decision_tree` and `reward_aware` (`169.778`).
- Lowest average first collection step: `reward_aware` (`9.111`).
- Lowest average first exit-capable tile step: `baseline_dfs` (`13.778`).
- Lowest average backtrack ratio: `baseline_dfs`, `decision_tree` and `reward_aware` (`0.485`).
- Highest average exit success rate: `baseline_dfs`, `decision_tree` and `reward_aware` (`1`).

## Summary by Bot Policy

| bot_name      |   runs |   avg_score |   total_score |   avg_steps |   avg_score_per_step |   avg_explore_reward |   avg_explore_revisit_ratio |   avg_backtrack_ratio |   avg_first_exit_step |   avg_first_collection_step |   exit_success_rate |
|:--------------|-------:|------------:|--------------:|------------:|---------------------:|---------------------:|----------------------------:|----------------------:|----------------------:|----------------------------:|--------------------:|
| baseline_dfs  |      9 |     169.778 |          1528 |          41 |                4.031 |                8.307 |                           0 |                 0.485 |                13.778 |                      14.667 |                   1 |
| decision_tree |      9 |     169.778 |          1528 |          41 |                4.031 |                8.307 |                           0 |                 0.485 |                16.889 |                      15.778 |                   1 |
| reward_aware  |      9 |     169.778 |          1528 |          41 |                4.031 |                8.307 |                           0 |                 0.485 |                14     |                       9.111 |                   1 |

## Summary by Maze Group and Bot Policy

| maze_group   | bot_name      |   runs |   avg_score |   avg_steps |   avg_score_per_step |   avg_explore_reward |   avg_explore_revisit_ratio |   exit_success_rate |
|:-------------|:--------------|-------:|------------:|------------:|---------------------:|---------------------:|----------------------------:|--------------------:|
| seen         | baseline_dfs  |      3 |     129.333 |          33 |                3.722 |                7.736 |                           0 |                   1 |
| seen         | decision_tree |      3 |     129.333 |          33 |                3.722 |                7.736 |                           0 |                   1 |
| seen         | reward_aware  |      3 |     129.333 |          33 |                3.722 |                7.736 |                           0 |                   1 |
| unseen       | baseline_dfs  |      6 |     190     |          45 |                4.186 |                8.592 |                           0 |                   1 |
| unseen       | decision_tree |      6 |     190     |          45 |                4.186 |                8.592 |                           0 |                   1 |
| unseen       | reward_aware  |      6 |     190     |          45 |                4.186 |                8.592 |                           0 |                   1 |

## Early Reward Checkpoints

This table shows whether a policy finds or secures reward earlier during exploration.

| bot_name      |   avg_score_in_bag_at_step_10 |   avg_score_in_bag_at_step_25 |   avg_score_in_bag_at_step_50 |   avg_score_progress_at_step_10 |   avg_score_progress_at_step_25 |   avg_score_progress_at_step_50 |
|:--------------|------------------------------:|------------------------------:|------------------------------:|--------------------------------:|--------------------------------:|--------------------------------:|
| baseline_dfs  |                        18.556 |                        82.333 |                       145.333 |                          79     |                         141.333 |                         169.778 |
| decision_tree |                        18.556 |                        83.222 |                       145.333 |                          81.111 |                         144.444 |                         169.778 |
| reward_aware  |                        31     |                        80.222 |                       118.444 |                          74.778 |                         138.222 |                         169.778 |

## Results by Maze and Bot

| maze_group   | maze_name        | bot_name      |   final_score_delta |   steps_logged |   score_per_step |   explore_avg_chosen_reward |   explore_revisit_ratio |   backtrack_ratio |   first_exit_step |   first_collection_step |   score_progress_at_step_10 |   score_progress_at_step_25 |   score_progress_at_step_50 | exit_found   |
|:-------------|:-----------------|:--------------|--------------------:|---------------:|-----------------:|----------------------------:|------------------------:|------------------:|------------------:|------------------------:|----------------------------:|----------------------------:|----------------------------:|:-------------|
| seen         | Example Maze     | baseline_dfs  |                 104 |             33 |            3.152 |                       6.5   |                       0 |             0.485 |                 8 |                       6 |                          62 |                          94 |                         104 | True         |
| seen         | Example Maze     | decision_tree |                 104 |             33 |            3.152 |                       6.5   |                       0 |             0.485 |                 8 |                       6 |                          62 |                          94 |                         104 | True         |
| seen         | Example Maze     | reward_aware  |                 104 |             33 |            3.152 |                       6.5   |                       0 |             0.485 |                 8 |                       6 |                          62 |                          94 |                         104 | True         |
| seen         | Gradius Pathways | baseline_dfs  |                 232 |             51 |            4.549 |                       9.28  |                       0 |             0.49  |                 8 |                      11 |                          91 |                         182 |                         232 | True         |
| seen         | Gradius Pathways | decision_tree |                 232 |             51 |            4.549 |                       9.28  |                       0 |             0.49  |                 8 |                      11 |                          91 |                         182 |                         232 | True         |
| seen         | Gradius Pathways | reward_aware  |                 232 |             51 |            4.549 |                       9.28  |                       0 |             0.49  |                 8 |                      11 |                          91 |                         182 |                         232 | True         |
| seen         | Hello Maze       | baseline_dfs  |                  52 |             15 |            3.467 |                       7.429 |                       0 |             0.467 |                 6 |                       8 |                          52 |                          52 |                          52 | True         |
| seen         | Hello Maze       | decision_tree |                  52 |             15 |            3.467 |                       7.429 |                       0 |             0.467 |                 6 |                       8 |                          52 |                          52 |                          52 | True         |
| seen         | Hello Maze       | reward_aware  |                  52 |             15 |            3.467 |                       7.429 |                       0 |             0.467 |                 6 |                       8 |                          52 |                          52 |                          52 | True         |
| unseen       | Dig Down         | baseline_dfs  |                 232 |             51 |            4.549 |                       9.28  |                       0 |             0.49  |                32 |                      31 |                          90 |                         180 |                         232 | True         |
| unseen       | Dig Down         | decision_tree |                 232 |             51 |            4.549 |                       9.28  |                       0 |             0.49  |                32 |                      31 |                          90 |                         180 |                         232 | True         |
| unseen       | Dig Down         | reward_aware  |                 232 |             51 |            4.549 |                       9.28  |                       0 |             0.49  |                30 |                      29 |                          90 |                         180 |                         232 | True         |
| unseen       | Exit             | baseline_dfs  |                  82 |             21 |            3.905 |                       8.2   |                       0 |             0.476 |                 2 |                       1 |                          62 |                          82 |                          82 | True         |
| unseen       | Exit             | decision_tree |                  82 |             21 |            3.905 |                       8.2   |                       0 |             0.476 |                 2 |                       1 |                          62 |                          82 |                          82 | True         |
| unseen       | Exit             | reward_aware  |                  82 |             21 |            3.905 |                       8.2   |                       0 |             0.476 |                 2 |                       1 |                          62 |                          82 |                          82 | True         |
| unseen       | Glasses          | baseline_dfs  |                 272 |             59 |            4.61  |                       9.379 |                       0 |             0.492 |                16 |                      37 |                         100 |                         211 |                         272 | True         |
| unseen       | Glasses          | decision_tree |                 272 |             59 |            4.61  |                       9.379 |                       0 |             0.492 |                16 |                      37 |                         100 |                         211 |                         272 | True         |
| unseen       | Glasses          | reward_aware  |                 272 |             59 |            4.61  |                       9.379 |                       0 |             0.492 |                18 |                       9 |                          81 |                         192 |                         272 | True         |
| unseen       | Loops            | baseline_dfs  |                 122 |             29 |            4.207 |                       8.714 |                       0 |             0.483 |                 7 |                      24 |                          81 |                         122 |                         122 | True         |
| unseen       | Loops            | decision_tree |                 122 |             29 |            4.207 |                       8.714 |                       0 |             0.483 |                19 |                      24 |                         100 |                         122 |                         122 | True         |
| unseen       | Loops            | reward_aware  |                 122 |             29 |            4.207 |                       8.714 |                       0 |             0.483 |                 9 |                       4 |                          62 |                         122 |                         122 | True         |
| unseen       | O Contra         | baseline_dfs  |                 196 |             51 |            3.843 |                       7.84  |                       0 |             0.49  |                15 |                       2 |                          73 |                         146 |                         196 | True         |
| unseen       | O Contra         | decision_tree |                 196 |             51 |            3.843 |                       7.84  |                       0 |             0.49  |                31 |                       2 |                          73 |                         165 |                         196 | True         |
| unseen       | O Contra         | reward_aware  |                 196 |             51 |            3.843 |                       7.84  |                       0 |             0.49  |                15 |                       2 |                          73 |                         146 |                         196 | True         |
| unseen       | Reverse          | baseline_dfs  |                 236 |             59 |            4     |                       8.138 |                       0 |             0.492 |                30 |                      12 |                         100 |                         203 |                         236 | True         |
| unseen       | Reverse          | decision_tree |                 236 |             59 |            4     |                       8.138 |                       0 |             0.492 |                30 |                      22 |                         100 |                         212 |                         236 | True         |
| unseen       | Reverse          | reward_aware  |                 236 |             59 |            4     |                       8.138 |                       0 |             0.492 |                30 |                      12 |                         100 |                         194 |                         236 | True         |

## Report Scope

This report is generated deterministically from evaluation results.
