# Telemetry Analysis

This report is generated from `experiments/action_logs.csv`.

The goal of this analysis is to inspect navigation telemetry and compare behavior between available bot policies.

## Overall Summary

- Rows logged: `468`
- Unique runs: `6`
- Mazes observed: `Example Maze, Gradius Pathways, Hello Maze`
- Bot policies observed: `baseline_dfs, reward_aware`
- Decision steps logged: `198`
- Chosen actions logged: `192`

## Runs by Bot Policy and Maze

This section shows which bot policies and mazes are represented in the telemetry dataset.

| bot_name     | maze_name        |   rows |   unique_runs |   decision_steps |   max_step |   max_score_in_hand |   max_score_in_bag |
|:-------------|:-----------------|-------:|--------------:|-----------------:|-----------:|--------------------:|-------------------:|
| baseline_dfs | Example Maze     |     79 |             1 |               33 |         32 |                  52 |                 94 |
| baseline_dfs | Gradius Pathways |    126 |             1 |               51 |         50 |                 140 |                 92 |
| baseline_dfs | Hello Maze       |     29 |             1 |               15 |         14 |                  51 |                 52 |
| reward_aware | Example Maze     |     79 |             1 |               33 |         32 |                  52 |                 94 |
| reward_aware | Gradius Pathways |    126 |             1 |               51 |         50 |                 140 |                 92 |
| reward_aware | Hello Maze       |     29 |             1 |               15 |         14 |                  51 |                 52 |

## Reward Distribution

This section looks at the immediate reward available on candidate destination tiles.

| metric   |   value |
|:---------|--------:|
| count    | 468     |
| mean     |   2.214 |
| std      |   4.119 |
| min      |   0     |
| 25%      |   0     |
| 50%      |   0     |
| 75%      |   1     |
| max      |  10     |

## Policy Reward Comparison

This compares the reward profile of available candidate actions with the actions actually selected by each bot policy.

| bot_name     |   candidate_rows |   avg_available_reward |   median_available_reward |   max_available_reward |   avg_available_actions |   chosen_rows |   avg_chosen_reward |   median_chosen_reward |   max_chosen_reward |   avg_chosen_visit_count |   chosen_reward_lift_vs_available_avg |
|:-------------|-----------------:|-----------------------:|--------------------------:|-----------------------:|------------------------:|--------------:|--------------------:|-----------------------:|--------------------:|-------------------------:|--------------------------------------:|
| baseline_dfs |              234 |                  2.214 |                         0 |                     10 |                    2.65 |            96 |               4.042 |                      0 |                  10 |                    0.573 |                                 1.828 |
| reward_aware |              234 |                  2.214 |                         0 |                     10 |                    2.65 |            96 |               4.042 |                      0 |                  10 |                    0.573 |                                 1.828 |

## Chosen vs Non-Chosen Candidate Actions

This compares selected candidate actions with the alternatives that were available at the same decision point, grouped by bot policy.

| bot_name     | is_chosen   |   rows |   avg_reward |   median_reward |   max_reward |   avg_candidate_visit_count |
|:-------------|:------------|-------:|-------------:|----------------:|-------------:|----------------------------:|
| baseline_dfs | False       |    138 |        0.942 |               0 |           10 |                       1.333 |
| baseline_dfs | True        |     96 |        4.042 |               0 |           10 |                       0.573 |
| reward_aware | False       |    138 |        0.942 |               0 |           10 |                       1.333 |
| reward_aware | True        |     96 |        4.042 |               0 |           10 |                       0.573 |

## Decision Type Summary

This section summarizes exploration and backtracking behavior by bot policy.

| bot_name     | decision_type   |   rows |   chosen_rows |   avg_reward |   avg_available_actions |   avg_path_depth |
|:-------------|:----------------|-------:|--------------:|-------------:|------------------------:|-----------------:|
| baseline_dfs | backtrack       |    108 |            48 |        0     |                   2.537 |            8.056 |
| baseline_dfs | explore         |    122 |            48 |        4.246 |                   2.787 |            7.557 |
| baseline_dfs | stop            |      4 |             0 |        0     |                   1.5   |            0     |
| reward_aware | backtrack       |    108 |            48 |        0     |                   2.537 |            8.056 |
| reward_aware | explore         |    122 |            48 |        4.246 |                   2.787 |            7.557 |
| reward_aware | stop            |      4 |             0 |        0     |                   1.5   |            0     |

## Reward Patterns by Candidate Flags

This section checks whether immediate rewards differ across candidate tile properties exposed by the API.

### Reward by `candidate_has_been_visited`

| bot_name     | candidate_has_been_visited   |   rows |   avg_reward |   median_reward |   max_reward |
|:-------------|:-----------------------------|-------:|-------------:|----------------:|-------------:|
| baseline_dfs | False                        |     64 |        8.094 |              10 |           10 |
| baseline_dfs | True                         |    170 |        0     |               0 |            0 |
| reward_aware | False                        |     64 |        8.094 |              10 |           10 |
| reward_aware | True                         |    170 |        0     |               0 |            0 |

### Reward by `candidate_allows_exit`

| bot_name     | candidate_allows_exit   |   rows |   avg_reward |   median_reward |   max_reward |
|:-------------|:------------------------|-------:|-------------:|----------------:|-------------:|
| baseline_dfs | False                   |    215 |        2.391 |               0 |           10 |
| baseline_dfs | True                    |     19 |        0.211 |               0 |            1 |
| reward_aware | False                   |    215 |        2.391 |               0 |           10 |
| reward_aware | True                    |     19 |        0.211 |               0 |            1 |

### Reward by `candidate_allows_score_collection`

| bot_name     | candidate_allows_score_collection   |   rows |   avg_reward |   median_reward |   max_reward |
|:-------------|:------------------------------------|-------:|-------------:|----------------:|-------------:|
| baseline_dfs | False                               |    223 |        2.305 |               0 |           10 |
| baseline_dfs | True                                |     11 |        0.364 |               0 |            1 |
| reward_aware | False                               |    223 |        2.305 |               0 |           10 |
| reward_aware | True                                |     11 |        0.364 |               0 |            1 |

### Reward by `candidate_is_start`

| bot_name     | candidate_is_start   |   rows |   avg_reward |   median_reward |   max_reward |
|:-------------|:---------------------|-------:|-------------:|----------------:|-------------:|
| baseline_dfs | False                |    227 |        2.282 |               0 |           10 |
| baseline_dfs | True                 |      7 |        0     |               0 |            0 |
| reward_aware | False                |    227 |        2.282 |               0 |           10 |
| reward_aware | True                 |      7 |        0     |               0 |            0 |

## Reward by Current Tile Branching Factor

This is an initial approximation for checking whether rewards differ when the bot is standing on a dead-end, corridor or junction-like tile.

Important note: `candidate_reward_on_destination` describes the reward on the destination tile, while `available_action_count` describes the current tile. A more precise dead-end/junction analysis will require reconstructing tile-level graph features in a later iteration.

| bot_name     |   current_tile_available_actions |   rows |   avg_reward |   median_reward |   max_reward |
|:-------------|---------------------------------:|-------:|-------------:|----------------:|-------------:|
| baseline_dfs |                                1 |     11 |        1.818 |               0 |           10 |
| baseline_dfs |                                2 |    104 |        1.615 |               0 |           10 |
| baseline_dfs |                                3 |     75 |        2.933 |               0 |           10 |
| baseline_dfs |                                4 |     44 |        2.5   |               0 |           10 |
| reward_aware |                                1 |     11 |        1.818 |               0 |           10 |
| reward_aware |                                2 |    104 |        1.615 |               0 |           10 |
| reward_aware |                                3 |     75 |        2.933 |               0 |           10 |
| reward_aware |                                4 |     44 |        2.5   |               0 |           10 |

## Initial Feature Signals

This table shows simple correlations with immediate destination reward. It is not a final model, but it helps identify candidate features for smarter policies.

| feature                           |   correlation_with_reward |
|:----------------------------------|--------------------------:|
| candidate_reward_on_destination   |                     1     |
| available_action_count            |                     0.104 |
| path_depth                        |                    -0.078 |
| candidate_is_start                |                    -0.094 |
| candidate_allows_score_collection |                    -0.1   |
| candidate_allows_exit             |                    -0.145 |
| candidate_visit_count             |                    -0.689 |
| candidate_has_been_visited        |                    -0.877 |

## Initial Feature Signals by Bot Policy

This section repeats the feature correlation analysis per bot policy. This becomes more useful once telemetry contains both baseline and smart bot runs.

### Feature Signals for `baseline_dfs`

| feature                           |   correlation_with_reward |
|:----------------------------------|--------------------------:|
| candidate_reward_on_destination   |                     1     |
| available_action_count            |                     0.104 |
| path_depth                        |                    -0.078 |
| candidate_is_start                |                    -0.094 |
| candidate_allows_score_collection |                    -0.1   |
| candidate_allows_exit             |                    -0.145 |
| candidate_visit_count             |                    -0.689 |
| candidate_has_been_visited        |                    -0.877 |

### Feature Signals for `reward_aware`

| feature                           |   correlation_with_reward |
|:----------------------------------|--------------------------:|
| candidate_reward_on_destination   |                     1     |
| available_action_count            |                     0.104 |
| path_depth                        |                    -0.078 |
| candidate_is_start                |                    -0.094 |
| candidate_allows_score_collection |                    -0.1   |
| candidate_allows_exit             |                    -0.145 |
| candidate_visit_count             |                    -0.689 |
| candidate_has_been_visited        |                    -0.877 |

## Preliminary Conclusion

The dataset now contains multiple bot policies. This makes it possible to start comparing policy behavior, especially selected reward profiles, decision types and revisit-related signals. The next step is to formalize this into a Step 4 evaluation workflow with consistent run-level metrics.
