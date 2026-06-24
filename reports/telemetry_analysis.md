# Telemetry Analysis

This report is generated from `experiments/action_logs.csv`.

The goal of this analysis is to inspect the data collected during baseline maze navigation before implementing a smarter bot.

## Overall Summary

- Rows logged: `29`
- Unique runs: `1`
- Mazes observed: `Hello Maze`
- Bot types observed: `baseline_dfs`
- Decision steps logged: `15`
- Chosen actions logged: `14`

## Reward Distribution

This section looks at the immediate reward available on candidate destination tiles.

| metric   |    value |
|:---------|---------:|
| count    | 29       |
| mean     |  2.13793 |
| std      |  4.09463 |
| min      |  0       |
| 25%      |  0       |
| 50%      |  0       |
| 75%      |  1       |
| max      | 10       |

## Chosen vs Non-Chosen Candidate Actions

This compares the candidate actions selected by the baseline bot with the alternatives that were available at the same decision point.

| is_chosen   |   rows |   avg_reward |   median_reward |   max_reward |   avg_candidate_visit_count |
|:------------|-------:|-------------:|----------------:|-------------:|----------------------------:|
| False       |     15 |     0.666667 |             0   |           10 |                    1.4      |
| True        |     14 |     3.71429  |             0.5 |           10 |                    0.571429 |

## Decision Type Summary

The baseline bot currently makes three types of decisions: `explore`, `backtrack` and `stop`.

| decision_type   |   rows |   chosen_rows |   avg_reward |   avg_available_actions |   avg_path_depth |
|:----------------|-------:|--------------:|-------------:|------------------------:|-----------------:|
| backtrack       |     13 |             7 |      0       |                 2.07692 |          3.15385 |
| explore         |     15 |             7 |      4.13333 |                 2.33333 |          2.53333 |
| stop            |      1 |             0 |      0       |                 1       |          0       |

## Reward Patterns by Candidate Flags

This section checks whether immediate rewards differ across candidate tile properties exposed by the API.

### Reward by `candidate_has_been_visited`

| candidate_has_been_visited   |   rows |   avg_reward |   median_reward |   max_reward |
|:-----------------------------|-------:|-------------:|----------------:|-------------:|
| False                        |      8 |         7.75 |              10 |           10 |
| True                         |     21 |         0    |               0 |            0 |

### Reward by `candidate_allows_exit`

| candidate_allows_exit   |   rows |   avg_reward |   median_reward |   max_reward |
|:------------------------|-------:|-------------:|----------------:|-------------:|
| False                   |     25 |         2.44 |               0 |           10 |
| True                    |      4 |         0.25 |               0 |            1 |

### Reward by `candidate_allows_score_collection`

| candidate_allows_score_collection   |   rows |   avg_reward |   median_reward |   max_reward |
|:------------------------------------|-------:|-------------:|----------------:|-------------:|
| False                               |     27 |      2.25926 |             0   |           10 |
| True                                |      2 |      0.5     |             0.5 |            1 |

### Reward by `candidate_is_start`

| candidate_is_start   |   rows |   avg_reward |   median_reward |   max_reward |
|:---------------------|-------:|-------------:|----------------:|-------------:|
| False                |     27 |       2.2963 |               0 |           10 |
| True                 |      2 |       0      |               0 |            0 |

## Reward by Current Tile Branching Factor

This is an initial approximation for checking whether rewards differ when the bot is standing on a dead-end, corridor or junction-like tile.

Important note: `candidate_reward_on_destination` describes the reward on the destination tile, while `available_action_count` describes the current tile. A more precise dead-end/junction analysis will require reconstructing tile-level graph features in a later iteration.

|   current_tile_available_actions |   rows |   avg_reward |   median_reward |   max_reward |
|---------------------------------:|-------:|-------------:|----------------:|-------------:|
|                                1 |      4 |      2.5     |               0 |           10 |
|                                2 |     16 |      1.375   |               0 |           10 |
|                                3 |      9 |      3.33333 |               0 |           10 |

## Initial Feature Signals

This table shows simple correlations with immediate destination reward. It is not a final model, but it helps identify candidate features for a smarter policy.

| feature                           |   correlation_with_reward |
|:----------------------------------|--------------------------:|
| candidate_reward_on_destination   |                  1        |
| available_action_count            |                  0.123339 |
| candidate_allows_score_collection |                 -0.110798 |
| candidate_is_start                |                 -0.144621 |
| candidate_allows_exit             |                 -0.187694 |
| path_depth                        |                 -0.281182 |
| candidate_visit_count             |                 -0.674468 |
| candidate_has_been_visited        |                 -0.860921 |

## Preliminary Conclusion

At this stage, the dataset is still small and collected only from baseline runs. Therefore, conclusions should be treated as exploratory.

The next step is to run the baseline bot on several mazes, collect more telemetry, and then use the observed reward and navigation patterns to design a simple smarter policy.
