# Telemetry Analysis

This report is generated from `experiments/action_logs.csv`.

The goal of this analysis is to inspect the data collected during baseline maze navigation before implementing a smarter bot.

## Overall Summary

- Rows logged: `234`
- Unique runs: `3`
- Mazes observed: `Example Maze, Gradius Pathways, Hello Maze`
- Bot types observed: `baseline_dfs`
- Decision steps logged: `99`
- Chosen actions logged: `96`

## Reward Distribution

This section looks at the immediate reward available on candidate destination tiles.

| metric   |     value |
|:---------|----------:|
| count    | 234       |
| mean     |   2.21368 |
| std      |   4.12327 |
| min      |   0       |
| 25%      |   0       |
| 50%      |   0       |
| 75%      |   0.75    |
| max      |  10       |

## Chosen vs Non-Chosen Candidate Actions

This compares the candidate actions selected by the baseline bot with the alternatives that were available at the same decision point.

| is_chosen   |   rows |   avg_reward |   median_reward |   max_reward |   avg_candidate_visit_count |
|:------------|-------:|-------------:|----------------:|-------------:|----------------------------:|
| False       |    138 |     0.942029 |               0 |           10 |                    1.33333  |
| True        |     96 |     4.04167  |               0 |           10 |                    0.572917 |

## Decision Type Summary

The baseline bot currently makes three types of decisions: `explore`, `backtrack` and `stop`.

| decision_type   |   rows |   chosen_rows |   avg_reward |   avg_available_actions |   avg_path_depth |
|:----------------|-------:|--------------:|-------------:|------------------------:|-----------------:|
| backtrack       |    108 |            48 |       0      |                 2.53704 |          8.05556 |
| explore         |    122 |            48 |       4.2459 |                 2.78689 |          7.55738 |
| stop            |      4 |             0 |       0      |                 1.5     |          0       |

## Reward Patterns by Candidate Flags

This section checks whether immediate rewards differ across candidate tile properties exposed by the API.

### Reward by `candidate_has_been_visited`

| candidate_has_been_visited   |   rows |   avg_reward |   median_reward |   max_reward |
|:-----------------------------|-------:|-------------:|----------------:|-------------:|
| False                        |     64 |      8.09375 |              10 |           10 |
| True                         |    170 |      0       |               0 |            0 |

### Reward by `candidate_allows_exit`

| candidate_allows_exit   |   rows |   avg_reward |   median_reward |   max_reward |
|:------------------------|-------:|-------------:|----------------:|-------------:|
| False                   |    215 |     2.3907   |               0 |           10 |
| True                    |     19 |     0.210526 |               0 |            1 |

### Reward by `candidate_allows_score_collection`

| candidate_allows_score_collection   |   rows |   avg_reward |   median_reward |   max_reward |
|:------------------------------------|-------:|-------------:|----------------:|-------------:|
| False                               |    223 |     2.30493  |               0 |           10 |
| True                                |     11 |     0.363636 |               0 |            1 |

### Reward by `candidate_is_start`

| candidate_is_start   |   rows |   avg_reward |   median_reward |   max_reward |
|:---------------------|-------:|-------------:|----------------:|-------------:|
| False                |    227 |      2.28194 |               0 |           10 |
| True                 |      7 |      0       |               0 |            0 |

## Reward by Current Tile Branching Factor

This is an initial approximation for checking whether rewards differ when the bot is standing on a dead-end, corridor or junction-like tile.

Important note: `candidate_reward_on_destination` describes the reward on the destination tile, while `available_action_count` describes the current tile. A more precise dead-end/junction analysis will require reconstructing tile-level graph features in a later iteration.

|   current_tile_available_actions |   rows |   avg_reward |   median_reward |   max_reward |
|---------------------------------:|-------:|-------------:|----------------:|-------------:|
|                                1 |     11 |      1.81818 |               0 |           10 |
|                                2 |    104 |      1.61538 |               0 |           10 |
|                                3 |     75 |      2.93333 |               0 |           10 |
|                                4 |     44 |      2.5     |               0 |           10 |

## Initial Feature Signals

This table shows simple correlations with immediate destination reward. It is not a final model, but it helps identify candidate features for a smarter policy.

| feature                           |   correlation_with_reward |
|:----------------------------------|--------------------------:|
| candidate_reward_on_destination   |                 1         |
| available_action_count            |                 0.103856  |
| path_depth                        |                -0.0776656 |
| candidate_is_start                |                -0.0944796 |
| candidate_allows_score_collection |                -0.0998649 |
| candidate_allows_exit             |                -0.14473   |
| candidate_visit_count             |                -0.688898  |
| candidate_has_been_visited        |                -0.876873  |

## Preliminary Conclusion

At this stage, the dataset is still small and collected only from baseline runs. Therefore, conclusions should be treated as exploratory.

The next step is to run the baseline bot on several mazes, collect more telemetry, and then use the observed reward and navigation patterns to design a simple smarter policy.
