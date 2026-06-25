# Decision Tree Training Report

This report summarizes the lightweight supervised model trained from maze telemetry.

The model is trained on candidate-action features. The target label is weakly supervised: for each decision point, the candidate action with the highest transparent preference score is marked as preferred.

## Training Summary

| Metric | Value | Explanation |
|:-------|------:|:------------|
| Telemetry rows before filter | `468` | Total telemetry rows available before training data filtering. |
| Telemetry rows after filter | `468` | Rows kept for training after selecting the configured training policies. |
| Training policies | `baseline_dfs, reward_aware` | Bot policies used as training data. |
| Training rows | `446` | Number of candidate-action rows used after feature preparation. |
| Train split rows | `312` | Rows used to fit the Decision Tree model. |
| Test split rows | `134` | Rows held out to validate the model. |
| Accuracy | `0.888` | Share of test rows where the model predicted the correct preferred/not-preferred label. |

## Metric Definitions

| Metric | Meaning |
|:-------|:--------|
| Precision | Of all actions predicted as a class, how many were actually that class. |
| Recall | Of all actual actions in a class, how many the model correctly found. |
| F1-score | Harmonic mean of precision and recall. Useful when both false positives and false negatives matter. |
| Support | Number of test examples for that class. |
| Accuracy | Overall share of correct predictions on the test split. |

## Confusion Matrix

Rows are actual labels and columns are predicted labels.

| Actual \ Predicted | not_preferred | preferred |
|:-------------------|--------------:|----------:|
| not_preferred | 55 | 7 |
| preferred | 8 | 64 |

## Classification Report

```text
               precision    recall  f1-score   support

not_preferred       0.87      0.89      0.88        62
    preferred       0.90      0.89      0.90        72

     accuracy                           0.89       134
    macro avg       0.89      0.89      0.89       134
 weighted avg       0.89      0.89      0.89       134

```

## Feature Importances

Feature importance indicates how much each feature contributed to the Decision Tree splits.

| feature                           |   importance |
|:----------------------------------|-------------:|
| candidate_visit_count             |        0.85  |
| candidate_allows_score_collection |        0.066 |
| candidate_has_been_visited        |        0.065 |
| path_depth                        |        0.012 |
| can_collect_score_here            |        0.006 |
| can_exit_maze_here                |        0.002 |
| current_score_in_hand             |        0     |
| current_score_in_bag              |        0     |
| candidate_reward_on_destination   |        0     |
| available_action_count            |        0     |
| candidate_allows_exit             |        0     |
| candidate_is_start                |        0     |

## Data-Driven Findings

- Training telemetry was filtered from `468` rows to `468` rows; `0` rows were excluded.
- The model was trained using telemetry from: `baseline_dfs, reward_aware`.
- Held-out test accuracy was `0.888`.
- The model predicted `119` out of `134` test rows correctly, with `7` false positives and `8` false negatives.
- The highest feature importance belongs to `candidate_visit_count` with importance `0.850`.
- `6` out of `12` features have non-zero importance in this trained tree.
- The root split feature of the trained tree is `candidate_visit_count`.
- Feature importances describe this fitted weakly supervised model only; they should not be interpreted as causal proof of final maze performance.

## Generated Artifacts

- `models/decision_tree_policy.joblib`
- `reports/decision_tree_policy.txt`
- `reports/decision_tree_policy.png`
- `reports/decision_tree_training_report.md`
