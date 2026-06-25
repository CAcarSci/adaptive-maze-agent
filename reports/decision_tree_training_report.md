# Decision Tree Training Report

This report summarizes the lightweight supervised model trained from maze telemetry.

The model is trained on candidate-action features. The target label is weakly supervised: for each decision point, the candidate action with the highest transparent preference score is marked as preferred.

## Training Summary

| Metric | Value | Explanation |
|:-------|------:|:------------|
| Telemetry rows before filter | `471` | Total telemetry rows available before training data filtering. |
| Telemetry rows after filter | `313` | Rows kept for training after selecting the configured training policies. |
| Training policies | `baseline_dfs, reward_aware` | Bot policies used as training data. |
| Training rows | `300` | Number of candidate-action rows used after feature preparation. |
| Train split rows | `210` | Rows used to fit the Decision Tree model. |
| Test split rows | `90` | Rows held out to validate the model. |
| Accuracy | `0.911` | Share of test rows where the model predicted the correct preferred/not-preferred label. |

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
| not_preferred | 36 | 5 |
| preferred | 3 | 46 |

## Classification Report

```text
               precision    recall  f1-score   support

not_preferred       0.92      0.88      0.90        41
    preferred       0.90      0.94      0.92        49

     accuracy                           0.91        90
    macro avg       0.91      0.91      0.91        90
 weighted avg       0.91      0.91      0.91        90

```

## Feature Importances

Feature importance indicates how much each feature contributed to the Decision Tree splits.

| feature                           |   importance |
|:----------------------------------|-------------:|
| candidate_visit_count             |        0.802 |
| candidate_has_been_visited        |        0.073 |
| candidate_allows_score_collection |        0.065 |
| can_collect_score_here            |        0.034 |
| candidate_allows_exit             |        0.012 |
| path_depth                        |        0.01  |
| can_exit_maze_here                |        0.003 |
| current_score_in_bag              |        0     |
| current_score_in_hand             |        0     |
| available_action_count            |        0     |
| candidate_reward_on_destination   |        0     |
| candidate_is_start                |        0     |

## Data-Driven Findings

- Training telemetry was filtered from `471` rows to `313` rows; `158` rows were excluded.
- The model was trained using telemetry from: `baseline_dfs, reward_aware`.
- The following observed policies were excluded from training: `decision_tree`.
- Held-out test accuracy was `0.911`.
- The model predicted `82` out of `90` test rows correctly, with `5` false positives and `3` false negatives.
- The highest feature importance belongs to `candidate_visit_count` with importance `0.802`.
- `7` out of `12` features have non-zero importance in this trained tree.
- The root split feature of the trained tree is `candidate_visit_count`.
- Feature importances describe this fitted weakly supervised model only; they should not be interpreted as causal proof of final maze performance.

## Generated Artifacts

- `models/decision_tree_policy.joblib`
- `reports/decision_tree_policy.txt`
- `reports/decision_tree_policy.png`
- `reports/decision_tree_training_report.md`
