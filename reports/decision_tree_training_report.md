# Decision Tree Training Report

This report summarizes the lightweight supervised model trained from maze telemetry.

The model is trained on candidate-action features. The target label is weakly supervised: for each decision point, the candidate action with the highest transparent preference score is marked as preferred.

## Training Summary

| Metric | Value | Explanation |
|:-------|------:|:------------|
| Training rows | `454` | Number of candidate-action rows used after feature preparation. |
| Train split rows | `317` | Rows used to fit the Decision Tree model. |
| Test split rows | `137` | Rows held out to validate the model. |
| Accuracy | `0.861` | Share of test rows where the model predicted the correct preferred/not-preferred label. |

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
| not_preferred | 52 | 10 |
| preferred | 9 | 66 |

Interpretation:

- `not_preferred → not_preferred`: correctly rejected candidate actions
- `not_preferred → preferred`: candidate actions incorrectly predicted as preferred
- `preferred → not_preferred`: preferred candidate actions missed by the model
- `preferred → preferred`: correctly selected preferred candidate actions

## Classification Report

```text
               precision    recall  f1-score   support

not_preferred       0.85      0.84      0.85        62
    preferred       0.87      0.88      0.87        75

     accuracy                           0.86       137
    macro avg       0.86      0.86      0.86       137
 weighted avg       0.86      0.86      0.86       137

```

## Feature Importances

Feature importance indicates how much each feature contributed to the Decision Tree splits. Higher values mean the feature was more influential in the learned decision rules.

| feature                           |   importance |
|:----------------------------------|-------------:|
| candidate_visit_count             |        0.858 |
| candidate_has_been_visited        |        0.082 |
| candidate_allows_score_collection |        0.039 |
| can_collect_score_here            |        0.01  |
| path_depth                        |        0.009 |
| can_exit_maze_here                |        0.002 |
| candidate_allows_exit             |        0     |
| current_score_in_bag              |        0     |
| current_score_in_hand             |        0     |
| available_action_count            |        0     |
| candidate_reward_on_destination   |        0     |
| candidate_is_start                |        0     |

## Interpretation

The Decision Tree learned that revisit-related features are highly important. The top-level split uses `candidate_visit_count`, which means the model first separates unvisited or barely visited candidate tiles from revisited ones.

This matches the intended navigation strategy: prefer exploration over repeatedly revisiting already explored tiles. The model also uses features such as `candidate_has_been_visited`, `candidate_allows_exit`, `candidate_allows_score_collection`, `can_collect_score_here`, and `can_exit_maze_here`.

The model should be interpreted as a lightweight, explainable ML policy trained from telemetry-derived labels. It is not trained on human labels or final maze outcomes. The next evaluation step should compare this policy against the baseline and reward-aware policies on unseen mazes.

## Generated Artifacts

- `models/decision_tree_policy.joblib`
- `reports/decision_tree_policy.txt`
- `reports/decision_tree_policy.png`
- `reports/decision_tree_training_report.md`
