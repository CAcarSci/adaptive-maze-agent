# Decision Tree Training Report

This report summarizes the lightweight supervised model trained from maze telemetry.

The model is trained on candidate action features. The target label is weakly supervised: for each decision point, the candidate action with the highest transparent preference score is marked as preferred.

## Training Summary

- Training rows: `300`
- Train split rows: `210`
- Test split rows: `90`
- Accuracy: `0.911`

## Confusion Matrix

Rows are actual labels and columns are predicted labels.

```text
[[36, 5], [3, 46]]
```

## Classification Report

```text
               precision    recall  f1-score   support

not_preferred       0.92      0.88      0.90        41
    preferred       0.90      0.94      0.92        49

     accuracy                           0.91        90
    macro avg       0.91      0.91      0.91        90
 weighted avg       0.91      0.91      0.91        90

```

## Generated Artifacts

- `models/decision_tree_policy.joblib`
- `reports/decision_tree_policy.txt`
- `reports/decision_tree_policy.png`
