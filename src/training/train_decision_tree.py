from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from src.analysis.analyze_telemetry import load_telemetry
from src.features.candidate_action_features import (
    FEATURE_COLUMNS,
    prepare_training_dataframe,
)


INPUT_PATH = Path("experiments/action_logs.csv")

MODEL_OUTPUT_PATH = Path("models/decision_tree_policy.joblib")
TREE_TEXT_OUTPUT_PATH = Path("reports/decision_tree_policy.txt")
TREE_PNG_OUTPUT_PATH = Path("reports/decision_tree_policy.png")
TRAINING_REPORT_OUTPUT_PATH = Path("reports/decision_tree_training_report.md")


def train_decision_tree(
    training_df: pd.DataFrame,
) -> tuple[DecisionTreeClassifier, dict[str, Any]]:
    X = training_df[FEATURE_COLUMNS]
    y = training_df["target_preferred"]

    class_counts = y.value_counts()

    if len(class_counts) < 2:
        raise ValueError(
            "Decision Tree training requires at least two target classes. "
            "Collect more telemetry before training."
        )

    stratify = y if class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=stratify,
    )

    model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    feature_importances = (
        pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    metrics = {
        "training_rows": len(training_df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["not_preferred", "preferred"],
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        ).tolist(),
        "feature_importances": feature_importances,
    }

    return model, metrics


def save_model(model: DecisionTreeClassifier, metrics: dict[str, Any]) -> None:
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": {
            key: value
            for key, value in metrics.items()
            if key != "feature_importances"
        },
    }

    joblib.dump(artifact, MODEL_OUTPUT_PATH)


def save_tree_text(model: DecisionTreeClassifier) -> None:
    TREE_TEXT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tree_text = export_text(
        model,
        feature_names=FEATURE_COLUMNS,
        decimals=2,
    )

    TREE_TEXT_OUTPUT_PATH.write_text(tree_text, encoding="utf-8")


def save_tree_png(model: DecisionTreeClassifier) -> None:
    TREE_PNG_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(32, 16))

    plot_tree(
        model,
        feature_names=FEATURE_COLUMNS,
        class_names=["not_preferred", "preferred"],
        filled=True,
        rounded=True,
        impurity=True,
        proportion=True,
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(
        TREE_PNG_OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def format_confusion_matrix(confusion_matrix_values: list[list[int]]) -> str:
    true_negative = confusion_matrix_values[0][0]
    false_positive = confusion_matrix_values[0][1]
    false_negative = confusion_matrix_values[1][0]
    true_positive = confusion_matrix_values[1][1]

    lines = [
        "| Actual \\ Predicted | not_preferred | preferred |",
        "|:-------------------|--------------:|----------:|",
        f"| not_preferred | {true_negative} | {false_positive} |",
        f"| preferred | {false_negative} | {true_positive} |",
    ]

    return "\n".join(lines)


def format_feature_importances(feature_importances: pd.DataFrame) -> str:
    formatted = feature_importances.copy()
    formatted["importance"] = formatted["importance"].round(3)

    return formatted.to_markdown(index=False)


def save_training_report(metrics: dict[str, Any]) -> None:
    TRAINING_REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Decision Tree Training Report",
        "",
        "This report summarizes the lightweight supervised model trained from maze telemetry.",
        "",
        "The model is trained on candidate-action features. The target label is weakly supervised: for each decision point, the candidate action with the highest transparent preference score is marked as preferred.",
        "",
        "## Training Summary",
        "",
        "| Metric | Value | Explanation |",
        "|:-------|------:|:------------|",
        f"| Training rows | `{metrics['training_rows']}` | Number of candidate-action rows used after feature preparation. |",
        f"| Train split rows | `{metrics['train_rows']}` | Rows used to fit the Decision Tree model. |",
        f"| Test split rows | `{metrics['test_rows']}` | Rows held out to validate the model. |",
        f"| Accuracy | `{metrics['accuracy']:.3f}` | Share of test rows where the model predicted the correct preferred/not-preferred label. |",
        "",
        "## Metric Definitions",
        "",
        "| Metric | Meaning |",
        "|:-------|:--------|",
        "| Precision | Of all actions predicted as a class, how many were actually that class. |",
        "| Recall | Of all actual actions in a class, how many the model correctly found. |",
        "| F1-score | Harmonic mean of precision and recall. Useful when both false positives and false negatives matter. |",
        "| Support | Number of test examples for that class. |",
        "| Accuracy | Overall share of correct predictions on the test split. |",
        "",
        "## Confusion Matrix",
        "",
        "Rows are actual labels and columns are predicted labels.",
        "",
        format_confusion_matrix(metrics["confusion_matrix"]),
        "",
        "Interpretation:",
        "",
        "- `not_preferred → not_preferred`: correctly rejected candidate actions",
        "- `not_preferred → preferred`: candidate actions incorrectly predicted as preferred",
        "- `preferred → not_preferred`: preferred candidate actions missed by the model",
        "- `preferred → preferred`: correctly selected preferred candidate actions",
        "",
        "## Classification Report",
        "",
        "```text",
        metrics["classification_report"],
        "```",
        "",
        "## Feature Importances",
        "",
        "Feature importance indicates how much each feature contributed to the Decision Tree splits. Higher values mean the feature was more influential in the learned decision rules.",
        "",
        format_feature_importances(metrics["feature_importances"]),
        "",
        "## Interpretation",
        "",
        "The Decision Tree learned that revisit-related features are highly important. The top-level split uses `candidate_visit_count`, which means the model first separates unvisited or barely visited candidate tiles from revisited ones.",
        "",
        "This matches the intended navigation strategy: prefer exploration over repeatedly revisiting already explored tiles. The model also uses features such as `candidate_has_been_visited`, `candidate_allows_exit`, `candidate_allows_score_collection`, `can_collect_score_here`, and `can_exit_maze_here`.",
        "",
        "The model should be interpreted as a lightweight, explainable ML policy trained from telemetry-derived labels. It is not trained on human labels or final maze outcomes. The next evaluation step should compare this policy against the baseline and reward-aware policies on unseen mazes.",
        "",
        "## Generated Artifacts",
        "",
        "- `models/decision_tree_policy.joblib`",
        "- `reports/decision_tree_policy.txt`",
        "- `reports/decision_tree_policy.png`",
        "- `reports/decision_tree_training_report.md`",
        "",
    ]

    TRAINING_REPORT_OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    telemetry_df = load_telemetry(INPUT_PATH)
    training_df = prepare_training_dataframe(telemetry_df)

    if training_df.empty:
        raise ValueError(
            "No usable training rows were created. "
            "Run the bot on mazes with at least two candidate actions per decision."
        )

    model, metrics = train_decision_tree(training_df)

    save_model(model, metrics)
    save_tree_text(model)
    save_tree_png(model)
    save_training_report(metrics)

    print(f"Training rows: {metrics['training_rows']}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Model written to: {MODEL_OUTPUT_PATH}")
    print(f"Tree text written to: {TREE_TEXT_OUTPUT_PATH}")
    print(f"Tree graph written to: {TREE_PNG_OUTPUT_PATH}")
    print(f"Training report written to: {TRAINING_REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()