from pathlib import Path

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


def train_decision_tree(training_df: pd.DataFrame) -> tuple[DecisionTreeClassifier, dict]:
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

    metrics = {
        "training_rows": len(training_df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["not_preferred", "preferred"],
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    return model, metrics


def save_model(model: DecisionTreeClassifier, metrics: dict) -> None:
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
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


def save_training_report(metrics: dict) -> None:
    TRAINING_REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Decision Tree Training Report",
        "",
        "This report summarizes the lightweight supervised model trained from maze telemetry.",
        "",
        "The model is trained on candidate action features. The target label is weakly supervised: for each decision point, the candidate action with the highest transparent preference score is marked as preferred.",
        "",
        "## Training Summary",
        "",
        f"- Training rows: `{metrics['training_rows']}`",
        f"- Train split rows: `{metrics['train_rows']}`",
        f"- Test split rows: `{metrics['test_rows']}`",
        f"- Accuracy: `{metrics['accuracy']:.3f}`",
        "",
        "## Confusion Matrix",
        "",
        "Rows are actual labels and columns are predicted labels.",
        "",
        "```text",
        str(metrics["confusion_matrix"]),
        "```",
        "",
        "## Classification Report",
        "",
        "```text",
        metrics["classification_report"],
        "```",
        "",
        "## Generated Artifacts",
        "",
        "- `models/decision_tree_policy.joblib`",
        "- `reports/decision_tree_policy.txt`",
        "- `reports/decision_tree_policy.png`",
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