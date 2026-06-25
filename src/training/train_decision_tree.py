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
TRAINING_BOT_NAMES = ["baseline_dfs", "reward_aware"]

MODEL_OUTPUT_PATH = Path("models/decision_tree_policy.joblib")
TREE_TEXT_OUTPUT_PATH = Path("reports/decision_tree_policy.txt")
TREE_PNG_OUTPUT_PATH = Path("reports/decision_tree_policy.png")
TRAINING_REPORT_OUTPUT_PATH = Path("reports/decision_tree_training_report.md")


def filter_training_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters telemetry used for Decision Tree training.

    The Decision Tree policy is excluded from training telemetry to avoid
    training the model on decisions produced by a previous version of itself.
    """

    if "bot_name" not in df.columns:
        raise ValueError("Telemetry must contain a `bot_name` column.")

    filtered_df = df[df["bot_name"].isin(TRAINING_BOT_NAMES)].copy()

    if filtered_df.empty:
        raise ValueError(
            "No usable training telemetry found. Expected telemetry from: "
            f"{', '.join(TRAINING_BOT_NAMES)}."
        )

    return filtered_df


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
        "root_split_feature": get_root_split_feature(model),
    }

    return model, metrics


def get_root_split_feature(model: DecisionTreeClassifier) -> str | None:
    tree = getattr(model, "tree_", None)

    if tree is None:
        return None

    feature_indices = getattr(tree, "feature", None)

    if feature_indices is None or len(feature_indices) == 0:
        return None

    root_feature_index = int(feature_indices[0])

    if root_feature_index < 0:
        return None

    if root_feature_index >= len(FEATURE_COLUMNS):
        return None

    return FEATURE_COLUMNS[root_feature_index]


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


def format_optional_list(values: list[str] | None) -> str:
    if not values:
        return "none"

    return ", ".join(values)


def create_training_findings(metrics: dict[str, Any]) -> list[str]:
    findings: list[str] = []

    telemetry_rows_before_filter = metrics.get("telemetry_rows_before_filter")
    telemetry_rows_after_filter = metrics.get("telemetry_rows_after_filter")
    training_bot_names = metrics.get("training_bot_names", [])

    if telemetry_rows_before_filter is not None and telemetry_rows_after_filter is not None:
        excluded_rows = telemetry_rows_before_filter - telemetry_rows_after_filter
        findings.append(
            f"Training telemetry was filtered from `{telemetry_rows_before_filter}` rows "
            f"to `{telemetry_rows_after_filter}` rows; `{excluded_rows}` rows were excluded."
        )

    if training_bot_names:
        findings.append(
            f"The model was trained using telemetry from: `{format_optional_list(training_bot_names)}`."
        )

    observed_bot_names = metrics.get("observed_bot_names_before_filter", [])
    excluded_bot_names = [
        bot_name
        for bot_name in observed_bot_names
        if bot_name not in training_bot_names
    ]

    if excluded_bot_names:
        findings.append(
            f"The following observed policies were excluded from training: "
            f"`{format_optional_list(excluded_bot_names)}`."
        )

    accuracy = metrics.get("accuracy")
    if accuracy is not None:
        findings.append(
            f"Held-out test accuracy was `{accuracy:.3f}`."
        )

    confusion_matrix_values = metrics.get("confusion_matrix")
    if confusion_matrix_values is not None:
        true_negative = confusion_matrix_values[0][0]
        false_positive = confusion_matrix_values[0][1]
        false_negative = confusion_matrix_values[1][0]
        true_positive = confusion_matrix_values[1][1]
        total_predictions = true_negative + false_positive + false_negative + true_positive
        correct_predictions = true_negative + true_positive

        findings.append(
            f"The model predicted `{correct_predictions}` out of `{total_predictions}` "
            f"test rows correctly, with `{false_positive}` false positives and "
            f"`{false_negative}` false negatives."
        )

    feature_importances = metrics.get("feature_importances")
    if feature_importances is not None and not feature_importances.empty:
        top_feature_row = feature_importances.iloc[0]
        top_feature = top_feature_row["feature"]
        top_importance = float(top_feature_row["importance"])

        findings.append(
            f"The highest feature importance belongs to `{top_feature}` "
            f"with importance `{top_importance:.3f}`."
        )

        non_zero_importance_count = int(
            (feature_importances["importance"] > 0).sum()
        )
        findings.append(
            f"`{non_zero_importance_count}` out of `{len(feature_importances)}` "
            "features have non-zero importance in this trained tree."
        )

    root_split_feature = metrics.get("root_split_feature")
    if root_split_feature is not None:
        findings.append(
            f"The root split feature of the trained tree is `{root_split_feature}`."
        )

    findings.append(
        "Feature importances describe this fitted weakly supervised model only; "
        "they should not be interpreted as causal proof of final maze performance."
    )

    return findings


def save_training_report(metrics: dict[str, Any]) -> None:
    TRAINING_REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    training_findings = create_training_findings(metrics)

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
        f"| Telemetry rows before filter | `{metrics['telemetry_rows_before_filter']}` | Total telemetry rows available before training data filtering. |",
        f"| Telemetry rows after filter | `{metrics['telemetry_rows_after_filter']}` | Rows kept for training after selecting the configured training policies. |",
        f"| Training policies | `{format_optional_list(metrics['training_bot_names'])}` | Bot policies used as training data. |",
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
        "## Classification Report",
        "",
        "```text",
        metrics["classification_report"],
        "```",
        "",
        "## Feature Importances",
        "",
        "Feature importance indicates how much each feature contributed to the Decision Tree splits.",
        "",
        format_feature_importances(metrics["feature_importances"]),
        "",
        "## Data-Driven Findings",
        "",
        *[f"- {finding}" for finding in training_findings],
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
    raw_telemetry_df = load_telemetry(INPUT_PATH)

    observed_bot_names_before_filter = sorted(
        raw_telemetry_df["bot_name"].dropna().unique().tolist()
    )

    telemetry_df = filter_training_telemetry(raw_telemetry_df)
    training_df = prepare_training_dataframe(telemetry_df)

    if training_df.empty:
        raise ValueError(
            "No usable training rows were created. "
            "Run the bot on mazes with at least two candidate actions per decision."
        )

    model, metrics = train_decision_tree(training_df)

    metrics["training_bot_names"] = TRAINING_BOT_NAMES
    metrics["observed_bot_names_before_filter"] = observed_bot_names_before_filter
    metrics["telemetry_rows_before_filter"] = len(raw_telemetry_df)
    metrics["telemetry_rows_after_filter"] = len(telemetry_df)

    save_model(model, metrics)
    save_tree_text(model)
    save_tree_png(model)
    save_training_report(metrics)

    print(f"Telemetry rows before filter: {metrics['telemetry_rows_before_filter']}")
    print(f"Telemetry rows after filter: {metrics['telemetry_rows_after_filter']}")
    print(f"Training bot names: {', '.join(TRAINING_BOT_NAMES)}")
    print(f"Training rows: {metrics['training_rows']}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Model written to: {MODEL_OUTPUT_PATH}")
    print(f"Tree text written to: {TREE_TEXT_OUTPUT_PATH}")
    print(f"Tree graph written to: {TREE_PNG_OUTPUT_PATH}")
    print(f"Training report written to: {TRAINING_REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()