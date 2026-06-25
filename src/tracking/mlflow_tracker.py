import math
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def is_mlflow_enabled() -> bool:
    value = os.getenv("ENABLE_MLFLOW", "true")

    return value.strip().lower() in {"1", "true", "yes", "y"}


def sanitize_mlflow_key(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    sanitized = sanitized.strip("_")

    return sanitized or "unknown"


def get_configured_mlflow() -> Any | None:
    if not is_mlflow_enabled():
        return None

    try:
        import mlflow
    except ImportError:
        print("MLflow is enabled but not installed. Skipping MLflow tracking.")
        return None

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "adaptive-maze-agent")

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    except Exception as error:
        print("MLflow tracking could not be configured. Skipping MLflow tracking.")
        print(f"Reason: {error}")
        return None

    return mlflow


@contextmanager
def mlflow_run(
    *,
    run_name: str,
    tags: dict[str, str] | None = None,
) -> Iterator[Any | None]:
    mlflow = get_configured_mlflow()

    if mlflow is None:
        yield None
        return

    run_context = None

    try:
        run_context = mlflow.start_run(run_name=run_name)
        run_context.__enter__()

        if tags:
            mlflow.set_tags(tags)

        yield mlflow

    except Exception as error:
        print("MLflow run failed. Continuing without MLflow tracking.")
        print(f"Reason: {error}")
        yield None

    finally:
        if run_context is not None:
            try:
                run_context.__exit__(None, None, None)
            except Exception as error:
                print("MLflow run could not be closed cleanly.")
                print(f"Reason: {error}")


def log_params(
    *,
    mlflow: Any | None,
    params: dict[str, Any],
) -> None:
    if mlflow is None:
        return

    for key, value in params.items():
        if value is None:
            continue

        try:
            mlflow.log_param(
                sanitize_mlflow_key(key),
                str(value),
            )
        except Exception as error:
            print(f"Skipping MLflow param '{key}': {error}")


def log_metrics(
    *,
    mlflow: Any | None,
    metrics: dict[str, float | int | bool | None],
) -> None:
    if mlflow is None:
        return

    for key, value in metrics.items():
        if value is None:
            continue

        try:
            metric_value = float(value)

            if not math.isfinite(metric_value):
                continue

            mlflow.log_metric(
                sanitize_mlflow_key(key),
                metric_value,
            )
        except Exception as error:
            print(f"Skipping MLflow metric '{key}': {error}")


def log_artifacts(
    *,
    mlflow: Any | None,
    artifact_paths: list[Path],
) -> None:
    if mlflow is None:
        return

    for artifact_path in artifact_paths:
        if not artifact_path.exists():
            continue

        try:
            mlflow.log_artifact(str(artifact_path))
        except Exception as error:
            print(f"Skipping MLflow artifact '{artifact_path}': {error}")