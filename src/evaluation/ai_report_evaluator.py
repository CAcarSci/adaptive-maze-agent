import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests


AI_SUMMARY_OUTPUT_PATH = Path("reports/evaluation_ai_summary.md")


@dataclass(frozen=True)
class LocalLlamaEvaluationConfig:
    enabled: bool
    auto_pull_model: bool
    base_url: str
    api_key: str | None
    model: str
    timeout_seconds: int
    model_pull_timeout_seconds: int


def parse_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "y"}


def load_local_llama_config() -> LocalLlamaEvaluationConfig:
    return LocalLlamaEvaluationConfig(
        enabled=parse_bool(os.getenv("ENABLE_LLAMA_EVALUATION"), default=True),
        auto_pull_model=parse_bool(os.getenv("LLAMA_AUTO_PULL_MODEL"), default=True),
        base_url=os.getenv(
            "LLAMA_EVALUATION_BASE_URL",
            "http://localhost:11434/v1",
        ).strip().rstrip("/"),
        api_key=os.getenv("LLAMA_EVALUATION_API_KEY", "ollama").strip() or None,
        model=os.getenv("LLAMA_EVALUATION_MODEL", "llama3.2:1b").strip(),
        timeout_seconds=int(os.getenv("LLAMA_EVALUATION_TIMEOUT_SECONDS", "60")),
        model_pull_timeout_seconds=int(
            os.getenv("LLAMA_MODEL_PULL_TIMEOUT_SECONDS", "600")
        ),
    )


def is_local_ollama_endpoint(base_url: str) -> bool:
    parsed_url = urlparse(base_url)
    hostname = parsed_url.hostname

    return hostname in {"localhost", "127.0.0.1", "::1"}


def is_ollama_cli_available() -> bool:
    return shutil.which("ollama") is not None


def is_ollama_model_available(model: str) -> bool:
    try:
        result = subprocess.run(
            ["ollama", "show", model],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except Exception:
        return False

    return result.returncode == 0


def pull_ollama_model(
    *,
    model: str,
    timeout_seconds: int,
) -> bool:
    print(f"Local Llama model '{model}' was not found. Pulling model with Ollama...")

    try:
        result = subprocess.run(
            ["ollama", "pull", model],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except Exception as error:
        print(f"Could not pull local Llama model: {model}")
        print(f"Reason: {error}")
        return False

    if result.returncode == 0:
        print(f"Local Llama model is ready: {model}")
        return True

    print(f"Could not pull local Llama model: {model}")
    print(result.stderr.strip() or result.stdout.strip())

    return False


def ensure_local_llama_model_available(config: LocalLlamaEvaluationConfig) -> bool:
    if not is_local_ollama_endpoint(config.base_url):
        return True

    if not is_ollama_cli_available():
        print("Ollama CLI is not available. Skipping Local Llama summary.")
        print("Install Ollama locally and run the evaluation again.")
        return False

    if is_ollama_model_available(config.model):
        return True

    if not config.auto_pull_model:
        print(
            f"Local Llama model '{config.model}' is missing and "
            "LLAMA_AUTO_PULL_MODEL=false."
        )
        return False

    return pull_ollama_model(
        model=config.model,
        timeout_seconds=config.model_pull_timeout_seconds,
    )


def build_policy_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        results_df.groupby("bot_name", dropna=False)
        .agg(
            run_count=("run_id", "count"),
            avg_final_score=("final_score_delta", "mean"),
            total_final_score=("final_score_delta", "sum"),
            avg_score_per_step=("score_per_step", "mean"),
            avg_steps_logged=("steps_logged", "mean"),
            avg_backtrack_ratio=("backtrack_ratio", "mean"),
            avg_first_exit_step=("first_exit_step", "mean"),
            avg_first_collection_step=("first_collection_step", "mean"),
            exit_success_rate=("exit_found", "mean"),
        )
        .reset_index()
        .round(3)
    )


def build_policy_group_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        results_df.groupby(["maze_group", "bot_name"], dropna=False)
        .agg(
            run_count=("run_id", "count"),
            avg_final_score=("final_score_delta", "mean"),
            avg_score_per_step=("score_per_step", "mean"),
            avg_backtrack_ratio=("backtrack_ratio", "mean"),
            exit_success_rate=("exit_found", "mean"),
        )
        .reset_index()
        .round(3)
    )


def build_llama_context(results_df: pd.DataFrame) -> str:
    policy_summary = build_policy_summary(results_df)
    policy_group_summary = build_policy_group_summary(results_df)

    evaluated_policies = sorted(
        results_df["bot_name"].dropna().unique().tolist()
    )
    evaluated_mazes = sorted(
        results_df["maze_name"].dropna().unique().tolist()
    )
    evaluated_groups = sorted(
        results_df["maze_group"].dropna().unique().tolist()
    )

    return "\n\n".join(
        [
            "Evaluation metadata:",
            f"- Evaluated policies: {', '.join(evaluated_policies)}",
            f"- Evaluated mazes: {', '.join(evaluated_mazes)}",
            f"- Maze groups: {', '.join(evaluated_groups)}",
            f"- Total runs: {len(results_df)}",
            "",
            "Policy summary:",
            policy_summary.to_markdown(index=False),
            "",
            "Policy summary by maze group:",
            policy_group_summary.to_markdown(index=False),
        ]
    )


def build_llama_messages(results_df: pd.DataFrame) -> list[dict[str, str]]:
    context = build_llama_context(results_df)

    system_message = (
        "You are a local Llama evaluation assistant. "
        "You summarize AI/ML policy evaluation results for a technical reviewer. "
        "Use only the data provided by the user. "
        "Do not invent numbers, causes or unsupported claims. "
        "Write exactly two short paragraphs. "
        "Do not use bullet points, headings or markdown tables. "
        "Paragraph 1 should summarize the main result. "
        "Paragraph 2 should explain the evaluation trade-off or limitation."
    )

    user_message = (
        "Summarize the following maze bot evaluation results in exactly two paragraphs.\n\n"
        f"{context}"
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def build_chat_completions_url(base_url: str) -> str:
    if base_url.endswith("/chat/completions"):
        return base_url

    return f"{base_url}/chat/completions"


def call_local_llama(
    *,
    config: LocalLlamaEvaluationConfig,
    messages: list[dict[str, str]],
) -> str:
    url = build_chat_completions_url(config.base_url)

    headers = {
        "Content-Type": "application/json",
    }

    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 350,
        "stream": False,
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=config.timeout_seconds,
    )
    response.raise_for_status()

    data = response.json()

    return str(data["choices"][0]["message"]["content"]).strip()


def normalize_two_paragraph_summary(summary: str) -> str:
    paragraphs = [
        paragraph.strip()
        for paragraph in summary.split("\n\n")
        if paragraph.strip()
    ]

    if len(paragraphs) >= 2:
        return "\n\n".join(paragraphs[:2])

    if len(paragraphs) == 1:
        return (
            paragraphs[0]
            + "\n\n"
            + "The deterministic evaluation metrics remain the source of truth. "
            "This Llama-generated summary is only a readable interpretation layer "
            "on top of the measured results."
        )

    return (
        "The local Llama model did not return a usable summary."
        + "\n\n"
        + "The deterministic evaluation metrics remain the source of truth."
    )


def write_ai_summary(summary: str) -> None:
    AI_SUMMARY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# AI Evaluation Summary",
        "",
        summary,
        "",
    ]

    AI_SUMMARY_OUTPUT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def append_ai_summary_to_report(
    *,
    report_path: Path,
    summary: str,
) -> None:
    if not report_path.exists():
        return

    current_report = report_path.read_text(encoding="utf-8").rstrip()
    clean_summary = summary.strip()

    ai_summary_section = "\n\n".join(
        [
            "## AI Evaluation Summary",
            clean_summary,
        ]
    )

    # Keep Report Scope as the final explanatory section.
    if "## Report Scope" in current_report:
        before_scope, report_scope = current_report.split(
            "## Report Scope",
            maxsplit=1,
        )

        updated_report = (
            before_scope.rstrip()
            + "\n\n"
            + ai_summary_section
            + "\n\n"
            + "## Report Scope"
            + report_scope
        )

        report_path.write_text(
            updated_report.rstrip() + "\n",
            encoding="utf-8",
        )
        return

    updated_report = (
        current_report
        + "\n\n"
        + ai_summary_section
        + "\n"
    )

    report_path.write_text(
        updated_report,
        encoding="utf-8",
    )


def generate_local_ai_evaluation_summary(
    *,
    results_df: pd.DataFrame,
    report_path: Path,
) -> str | None:
    config = load_local_llama_config()

    if not config.enabled:
        print("AI evaluation summary is disabled by configuration.")
        return None

    if not config.base_url:
        print("AI_EVALUATION_BASE_URL is missing. Skipping AI summary.")
        return None

    if not config.model:
        print("AI_EVALUATION_MODEL is missing. Skipping AI summary.")
        return None

    if results_df.empty:
        print("Evaluation results are empty. Skipping AI summary.")
        return None

    if not ensure_local_llama_model_available(config):
        return None

    try:
        messages = build_llama_messages(results_df)

        summary = call_local_llama(
            config=config,
            messages=messages,
        )

        normalized_summary = normalize_two_paragraph_summary(summary)

        write_ai_summary(normalized_summary)

        append_ai_summary_to_report(
            report_path=report_path,
            summary=normalized_summary,
        )

        print(f"AI evaluation summary written to: {AI_SUMMARY_OUTPUT_PATH}")

        return normalized_summary

    except Exception as error:
        print("AI evaluation summary failed. Continuing without AI summary.")
        print(f"Reason: {error}")
        return None