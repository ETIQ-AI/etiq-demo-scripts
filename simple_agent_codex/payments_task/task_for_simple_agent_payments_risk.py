from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from simple_agent.simple_agent import (
        AgentConfig,
        AgentOutcome,
        AttemptOutcome,
        run_pipeline_agent,
    )
except ImportError:
    from simple_agent import AgentConfig, AgentOutcome, AttemptOutcome, run_pipeline_agent


TASK_METRICS_PREFIX = "TASK_METRICS_JSON="
TASK_METRICS_FIELDNAMES = [
    "run_id",
    "run_timestamp_utc",
    "task_id",
    "model",
    "output_script",
    "final_agent_status",
    "attempt_index",
    "is_final_attempt",
    "execution_status",
    "execution_exit_code",
    "rewrite_count",
    "attempt_code_summary",
    "artifact_validation_status",
    "artifact_validation_issue_found",
    "artifact_validation_failing_artifact",
    "artifact_validation_failure_reason",
    "metrics_capture_status",
    "metrics_capture_error",
    "primary_validation_roc_auc",
    "primary_validation_accuracy",
    "primary_validation_precision",
    "primary_validation_recall",
    "primary_validation_f1",
    "challenge_roc_auc",
    "challenge_accuracy",
    "challenge_precision",
    "challenge_recall",
    "challenge_f1",
]


def _default_output_script() -> Path:
    return Path(__file__).with_name("payments_risk_generated_pipeline.py")


def _task_metrics_csv_path() -> Path:
    return Path(__file__).with_name("task_metrics.csv")


class TaskSpec(BaseModel):
    task_id: str
    title: str
    dataset_name: list[str] = Field(default_factory=list)
    dataset_loader: str | None = None
    acceptance_criteria: list[str] = Field(default_factory=list)
    prompt: str


def build_payments_risk_task_prompt() -> str:
    return "\n".join(
        [
            (
                "Build a complete runnable Python script that trains a chargeback-risk model "
                "using these exact local file paths: "
                "C:/Users/raluc/uv_demo15/new_task_for_agent/payments_train.csv, "
                "C:/Users/raluc/uv_demo15/new_task_for_agent/payments_challenge.csv, "
                "C:/Users/raluc/uv_demo15/new_task_for_agent/customers.csv, and "
                "C:/Users/raluc/uv_demo15/new_task_for_agent/merchant_reference.json."
            ),
            "",
            "Use those exact file path strings directly in the generated code. Do not derive them dynamically and do not use __file__.",
            "",
            "Use the exact column names provided by those files. Do not invent or rename columns.",
            "payments_train.csv columns: payment_id, customer_id, merchant_id, payment_timestamp, amount, currency, channel, device_type, is_international, days_since_last_payment, is_chargeback.",
            "payments_challenge.csv columns: payment_id, customer_id, merchant_id, payment_timestamp, amount, currency, channel, device_type, is_international, days_since_last_payment, is_chargeback.",
            "customers.csv columns: customer_id, signup_date, segment, age_band, home_country.",
            (
                "merchant_reference.json is a list of merchant objects with these fields: "
                "merchant_id, profile.category, profile.tier, risk.risk_band, "
                "risk.avg_settlement_delay_days."
            ),
            "Use payment_timestamp as the transaction timestamp column for any time-based feature engineering.",
            "The customers file joins on customer_id.",
            "The merchant reference JSON joins on merchant_id and includes nested profile and risk fields.",
            "The target column in both payments files is is_chargeback.",
            "",
            (
                "Use the transaction data together with the reference data to prepare a modeling table, "
                "choose reasonable preprocessing for mixed data types and missing values, and train a model "
                "that aims for a validation ROC AUC of at least 0.80 on the primary dataset."
            ),
            "Use an out-of-time split on payments_train.csv by sorting on payment_timestamp and using the earliest 80% of rows for training and the latest 20% for validation.",
            "Do not use payment_id, customer_id, or merchant_id as direct model features.",
            "",
            (
                "After selecting the model, evaluate it on payments_challenge.csv and report the main "
                "classification metrics there as well. The challenge dataset is intentionally later and a "
                "bit noisier than the primary dataset, so the preprocessing should handle missing values and "
                "unseen categorical values robustly. Keep the major pandas artifacts and metric tables assigned "
                "to named variables so lineage tooling can capture them."
            ),
            "",
            (
                "At the end of the script, print exactly one line that starts with "
                f"{TASK_METRICS_PREFIX} followed by a compact JSON object with two top-level keys: "
                "primary_validation_metrics and challenge_metrics. Each metrics object must include "
                "roc_auc, accuracy, precision, recall, and f1."
            ),
            "",
            "Do not download external data. Return only the Python script.",
        ]
    )


def build_payments_risk_task_spec() -> TaskSpec:
    return TaskSpec(
        task_id="payments_risk_generalization_v1",
        title="Payments Chargeback Risk Generalization",
        dataset_name=[
            "payments_train",
            "payments_challenge",
            "customers",
            "merchant_reference",
        ],
        dataset_loader="pandas.read_csv() + json.load() on local files in the same directory",
        acceptance_criteria=[
            "The generated script executes without runtime errors.",
            "The first captured dataframe is the main joined non-empty modeling dataframe rather than a placeholder dataframe.",
            "All checked artifacts in the reachable lineage are non-empty.",
            "The generated pipeline uses an out-of-time validation split based on payment_timestamp.",
            "The generated pipeline excludes payment_id, customer_id, and merchant_id from direct model features.",
            "The generated pipeline produces prediction and metrics dataframes for both primary validation and challenge evaluation.",
            "The metrics dataframe content includes at least roc_auc, accuracy, precision, recall, and f1.",
        ],
        prompt=build_payments_risk_task_prompt(),
    )


PAYMENTS_RISK_TASK = build_payments_risk_task_spec()


def build_payments_risk_agent_config(
    model: str,
    *,
    max_rewrites: int = 5,
    output_script: Path | None = None,
) -> AgentConfig:
    return AgentConfig(
        task=PAYMENTS_RISK_TASK.prompt,
        model=model,
        max_rewrites=max_rewrites,
        output_script=output_script or _default_output_script(),
    )


def _extract_task_metrics_payload(stdout: str) -> tuple[dict[str, object] | None, str]:
    for line in reversed(stdout.splitlines()):
        if not line.startswith(TASK_METRICS_PREFIX):
            continue
        payload_text = line[len(TASK_METRICS_PREFIX) :].strip()
        if not payload_text:
            return None, "metrics line was present but empty"
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            return None, f"invalid metrics json: {exc.msg}"
        if not isinstance(payload, dict):
            return None, "metrics payload was not a JSON object"
        return payload, ""
    return None, "metrics line not found in script stdout"


def _metric_value(metrics: dict[str, object], key: str) -> str:
    value = metrics.get(key, "")
    return "" if value is None else str(value)


def _build_task_metrics_row(
    outcome: AgentOutcome,
    attempt_outcome: AttemptOutcome,
    *,
    model: str,
    run_id: str,
    run_timestamp_utc: str,
) -> dict[str, str]:
    metrics_payload, metrics_error = _extract_task_metrics_payload(
        attempt_outcome.execution_result.stdout
    )
    primary_metrics = {}
    challenge_metrics = {}
    metrics_capture_status = "missing"

    if metrics_payload is not None:
        primary_candidate = metrics_payload.get("primary_validation_metrics", {})
        challenge_candidate = metrics_payload.get("challenge_metrics", {})
        if isinstance(primary_candidate, dict) and isinstance(challenge_candidate, dict):
            primary_metrics = primary_candidate
            challenge_metrics = challenge_candidate
            metrics_capture_status = "captured"
            metrics_error = ""
        else:
            metrics_capture_status = "invalid_shape"
            metrics_error = (
                "metrics payload did not include object-valued primary_validation_metrics "
                "and challenge_metrics"
            )

    failing_artifact = attempt_outcome.validation_result.failing_artifact
    validation_issue_found = attempt_outcome.validation_result.status != "passed"

    return {
        "run_id": run_id,
        "run_timestamp_utc": run_timestamp_utc,
        "task_id": PAYMENTS_RISK_TASK.task_id,
        "model": model,
        "output_script": str(outcome.final_code_path),
        "final_agent_status": outcome.status,
        "attempt_index": str(attempt_outcome.attempt_index),
        "is_final_attempt": str(
            attempt_outcome.attempt_index == outcome.attempt_outcomes[-1].attempt_index
        ),
        "execution_status": attempt_outcome.execution_result.status,
        "execution_exit_code": str(attempt_outcome.execution_result.exit_code),
        "rewrite_count": str(outcome.rewrite_count),
        "attempt_code_summary": attempt_outcome.code_summary,
        "artifact_validation_status": attempt_outcome.validation_result.status,
        "artifact_validation_issue_found": str(validation_issue_found),
        "artifact_validation_failing_artifact": (
            "" if failing_artifact is None else failing_artifact.state_name
        ),
        "artifact_validation_failure_reason": attempt_outcome.validation_result.failure_reason
        or "",
        "metrics_capture_status": metrics_capture_status,
        "metrics_capture_error": metrics_error,
        "primary_validation_roc_auc": _metric_value(primary_metrics, "roc_auc"),
        "primary_validation_accuracy": _metric_value(primary_metrics, "accuracy"),
        "primary_validation_precision": _metric_value(primary_metrics, "precision"),
        "primary_validation_recall": _metric_value(primary_metrics, "recall"),
        "primary_validation_f1": _metric_value(primary_metrics, "f1"),
        "challenge_roc_auc": _metric_value(challenge_metrics, "roc_auc"),
        "challenge_accuracy": _metric_value(challenge_metrics, "accuracy"),
        "challenge_precision": _metric_value(challenge_metrics, "precision"),
        "challenge_recall": _metric_value(challenge_metrics, "recall"),
        "challenge_f1": _metric_value(challenge_metrics, "f1"),
    }


def _migrate_task_metrics_rows(existing_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    migrated_rows: list[dict[str, str]] = []
    for index, row in enumerate(existing_rows):
        migrated_row = {field: "" for field in TASK_METRICS_FIELDNAMES}
        for field in (
            "run_timestamp_utc",
            "task_id",
            "model",
            "output_script",
            "execution_status",
            "execution_exit_code",
            "rewrite_count",
            "artifact_validation_status",
            "artifact_validation_issue_found",
            "artifact_validation_failing_artifact",
            "artifact_validation_failure_reason",
            "metrics_capture_status",
            "metrics_capture_error",
            "primary_validation_roc_auc",
            "primary_validation_accuracy",
            "primary_validation_precision",
            "primary_validation_recall",
            "primary_validation_f1",
            "challenge_roc_auc",
            "challenge_accuracy",
            "challenge_precision",
            "challenge_recall",
            "challenge_f1",
        ):
            if field in row:
                migrated_row[field] = row[field]
        migrated_row["run_id"] = f"legacy-run-{index + 1}"
        migrated_row["final_agent_status"] = row.get("agent_status", "")
        migrated_row["attempt_index"] = row.get("rewrite_count", "")
        migrated_row["is_final_attempt"] = "True"
        migrated_row["attempt_code_summary"] = row.get("code_summary", "")
        migrated_rows.append(migrated_row)
    return migrated_rows


def _append_task_metrics_rows(rows: list[dict[str, str]]) -> Path:
    metrics_csv_path = _task_metrics_csv_path()
    existing_rows: list[dict[str, str]] = []
    rewrite_with_new_header = False

    if metrics_csv_path.exists() and metrics_csv_path.stat().st_size > 0:
        with metrics_csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            existing_rows = list(reader)
            existing_fieldnames = reader.fieldnames or []
        if existing_fieldnames != TASK_METRICS_FIELDNAMES:
            existing_rows = _migrate_task_metrics_rows(existing_rows)
            rewrite_with_new_header = True

    write_header = rewrite_with_new_header or not metrics_csv_path.exists() or metrics_csv_path.stat().st_size == 0
    mode = "w" if rewrite_with_new_header else "a"

    with metrics_csv_path.open(mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TASK_METRICS_FIELDNAMES)
        if write_header:
            writer.writeheader()
            for existing_row in existing_rows:
                writer.writerow(existing_row)
        for row in rows:
            writer.writerow(row)

    return metrics_csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the payments risk task setup or launch the simple_agent on that task."
    )
    parser.add_argument("--model", help="Model name to use when launching the agent.")
    parser.add_argument(
        "--max-rewrites",
        type=int,
        default=5,
        help="Rewrite budget to use if --model is provided.",
    )
    parser.add_argument(
        "--output-script",
        type=Path,
        default=_default_output_script(),
        help="Output script path to use if --model is provided.",
    )
    parser.add_argument(
        "--print-task",
        action="store_true",
        help="Print the payments risk task spec before exiting or before launching the agent.",
    )
    args = parser.parse_args()

    if args.print_task or not args.model:
        print("Payments Risk Task Spec")
        print(PAYMENTS_RISK_TASK.model_dump_json(indent=2))

    if not args.model:
        return

    config = build_payments_risk_agent_config(
        args.model,
        max_rewrites=args.max_rewrites,
        output_script=args.output_script,
    )
    outcome = run_pipeline_agent(config)
    run_id = uuid4().hex
    run_timestamp_utc = datetime.now(timezone.utc).isoformat()
    attempt_outcomes = outcome.attempt_outcomes or [
        AttemptOutcome(
            attempt_index=outcome.rewrite_count,
            execution_result=outcome.final_execution_result,
            validation_result=outcome.validation_result,
            lineage_dot=outcome.lineage_dot,
            checked_artifact_order=outcome.checked_artifact_order,
            code_summary=outcome.code_summary,
        )
    ]
    metrics_rows = [
        _build_task_metrics_row(
            outcome,
            attempt_outcome,
            model=args.model,
            run_id=run_id,
            run_timestamp_utc=run_timestamp_utc,
        )
        for attempt_outcome in attempt_outcomes
    ]
    metrics_csv_path = _append_task_metrics_rows(metrics_rows)
    print()
    print("Agent Outcome")
    print(outcome.model_dump_json(indent=2))
    print()
    print(f"Task metrics appended to {metrics_csv_path}")


if __name__ == "__main__":
    main()
