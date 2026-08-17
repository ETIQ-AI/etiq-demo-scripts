from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = Path(".etiq") / "reports"

sys.path.insert(0, str(REPO_ROOT))

from etiq_runner.validation import ScanValidationError, validate_scan_result


JOBS: list[dict[str, Any]] = [
    {
        "name": "protected_pipeline",
        "kind": "protected",
        "command": ["py", "-3.11", "workflows/protected_pipeline/run.py"],
        "output_dir": ".etiq/artifacts/protected_pipeline",
        "entry_file": "workflows/protected_pipeline/workflow.py",
        "expected_lineage_format": "json",
    },
    {
        "name": "structured_data_example",
        "kind": "example",
        "command": ["py", "-3.11", "etiq_agent_wrapper.py"],
        "output_dir": ".etiq/artifacts/structured_data",
        "entry_file": "examples/structured_data/iris_lineage_test.py",
        "expected_lineage_format": "json",
    },
    {
        "name": "agentic_unstructured_example_mock",
        "kind": "example",
        "command": [
            "py",
            "-3.11",
            "etiq_agent_wrapper.py",
            "--entry",
            "examples/agentic_unstructured/openai_unstructured_example.py",
            "--mock-openai",
            "--output-dir",
            ".etiq/artifacts/agentic_unstructured_example_mock",
        ],
        "output_dir": ".etiq/artifacts/agentic_unstructured_example_mock",
        "entry_file": "examples/agentic_unstructured/openai_unstructured_example.py",
        "expected_lineage_format": "json",
    },
]


LABELS = {
    "PASS": '<span style="color:green">PASS</span>',
    "WARN": '<span style="color:orange">WARN</span>',
    "FAIL": '<span style="color:red">FAIL</span>',
    "FOUND": '<span style="color:green">FOUND</span>',
    "MISSING": '<span style="color:red">MISSING</span>',
    "INVALID": '<span style="color:red">INVALID</span>',
    "NOT RUN": '<span style="color:gray">NOT RUN</span>',
    "NOT CHECKED": '<span style="color:gray">NOT CHECKED</span>',
}

GUARD_ERROR = "This workflow must be run through the Etiq wrapper."


def _repo_path(workspace_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (workspace_root / path).resolve()


def _relative_to_workspace(workspace_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _report_link(report_dir: Path, target: Path, text: str) -> str:
    relative = os.path.relpath(target.resolve(), report_dir.resolve()).replace("\\", "/")
    return f"[{text}]({relative})"


def _label(status: str) -> str:
    return LABELS.get(status, status)


def _command_text(command: list[str]) -> str:
    return " ".join(command)


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(loaded, dict):
        return None, "scan-result.json must contain a JSON object"
    return loaded, None


def _lineage_path(
    output_dir: Path,
    scan_result: dict[str, Any] | None,
    job: dict[str, Any],
    *,
    read_error: str | None = None,
) -> Path | None:
    if scan_result is None:
        if read_error:
            return None
        return output_dir / f"lineage.{job.get('expected_lineage_format', 'json')}"

    lineage_name = scan_result.get("lineage_path")
    if isinstance(lineage_name, str) and lineage_name.strip():
        return output_dir / lineage_name
    return None


def _scan_error_summary(scan_errors: Any) -> str:
    if not scan_errors:
        return "none"
    if not isinstance(scan_errors, list):
        return str(scan_errors)
    pieces = []
    for error in scan_errors[:5]:
        if isinstance(error, dict):
            error_type = error.get("type")
            message = error.get("message")
            if error_type and message:
                pieces.append(f"{error_type}: {message}")
            else:
                pieces.append(json.dumps(error, sort_keys=True))
        else:
            pieces.append(str(error))
    if len(scan_errors) > 5:
        pieces.append(f"... {len(scan_errors) - 5} more")
    return "; ".join(pieces)


def _lineage_link(report_dir: Path, lineage_path: Path | None) -> str:
    if lineage_path is None:
        return "lineage unavailable"
    return _report_link(report_dir, lineage_path, "lineage")


def _check_direct_guard(workspace_root: Path, job: dict[str, Any], *, enabled: bool) -> tuple[str, str]:
    if job.get("kind") != "protected":
        return "NOT CHECKED", "Direct-run guard applies only to protected workflows."
    if not enabled:
        return "NOT CHECKED", "Direct-run guard check was skipped."

    entry_file = job.get("entry_file")
    if not entry_file:
        return "FAIL", "No protected workflow entry file configured."

    env = os.environ.copy()
    env.pop("RUNNING_UNDER_ETIQ", None)
    completed = subprocess.run(
        [sys.executable, str(_repo_path(workspace_root, entry_file))],
        cwd=workspace_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0 and GUARD_ERROR in completed.stderr:
        return "PASS", "Direct workflow execution is blocked by the guard."
    if completed.returncode == 0:
        return "FAIL", "Direct-run guard failed because direct workflow execution succeeded."
    return "FAIL", "Direct workflow execution failed, but not with the expected guard error."


def _validate_protected(
    workspace_root: Path,
    job: dict[str, Any],
    scan_result: dict[str, Any] | None,
) -> tuple[str, str]:
    if job.get("kind") != "protected":
        return "NOT CHECKED", "No protected-workflow validation is configured for this job."
    if scan_result is None:
        return "NOT RUN", "No scan result available to validate."

    try:
        validate_scan_result(
            workspace_root,
            job["output_dir"],
            job.get("entry_file"),
            expected_lineage_format=job.get("expected_lineage_format"),
        )
    except ScanValidationError as exc:
        return "FAIL", str(exc)
    return "PASS", "Protected workflow validation accepted the artifacts."


def assess_job(
    workspace_root: str | Path,
    job: dict[str, Any],
    report_dir: str | Path,
    *,
    check_direct_guard: bool = True,
) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    output_dir = _repo_path(root, job["output_dir"])
    scan_result_path = output_dir / "scan-result.json"
    scan_result, read_error = _read_json(scan_result_path)
    lineage_path = _lineage_path(output_dir, scan_result, job, read_error=read_error)
    lineage_exists = lineage_path.exists() if lineage_path is not None else False

    if read_error:
        validation_status = "FAIL"
        validation_message = read_error
    else:
        validation_status, validation_message = _validate_protected(root, job, scan_result)

    direct_guard_status, direct_guard_message = _check_direct_guard(
        root,
        job,
        enabled=check_direct_guard,
    )

    issues: list[str] = []
    limitations = []
    if scan_result is None:
        if read_error:
            gate = "FAIL"
            evidence = "INVALID"
            status = "invalid_json"
            issues.append(f"scan-result.json is not readable JSON: {read_error}")
        else:
            gate = "NOT RUN"
            evidence = "MISSING"
            status = "not_run"
            issues.append("No valid Etiq evidence found because scan-result.json is missing.")
    else:
        evidence = "FOUND"
        status = str(scan_result.get("status") or "unknown")
        scan_errors = scan_result.get("scan_errors")
        limitations = list(scan_result.get("limitations") or [])

        if status != "completed":
            issues.append(f"scan-result.json status is {status!r}, not 'completed'.")
        if scan_errors:
            issues.append("scan-result.json contains scan_errors.")
        if lineage_path is None:
            issues.append("scan-result.json does not identify a lineage artifact.")
        elif not lineage_exists:
            issues.append("Lineage artifact is missing.")
        if validation_status == "FAIL":
            issues.append(f"Protected validation failed: {validation_message}")
        if direct_guard_status == "FAIL":
            issues.append(direct_guard_message)

        if issues:
            gate = "FAIL"
        elif limitations:
            gate = "WARN"
        else:
            gate = "PASS"

    counts = scan_result.get("counts", {}) if scan_result else {}
    captured_objects = scan_result.get("captured_objects", {}) if scan_result else {}
    report_dir_path = _repo_path(root, report_dir)

    return {
        "job": job,
        "gate": gate,
        "evidence": evidence,
        "status": status,
        "issues": issues,
        "limitations": limitations,
        "output_dir": output_dir,
        "scan_result_path": scan_result_path,
        "lineage_path": lineage_path,
        "lineage_exists": lineage_exists,
        "scan_result": scan_result,
        "scan_errors": scan_result.get("scan_errors", []) if scan_result else [],
        "counts": {
            "dataframes": counts.get("dataframes", 0),
            "models": counts.get("models", 0),
            "agents": counts.get("agents", 0),
            "unstructured": counts.get("unstructured", 0),
        },
        "captured_objects": {
            "dataframes": captured_objects.get("dataframes", []),
            "models": captured_objects.get("models", []),
            "agents": captured_objects.get("agents", []),
            "unstructured": captured_objects.get("unstructured", []),
        },
        "validation_status": validation_status,
        "validation_message": validation_message,
        "direct_guard_status": direct_guard_status,
        "direct_guard_message": direct_guard_message,
        "scan_link": _report_link(report_dir_path, scan_result_path, "scan-result"),
        "lineage_link": _lineage_link(report_dir_path, lineage_path),
        "output_display": _relative_to_workspace(root, output_dir),
    }


def _format_time(generated_at: datetime) -> str:
    value = generated_at.astimezone()
    timezone = value.strftime("%z")
    if timezone:
        timezone = f"{timezone[:3]}:{timezone[3:]}"
    return value.strftime("%Y-%m-%d %H:%M ") + timezone


def _format_names(names: list[Any]) -> str:
    if not names:
        return "none"
    return ", ".join(str(name) for name in names)


def render_report(
    assessments: list[dict[str, Any]],
    workspace_root: str | Path,
    *,
    generated_at: datetime | None = None,
) -> str:
    root = Path(workspace_root).resolve()
    generated_time = _format_time(generated_at or datetime.now().astimezone())
    lines = [
        "# Etiq Run Report",
        "",
        f"Generated: {generated_time}",
        f"Workspace: {root}",
        "",
        "## Summary",
        "",
        "| Job | Gate | Etiq Evidence | Status | Artifacts |",
        "| --- | --- | --- | --- | --- |",
    ]

    for assessment in assessments:
        artifacts = f"{assessment['scan_link']}, {assessment['lineage_link']}"
        lines.append(
            "| {job} | {gate} | {evidence} | {status} | {artifacts} |".format(
                job=assessment["job"]["name"],
                gate=_label(assessment["gate"]),
                evidence=_label(assessment["evidence"]),
                status=assessment["status"],
                artifacts=artifacts,
            )
        )

    lines.append("")

    for assessment in assessments:
        job = assessment["job"]
        counts = assessment["counts"]
        lines.extend(
            [
                f"## {job['name']}",
                "",
                f"Gate: {_label(assessment['gate'])}",
                f"Command: `{_command_text(job['command'])}`",
                f"Output: `{assessment['output_display']}/`",
                "",
                "Artifacts:",
                f"- {assessment['scan_link']}",
                f"- {assessment['lineage_link']} ({'found' if assessment['lineage_exists'] else 'missing'})",
                "",
                "Evidence:",
                f"- `status`: {assessment['status']}",
                f"- `scan_errors`: {_scan_error_summary(assessment['scan_errors'])}",
                f"- `entry_file_path`: {job.get('entry_file', 'not configured')}",
                f"- `entry_file_hash`: {(assessment['scan_result'] or {}).get('entry_file_hash', 'not available')}",
                f"- dataframe states: {counts['dataframes']}; unique dataframe names: {_format_names(assessment['captured_objects']['dataframes'])}",
                f"- model states: {counts['models']}; unique model names: {_format_names(assessment['captured_objects']['models'])}",
                f"- agent states: {counts['agents']}; unique agent names: {_format_names(assessment['captured_objects']['agents'])}",
                f"- unstructured states: {counts['unstructured']}; unique unstructured names: {_format_names(assessment['captured_objects']['unstructured'])}",
                "",
                "Checks:",
                f"- validation: {_label(assessment['validation_status'])} - {assessment['validation_message']}",
                f"- direct-run guard: {_label(assessment['direct_guard_status'])} - {assessment['direct_guard_message']}",
                "",
                "Assessment:",
            ]
        )

        if assessment["issues"]:
            for issue in assessment["issues"]:
                lines.append(f"- {issue}")
            if assessment["status"] == "not_run":
                lines.append("- No acceptable Etiq evidence was produced for this job.")
            else:
                lines.append("- Possible bypass or incomplete run if source changes are not reflected in current artifacts.")
        elif assessment["gate"] == "WARN":
            lines.append("- Etiq artifacts exist, but limitations were recorded.")
            for limitation in assessment["limitations"]:
                lines.append(f"- {limitation}")
        else:
            lines.append("- The available Etiq evidence is acceptable for this job.")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def generate_report(
    workspace_root: str | Path = REPO_ROOT,
    *,
    jobs: list[dict[str, Any]] | None = None,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    check_direct_guard: bool = True,
    generated_at: datetime | None = None,
) -> Path:
    root = Path(workspace_root).resolve()
    report_dir_path = _repo_path(root, report_dir)
    assessments = [
        assess_job(
            root,
            job,
            report_dir_path,
            check_direct_guard=check_direct_guard,
        )
        for job in (jobs or JOBS)
    ]
    report_dir_path.mkdir(parents=True, exist_ok=True)
    report_path = report_dir_path / "latest-run-report.md"
    report_path.write_text(
        render_report(assessments, root, generated_at=generated_at),
        encoding="utf-8",
    )
    return report_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a Markdown report for known Etiq jobs.")
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=REPO_ROOT,
        help="Workspace root. Defaults to the repository root.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Workspace-local directory for latest-run-report.md.",
    )
    parser.add_argument(
        "--skip-direct-guard-check",
        action="store_true",
        help="Do not execute protected workflow files directly to verify the guard.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    report_path = generate_report(
        args.workspace_root,
        report_dir=args.report_dir,
        check_direct_guard=not args.skip_direct_guard_check,
    )
    root = Path(args.workspace_root).resolve()
    print(
        json.dumps(
            {
                "report_path": _relative_to_workspace(root, report_path),
                "status": "completed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
