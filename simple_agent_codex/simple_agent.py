from __future__ import annotations

import argparse
import subprocess
import sys
from collections import deque
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from etiq_copilot.engine.implementations.scanner.code_scanner import DebuggerCodeScanner
from etiq_copilot.engine.implementations.scanner.scan_results import CodeScannerResult

try:
    from .verification_functions_codex import get_empty_objects
except ImportError:
    from verification_functions_codex import get_empty_objects


SYSTEM_PROMPT = """You write complete runnable Python scripts for data science tasks.

Return a structured response that matches the requested schema.
Requirements for python_code:
- Return one full Python script.
- Do not use markdown fences.
- Make the script runnable as-is.
- Prefer common Python data science libraries when they fit the task.
- Produce intermediate artifacts that are non-empty.
"""


def _default_output_script() -> Path:
    return Path(__file__).with_name("generated_pipeline.py")


class AgentConfig(BaseModel):
    task: str
    model: str
    max_rewrites: int = 5
    output_script: Path = Field(default_factory=_default_output_script)


class CodeDraft(BaseModel):
    python_code: str
    summary: str


class ExecutionResult(BaseModel):
    script_path: Path
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    status: Literal["succeeded", "failed"]


class ArtifactInfo(BaseModel):
    state_id: UUID
    state_name: str
    names: list[str] = Field(default_factory=list)
    line_no: int | None = None
    child_state_ids: list[UUID] = Field(default_factory=list)
    empty: bool | None = None


class ValidationResult(BaseModel):
    status: Literal["passed", "failed", "no_artifacts", "skipped"]
    checked_artifacts: list[ArtifactInfo] = Field(default_factory=list)
    failing_artifact: ArtifactInfo | None = None
    failure_reason: str | None = None
    lineage_dot: str = ""


class AgentOutcome(BaseModel):
    status: Literal["succeeded", "max_rewrites_exceeded"]
    final_code_path: Path
    final_execution_result: ExecutionResult
    validation_result: ValidationResult
    lineage_dot: str = ""
    checked_artifact_order: list[str] = Field(default_factory=list)
    rewrite_count: int = 0
    code_summary: str = ""


def scan_file(
    scan_file_path: Path | str,
) -> CodeScannerResult:
    """Analyze Python code and return scan results."""
    scan_file_path = Path(scan_file_path)
    scan_results = CodeScannerResult()
    original_code: str | None = None
    test_scanner = DebuggerCodeScanner()
    try:
        original_code = Path(scan_file_path).read_text(encoding="utf-8")
        scan_results = test_scanner.scan_code(code_str=original_code)
    except Exception:
        raise
    return scan_results


def _safe_create_lineage_graph(scan_result: CodeScannerResult) -> str:
    try:
        lineage_dot = scan_result.create_full_lineage_graph()
    except Exception:
        return ""
    return lineage_dot or ""


def _build_generation_agent(model_name: str) -> Agent[None, CodeDraft]:
    return Agent(
        model_name,
        output_type=CodeDraft,
        instructions=SYSTEM_PROMPT,
    )


def _clean_python_code(code: str) -> str:
    stripped = code.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def _artifact_state_id(artifact: object) -> UUID:
    state_id = getattr(artifact, "state_id", None)
    if not isinstance(state_id, UUID):
        raise ValueError("Expected an etiq_copilot artifact with a UUID state_id.")
    return state_id


def _artifact_names(artifact: object) -> list[str]:
    names = getattr(artifact, "names", None)
    if names is None:
        return []
    if isinstance(names, str):
        return [names]
    return sorted(str(name) for name in names)


def _artifact_state_name(artifact: object) -> str:
    state_name = getattr(artifact, "state_name", None)
    if isinstance(state_name, str) and state_name:
        return state_name
    names = _artifact_names(artifact)
    if names:
        return names[0]
    return str(_artifact_state_id(artifact))


def _artifact_line_no(artifact: object) -> int | None:
    line_no = getattr(artifact, "line_no", None)
    if isinstance(line_no, int):
        return line_no
    return None


def _artifact_children(artifact: object) -> list[object]:
    children = getattr(artifact, "children", ())
    if not isinstance(children, Iterable):
        return []
    return sorted(children, key=_artifact_sort_key)


def _artifact_sort_key(artifact: object) -> tuple[int, str, str]:
    line_no = _artifact_line_no(artifact)
    safe_line_no = line_no if line_no is not None else sys.maxsize
    return (safe_line_no, _artifact_state_name(artifact), str(_artifact_state_id(artifact)))


def _artifact_label(info: ArtifactInfo) -> str:
    if info.state_name:
        return info.state_name
    if info.names:
        return info.names[0]
    return str(info.state_id)


def _build_artifact_info(artifact: object, *, empty: bool | None = None) -> ArtifactInfo:
    return ArtifactInfo(
        state_id=_artifact_state_id(artifact),
        state_name=_artifact_state_name(artifact),
        names=_artifact_names(artifact),
        line_no=_artifact_line_no(artifact),
        child_state_ids=[_artifact_state_id(child) for child in _artifact_children(artifact)],
        empty=empty,
    )


def _build_prompt(
    config: AgentConfig,
    *,
    previous_code: str | None = None,
    execution_result: ExecutionResult | None = None,
    validation_result: ValidationResult | None = None,
) -> str:
    sections = [
        "Write a complete Python script for the following data science task.",
        f"Task:\n{config.task}",
        (
            "The script must run from disk as a standalone file and should create non-empty "
            "data artifacts during its pipeline."
        ),
    ]

    if previous_code is None:
        sections.append("This is the initial draft.")
    else:
        sections.append("This is a rewrite. Return a full replacement script.")
        sections.append(f"Previous code:\n{previous_code}")

    if execution_result is not None:
        sections.append(
            "Most recent execution result:\n"
            f"- status: {execution_result.status}\n"
            f"- exit_code: {execution_result.exit_code}\n"
            f"- stdout:\n{execution_result.stdout}\n"
            f"- stderr:\n{execution_result.stderr}"
        )

    if validation_result is not None:
        sections.append(f"Most recent validation status: {validation_result.status}")
        if validation_result.failure_reason:
            sections.append(f"Validation failure reason:\n{validation_result.failure_reason}")
        if validation_result.failing_artifact is not None:
            failing_artifact = validation_result.failing_artifact
            sections.append(
                "Failing artifact details:\n"
                f"- state_name: {failing_artifact.state_name}\n"
                f"- state_id: {failing_artifact.state_id}\n"
                f"- line_no: {failing_artifact.line_no}"
            )

    sections.append(
        "Return valid structured output. Put only Python source code in python_code, with no backticks."
    )
    return "\n\n".join(sections)


def _generate_code(
    agent: Agent[None, CodeDraft],
    config: AgentConfig,
    *,
    previous_code: str | None = None,
    execution_result: ExecutionResult | None = None,
    validation_result: ValidationResult | None = None,
) -> CodeDraft:
    prompt = _build_prompt(
        config,
        previous_code=previous_code,
        execution_result=execution_result,
        validation_result=validation_result,
    )
    result = agent.run_sync(prompt)
    return CodeDraft(
        python_code=_clean_python_code(result.output.python_code),
        summary=result.output.summary.strip(),
    )


def _write_generated_script(script_path: Path, code: str) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(code, encoding="utf-8")


def _run_script(script_path: Path) -> ExecutionResult:
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        cwd=str(script_path.parent),
        check=False,
    )
    status: Literal["succeeded", "failed"] = "succeeded" if completed.returncode == 0 else "failed"
    return ExecutionResult(
        script_path=script_path,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        status=status,
    )


def _validate_artifact_lineage(
    root_artifact: object,
    *,
    verify_fn: Callable[[list[object]], list[object]],
    lineage_dot: str,
) -> ValidationResult:
    checked_artifacts: list[ArtifactInfo] = []
    visited: set[UUID] = set()
    queue: deque[object] = deque([root_artifact])

    while queue:
        current_artifact = queue.popleft()
        current_state_id = _artifact_state_id(current_artifact)
        if current_state_id in visited:
            continue
        visited.add(current_state_id)

        failing_objects = verify_fn([current_artifact])
        is_empty = bool(failing_objects)
        current_info = _build_artifact_info(current_artifact, empty=is_empty)
        checked_artifacts.append(current_info)

        if is_empty:
            return ValidationResult(
                status="failed",
                checked_artifacts=checked_artifacts,
                failing_artifact=current_info,
                failure_reason=(
                    f"Artifact '{current_info.state_name}' is empty or failed the verification check."
                ),
                lineage_dot=lineage_dot,
            )

        for child in _artifact_children(current_artifact):
            child_state_id = _artifact_state_id(child)
            if child_state_id not in visited:
                queue.append(child)

    return ValidationResult(
        status="passed",
        checked_artifacts=checked_artifacts,
        lineage_dot=lineage_dot,
    )


def run_pipeline_agent(
    config: AgentConfig,
    verify_fn: Callable[[list[object]], list[object]] = get_empty_objects,
) -> AgentOutcome:
    agent = _build_generation_agent(config.model)
    script_path = config.output_script.resolve()
    previous_code: str | None = None
    last_execution = ExecutionResult(
        script_path=script_path,
        exit_code=-1,
        stdout="",
        stderr="",
        status="failed",
    )
    last_validation = ValidationResult(
        status="skipped",
        failure_reason="Validation has not run yet.",
    )
    last_summary = ""

    for attempt in range(config.max_rewrites + 1):
        draft = _generate_code(
            agent,
            config,
            previous_code=previous_code,
            execution_result=None if attempt == 0 else last_execution,
            validation_result=None if attempt == 0 else last_validation,
        )
        previous_code = draft.python_code
        last_summary = draft.summary

        _write_generated_script(script_path, draft.python_code)
        last_execution = _run_script(script_path)

        if last_execution.status == "failed":
            last_validation = ValidationResult(
                status="skipped",
                failure_reason="Execution failed before artifact validation could run.",
            )
            if attempt == config.max_rewrites:
                return AgentOutcome(
                    status="max_rewrites_exceeded",
                    final_code_path=script_path,
                    final_execution_result=last_execution,
                    validation_result=last_validation,
                    checked_artifact_order=[],
                    rewrite_count=attempt,
                    code_summary=last_summary,
                )
            continue

        scan_result = scan_file(script_path)
        object_state = scan_result.get_dataframes()
        lineage_dot = _safe_create_lineage_graph(scan_result)

        if not object_state:
            last_validation = ValidationResult(
                status="no_artifacts",
                failure_reason="etiq_copilot returned no artifacts from get_dataframes().",
                lineage_dot=lineage_dot,
            )
            if attempt == config.max_rewrites:
                return AgentOutcome(
                    status="max_rewrites_exceeded",
                    final_code_path=script_path,
                    final_execution_result=last_execution,
                    validation_result=last_validation,
                    lineage_dot=lineage_dot,
                    checked_artifact_order=[],
                    rewrite_count=attempt,
                    code_summary=last_summary,
                )
            continue

        last_validation = _validate_artifact_lineage(
            object_state[0],
            verify_fn=verify_fn,
            lineage_dot=lineage_dot,
        )
        checked_artifact_order = [_artifact_label(info) for info in last_validation.checked_artifacts]

        if last_validation.status == "passed":
            return AgentOutcome(
                status="succeeded",
                final_code_path=script_path,
                final_execution_result=last_execution,
                validation_result=last_validation,
                lineage_dot=lineage_dot,
                checked_artifact_order=checked_artifact_order,
                rewrite_count=attempt,
                code_summary=last_summary,
            )

        if attempt == config.max_rewrites:
            return AgentOutcome(
                status="max_rewrites_exceeded",
                final_code_path=script_path,
                final_execution_result=last_execution,
                validation_result=last_validation,
                lineage_dot=lineage_dot,
                checked_artifact_order=checked_artifact_order,
                rewrite_count=attempt,
                code_summary=last_summary,
            )

    return AgentOutcome(
        status="max_rewrites_exceeded",
        final_code_path=script_path,
        final_execution_result=last_execution,
        validation_result=last_validation,
        lineage_dot=last_validation.lineage_dot,
        checked_artifact_order=[_artifact_label(info) for info in last_validation.checked_artifacts],
        rewrite_count=config.max_rewrites,
        code_summary=last_summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate, run, and validate a simple pipeline-coding agent.")
    parser.add_argument("--task", required=True, help="The data science task to solve.")
    parser.add_argument("--model", required=True, help="PydanticAI model identifier, for example openai:gpt-5.")
    parser.add_argument(
        "--max-rewrites",
        type=int,
        default=3,
        help="Maximum number of full-script rewrites after the initial generation.",
    )
    parser.add_argument(
        "--output-script",
        type=Path,
        default=_default_output_script(),
        help="Where to save the generated pipeline script.",
    )
    args = parser.parse_args()

    config = AgentConfig(
        task=args.task,
        model=args.model,
        max_rewrites=args.max_rewrites,
        output_script=args.output_script,
    )
    outcome = run_pipeline_agent(config)
    print(outcome.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
