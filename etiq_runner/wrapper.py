from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from etiq_copilot.engine.implementations.scanner.scan_results import CodeScannerResult
else:
    CodeScannerResult = Any

DebuggerCodeScanner: Any = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "1.0"
SUPPORTED_LINEAGE_FORMATS = {"json", "dot"}
ETIQ_GUARD_ENV = "RUNNING_UNDER_ETIQ"


class EntryFileError(ValueError):
    """Raised when a requested entry file cannot be scanned safely."""


class ScannerFailure(RuntimeError):
    """Raised when Etiq cannot be imported, constructed, or invoked."""


class TargetFailure(RuntimeError):
    """Raised when the target cannot be parsed or executed by Etiq."""


def _package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit_sha(workspace_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:  # noqa: BLE001 - git metadata is optional.
        return None
    if completed.returncode != 0:
        return None
    sha = completed.stdout.strip()
    return sha or None


def _base_result(
    *,
    workspace_root: Path | None,
    entry_file: str | Path,
    output_dir: str | Path,
    lineage_format: str,
) -> dict[str, Any]:
    output_dir_value = str(output_dir)
    if workspace_root is not None:
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = workspace_root / output_path
        try:
            output_dir_value = _relative_path(output_path, workspace_root)
        except (ValueError, OSError):
            output_dir_value = str(output_dir)

    return {
        "schema_version": SCHEMA_VERSION,
        "etiq_copilot_version": _package_version("etiq-copilot"),
        "workspace_root": ".",
        "target": str(entry_file).replace("\\", "/"),
        "entry_file_path": str(entry_file).replace("\\", "/"),
        "entry_file_hash": None,
        "git_commit_sha": _git_commit_sha(workspace_root) if workspace_root is not None else None,
        "output_dir": output_dir_value.replace("\\", "/"),
        "lineage_format": lineage_format,
        "status": "scanner_failed",
        "scan_errors": [],
        "captured_objects": {
            "dataframes": [],
            "models": [],
            "agents": [],
            "unstructured": [],
        },
        "counts": {
            "dataframes": 0,
            "models": 0,
            "agents": 0,
            "unstructured": 0,
        },
        "source_evidence": [],
        "deliberate_error_detected": False,
        "lineage_path": None,
        "lineage_generated": False,
        "limitations": [],
    }


def _write_scan_result(result: dict[str, Any], output_path: Path) -> Path:
    output_path.mkdir(parents=True, exist_ok=True)
    result_path = output_path / "scan-result.json"
    result["result_path"] = result_path.name
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result_path


def _validate_workspace_root(workspace_root: str | Path) -> Path:
    root = Path(workspace_root).resolve()
    if not root.exists() or not root.is_dir():
        raise EntryFileError(f"workspace_root must be an existing directory: {workspace_root}")
    return root


def _validate_output_dir(output_dir: str | Path, workspace_root: Path) -> Path:
    requested = Path(output_dir)
    resolved = requested.resolve() if requested.is_absolute() else (workspace_root / requested).resolve()
    try:
        resolved.relative_to(workspace_root)
    except ValueError as exc:
        raise EntryFileError(f"output_dir must stay inside workspace: {output_dir}") from exc
    return resolved


def resolve_entry_file(entry_file: str | Path, repo_root: Path | None = None) -> Path:
    """Resolve a workspace-relative Python entry file without escaping the workspace."""
    root = (repo_root or REPO_ROOT).resolve()
    requested = Path(entry_file)
    if requested.is_absolute():
        raise EntryFileError(f"entry_file must be relative to workspace_root: {entry_file}")

    resolved = (root / requested).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise EntryFileError(f"entry_file must stay inside workspace: {entry_file}") from exc
    if not resolved.exists():
        raise EntryFileError(f"entry_file not found inside workspace: {entry_file}")
    if resolved.suffix != ".py":
        raise EntryFileError(f"entry_file must be a Python file: {entry_file}")
    return resolved


@contextlib.contextmanager
def _scan_import_context(entry_path: Path, workspace_root: Path) -> Iterator[None]:
    original_cwd = Path.cwd()
    original_sys_path = list(sys.path)
    original_guard = os.environ.get(ETIQ_GUARD_ENV)
    guard_was_set = ETIQ_GUARD_ENV in os.environ
    sys.path.insert(0, str(entry_path.parent))
    sys.path.insert(0, str(workspace_root))
    os.chdir(workspace_root)
    os.environ[ETIQ_GUARD_ENV] = "1"
    try:
        yield
    finally:
        if guard_was_set and original_guard is not None:
            os.environ[ETIQ_GUARD_ENV] = original_guard
        else:
            os.environ.pop(ETIQ_GUARD_ENV, None)
        os.chdir(original_cwd)
        sys.path[:] = original_sys_path


def _preload_common_imports(source: str) -> None:
    if "sklearn" not in source:
        return
    try:
        import sklearn.datasets  # noqa: F401
        import sklearn.ensemble  # noqa: F401
        import sklearn.model_selection  # noqa: F401
    except Exception:
        pass


def _is_target_exception(exc: Exception) -> bool:
    exc_name = type(exc).__name__.lower()
    message = str(exc).lower()
    return (
        isinstance(exc, (RuntimeError, SyntaxError))
        or "syntax" in exc_name
        or "parsing python code failed" in message
    )


def _debugger_code_scanner_class() -> Any:
    global DebuggerCodeScanner
    if DebuggerCodeScanner is not None:
        return DebuggerCodeScanner
    try:
        from etiq_copilot.engine.implementations.scanner.code_scanner import (
            DebuggerCodeScanner as LoadedDebuggerCodeScanner,
        )
    except Exception as exc:  # noqa: BLE001 - missing/incompatible Etiq is scanner setup failure.
        raise ScannerFailure(f"Unable to import DebuggerCodeScanner: {exc}") from exc
    DebuggerCodeScanner = LoadedDebuggerCodeScanner
    return DebuggerCodeScanner


def _execute_scan(entry_path: Path, workspace_root: Path) -> CodeScannerResult:
    try:
        source = entry_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TargetFailure(f"Unable to read entry_file: {exc}") from exc

    _preload_common_imports(source)

    try:
        scanner = _debugger_code_scanner_class()()
    except Exception as exc:  # noqa: BLE001 - scanner construction is package/infrastructure.
        if isinstance(exc, ScannerFailure):
            raise
        raise ScannerFailure(f"Unable to construct DebuggerCodeScanner: {exc}") from exc

    try:
        with _scan_import_context(entry_path, workspace_root):
            return scanner.scan_code(code_str=source)
    except Exception as exc:  # noqa: BLE001 - Etiq surfaces target parse/runtime failures here.
        if _is_target_exception(exc):
            raise TargetFailure(str(exc)) from exc
        raise ScannerFailure(f"Etiq scanner invocation failed: {exc}") from exc


def scan_entry_file(entry_file: str | Path, workspace_root: str | Path | None = None) -> CodeScannerResult:
    """Run an entry file through Etiq and return the raw scan result."""
    root = _validate_workspace_root(workspace_root or REPO_ROOT)
    entry_path = resolve_entry_file(entry_file, root)
    return _execute_scan(entry_path, root)


def _safe_error_item(error: Any) -> Any:
    if isinstance(error, dict):
        safe: dict[str, Any] = {}
        for key in ("type", "message", "filename", "line_number"):
            if key in error:
                safe[key] = error[key]
        if not safe:
            return str(error)
        return safe
    return str(error)


def _scan_errors(scan_result: CodeScannerResult) -> list[Any]:
    errors = getattr(scan_result, "scan_errors", None)
    if errors is None:
        return []
    if isinstance(errors, (list, tuple, set)):
        return [_safe_error_item(error) for error in errors]
    return [_safe_error_item(errors)]


def _safe_call(scan_result: CodeScannerResult, method_name: str) -> tuple[list[str], str | None]:
    method = getattr(scan_result, method_name, None)
    if method is None:
        return [], f"{method_name} is not available in this etiq-copilot version"
    try:
        values = method()
    except Exception as exc:  # noqa: BLE001 - scanner APIs vary by package release.
        return [], f"{method_name} failed: {exc}"
    return sorted(str(value) for value in values), None


def _safe_states(scan_result: CodeScannerResult, method_name: str) -> tuple[list[Any], str | None]:
    method = getattr(scan_result, method_name, None)
    if method is None:
        return [], f"{method_name} is not available in this etiq-copilot version"
    try:
        states = method()
    except Exception as exc:  # noqa: BLE001 - scanner APIs vary by package release.
        return [], f"{method_name} failed: {exc}"
    return list(states or []), None


def _state_names(state: object) -> list[str]:
    names = getattr(state, "names", None)
    if names is None:
        state_name = getattr(state, "state_name", None)
        return [str(state_name)] if state_name else []
    if isinstance(names, str):
        return [names]
    return sorted(str(name) for name in names)


def _node_source(node: object | None) -> str | None:
    if node is None:
        return None
    as_string = getattr(node, "as_string", None)
    if callable(as_string):
        try:
            return str(as_string())
        except Exception:  # noqa: BLE001
            return None
    return None


def _node_scope(node: object | None) -> str | None:
    if node is None:
        return None
    scope = getattr(node, "scope", None)
    if callable(scope):
        try:
            return type(scope()).__name__
        except Exception:  # noqa: BLE001
            return None
    return None


def _source_evidence(states: list[object]) -> list[dict[str, Any]]:
    evidence = []
    for state in states:
        node = getattr(state, "node", None)
        evidence.append(
            {
                "names": _state_names(state),
                "state_type": type(state).__name__,
                "line_no": getattr(state, "line_no", None),
                "node_type": type(node).__name__ if node is not None else None,
                "source": _node_source(node),
                "scope": _node_scope(node),
            }
        )
    return sorted(evidence, key=lambda item: (item["line_no"] or 0, item["names"]))


def _lineage_graph(scan_result: CodeScannerResult, graph_format: str) -> tuple[str, str | None]:
    try:
        return scan_result.create_full_lineage_graph(graph_format=graph_format) or "", None
    except TypeError:
        if graph_format == "dot":
            try:
                return scan_result.create_full_lineage_graph() or "", None
            except Exception as exc:  # noqa: BLE001
                return "", f"DOT lineage generation failed: {exc}"
        return "", f"Lineage graph_format={graph_format!r} is not supported"
    except Exception as exc:  # noqa: BLE001
        return "", f"{graph_format.upper()} lineage generation failed: {exc}"


def _exception_error(exc: Exception) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


def _finalize_failure(
    result: dict[str, Any],
    *,
    status: str,
    error: str | Exception,
    output_path: Path | None = None,
) -> dict[str, Any]:
    result["status"] = status
    result["scan_errors"] = [_exception_error(error)] if isinstance(error, Exception) else [error]
    result["limitations"].append("No usable CodeScannerResult was returned by Etiq.")
    if output_path is not None:
        _write_scan_result(result, output_path)
    return result


def run_scan(
    workspace_root: str | Path,
    entry_file: str | Path,
    output_dir: str | Path,
    lineage_format: str = "json",
) -> dict[str, Any]:
    """Run Etiq and write one wrapper-normalized scan-result.json plus lineage artifact."""
    normalized_format = str(lineage_format).lower()
    result = _base_result(
        workspace_root=None,
        entry_file=entry_file,
        output_dir=output_dir,
        lineage_format=normalized_format,
    )

    try:
        root = _validate_workspace_root(workspace_root)
        result = _base_result(
            workspace_root=root,
            entry_file=entry_file,
            output_dir=output_dir,
            lineage_format=normalized_format,
        )
        output_path = _validate_output_dir(output_dir, root)
        output_path.mkdir(parents=True, exist_ok=True)
    except EntryFileError as exc:
        result["status"] = "invalid_input"
        result["scan_errors"] = [str(exc)]
        result["limitations"].append("Input validation failed before Etiq was invoked.")
        return result

    try:
        if normalized_format not in SUPPORTED_LINEAGE_FORMATS:
            raise EntryFileError(
                "lineage_format must be one of: "
                + ", ".join(sorted(SUPPORTED_LINEAGE_FORMATS))
        )
        entry_path = resolve_entry_file(entry_file, root)
        result["target"] = _relative_path(entry_path, root)
        result["entry_file_path"] = result["target"]
        result["entry_file_hash"] = _file_sha256(entry_path)
    except EntryFileError as exc:
        result["status"] = "invalid_input"
        result["scan_errors"] = [str(exc)]
        result["limitations"].append("Input validation failed before Etiq was invoked.")
        _write_scan_result(result, output_path)
        return result

    try:
        scan_result = _execute_scan(entry_path, root)
    except TargetFailure as exc:
        return _finalize_failure(result, status="target_failed", error=exc, output_path=output_path)
    except ScannerFailure as exc:
        return _finalize_failure(result, status="scanner_failed", error=exc, output_path=output_path)

    scan_errors = _scan_errors(scan_result)
    limitations: list[str] = []
    if any(isinstance(error, dict) for error in scan_errors):
        limitations.append("scan_errors were sanitized to omit raw tracebacks and absolute paths.")

    dataframe_names, dataframe_note = _safe_call(scan_result, "list_dataframes")
    model_names, model_note = _safe_call(scan_result, "list_models")
    agent_names, agent_note = _safe_call(scan_result, "list_agents")

    dataframe_states, dataframe_states_note = _safe_states(scan_result, "get_dataframes")
    model_states, model_states_note = _safe_states(scan_result, "get_models")
    agent_states, agent_states_note = _safe_states(scan_result, "get_agent_states")
    unstructured_states, unstructured_note = _safe_states(scan_result, "get_unstructured_states")

    unstructured_names = sorted(
        {
            name
            for state in unstructured_states
            for name in _state_names(state)
        }
    )
    all_state_evidence = _source_evidence(
        dataframe_states + model_states + agent_states + unstructured_states
    )
    all_names = {
        name
        for evidence in all_state_evidence
        for name in evidence.get("names", [])
    }

    lineage_content, lineage_note = _lineage_graph(scan_result, normalized_format)
    lineage_path = None
    if lineage_content:
        lineage_file = output_path / f"lineage.{normalized_format}"
        lineage_file.write_text(lineage_content, encoding="utf-8")
        lineage_path = lineage_file.name
    elif lineage_note:
        limitations.append(lineage_note)
    else:
        limitations.append(f"{normalized_format.upper()} lineage generation returned no content.")

    for note in (
        dataframe_note,
        model_note,
        agent_note,
        dataframe_states_note,
        model_states_note,
        agent_states_note,
        unstructured_note,
    ):
        if note:
            limitations.append(note)

    result.update(
        {
            "scan_errors": scan_errors,
            "captured_objects": {
                "dataframes": dataframe_names,
                "models": model_names,
                "agents": agent_names,
                "unstructured": unstructured_names,
            },
            "counts": {
                "dataframes": len(dataframe_states),
                "models": len(model_states),
                "agents": len(agent_states),
                "unstructured": len(unstructured_states),
            },
            "source_evidence": all_state_evidence,
            "deliberate_error_detected": "deliberate_empty_features" in all_names,
            "lineage_path": lineage_path,
            "lineage_generated": bool(lineage_path),
            "limitations": limitations,
        }
    )

    if scan_errors:
        has_evidence = any(result["captured_objects"].values()) or bool(lineage_path)
        result["status"] = "partial" if has_evidence else "target_failed"
    elif lineage_path:
        result["status"] = "completed"
    else:
        result["status"] = "partial"

    _write_scan_result(result, output_path)
    return result


def build_scan_summary(entry_file: str | Path) -> dict[str, Any]:
    """Compatibility wrapper returning the new scan-result shape for the repo workspace."""
    return run_scan(
        workspace_root=REPO_ROOT,
        entry_file=entry_file,
        output_dir=Path(".etiq") / "artifacts" / "structured_data",
        lineage_format="json",
    )


def write_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
    """Write a JSON summary to a workspace-local path."""
    path = Path(output_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise EntryFileError(f"JSON output path must stay inside workspace: {output_path}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return path
