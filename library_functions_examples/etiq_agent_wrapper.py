from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

from etiq_copilot.engine.implementations.scanner.code_scanner import DebuggerCodeScanner
from etiq_copilot.engine.implementations.scanner.scan_results import CodeScannerResult


REPO_ROOT = Path(__file__).resolve().parents[1]


class EntryFileError(ValueError):
    """Raised when a requested entry file cannot be scanned safely."""


def _package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def resolve_entry_file(entry_file: str | Path, repo_root: Path | None = None) -> Path:
    """Resolve a workspace-relative Python entry file without escaping the repo."""
    root = (repo_root or REPO_ROOT).resolve()
    requested = Path(entry_file)
    candidates = []

    if requested.is_absolute():
        candidates.append(requested)
    else:
        candidates.append(root / requested)
        candidates.append(Path.cwd() / requested)

    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        if resolved.exists():
            if resolved.suffix != ".py":
                raise EntryFileError(f"Entry file must be a Python file: {entry_file}")
            return resolved

    raise EntryFileError(f"Entry file not found inside workspace: {entry_file}")


@contextlib.contextmanager
def _scan_import_context(entry_path: Path) -> Iterator[None]:
    original_cwd = Path.cwd()
    original_sys_path = list(sys.path)
    sys.path.insert(0, str(entry_path.parent))
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_sys_path


def scan_entry_file(entry_file: str | Path) -> CodeScannerResult:
    """Run an entry file through Etiq and return the raw scan result."""
    entry_path = resolve_entry_file(entry_file)
    source = entry_path.read_text(encoding="utf-8")
    if "sklearn" in source:
        try:
            import sklearn.datasets  # noqa: F401
            import sklearn.ensemble  # noqa: F401
            import sklearn.model_selection  # noqa: F401
        except Exception:
            pass
    scanner = DebuggerCodeScanner()
    with _scan_import_context(entry_path):
        return scanner.scan_code(code_str=source)


def _scan_errors(scan_result: CodeScannerResult) -> list[str]:
    errors = getattr(scan_result, "scan_errors", None)
    if errors is None:
        return []
    if isinstance(errors, (list, tuple)):
        return [str(error) for error in errors]
    return [str(errors)]


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


def build_scan_summary(entry_file: str | Path) -> dict[str, Any]:
    """Scan a workspace-relative entry file and return a stable structured summary."""
    entry_path = resolve_entry_file(entry_file)
    scan_result = scan_entry_file(entry_path)
    scan_errors = _scan_errors(scan_result)

    dataframe_names, dataframe_note = _safe_call(scan_result, "list_dataframes")
    model_names, model_note = _safe_call(scan_result, "list_models")
    agent_names, agent_note = _safe_call(scan_result, "list_agents")
    state_names, state_note = _safe_call(scan_result, "list_states")

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

    lineage_json, lineage_json_note = _lineage_graph(scan_result, "json")
    lineage_dot, lineage_dot_note = _lineage_graph(scan_result, "dot")

    all_state_evidence = _source_evidence(
        dataframe_states + model_states + agent_states + unstructured_states
    )
    all_names = {
        name
        for evidence in all_state_evidence
        for name in evidence.get("names", [])
    }

    notes = [
        note
        for note in [
            dataframe_note,
            model_note,
            agent_note,
            state_note,
            dataframe_states_note,
            model_states_note,
            agent_states_note,
            unstructured_note,
            lineage_json_note,
            lineage_dot_note,
        ]
        if note
    ]

    return {
        "target_file": entry_path.relative_to(REPO_ROOT).as_posix(),
        "etiq_copilot_version": _package_version("etiq-copilot"),
        "scan_errors": scan_errors,
        "captured_objects": {
            "states": state_names,
            "dataframes": dataframe_names,
            "models": model_names,
            "agents": agent_names,
            "unstructured": unstructured_names,
        },
        "counts": {
            "states": len(state_names),
            "dataframes": len(dataframe_states),
            "models": len(model_states),
            "agents": len(agent_states),
            "unstructured": len(unstructured_states),
        },
        "source_evidence": all_state_evidence,
        "deliberate_error_detected": "deliberate_empty_features" in all_names,
        "lineage_json": lineage_json,
        "lineage_dot": lineage_dot,
        "lineage_generated": {
            "json": bool(lineage_json),
            "dot": bool(lineage_dot),
        },
        "optional_api_notes": notes,
    }


def write_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
    """Write a scan summary to a workspace-local JSON file."""
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
