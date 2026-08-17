from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from etiq_runner.wrapper import (
    SUPPORTED_LINEAGE_FORMATS,
    _file_sha256,
    _relative_path,
    _validate_output_dir,
    _validate_workspace_root,
    resolve_entry_file,
)


class ScanValidationError(RuntimeError):
    """Raised when a scan-result.json cannot be accepted for a protected workflow."""


def _lineage_artifact_path(output_path: Path, lineage_path: str) -> Path:
    requested = Path(lineage_path)
    if requested.is_absolute():
        raise ScanValidationError("lineage_path must be relative to the output directory")
    resolved = (output_path / requested).resolve()
    try:
        resolved.relative_to(output_path.resolve())
    except ValueError as exc:
        raise ScanValidationError("lineage_path must stay inside the output directory") from exc
    return resolved


def validate_scan_result(
    workspace_root: str | Path,
    output_dir: str | Path,
    entry_file: str | Path | None = None,
    *,
    strict_completed: bool = True,
    expected_lineage_format: str | None = None,
) -> dict[str, Any]:
    """Validate scan-result.json and its lineage artifact for a protected workflow."""
    root = _validate_workspace_root(workspace_root)
    output_path = _validate_output_dir(output_dir, root)
    result_path = output_path / "scan-result.json"
    if not result_path.exists():
        raise ScanValidationError(f"scan-result.json is missing: {result_path}")

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScanValidationError(f"scan-result.json is not readable JSON: {exc}") from exc

    if strict_completed and result.get("status") != "completed":
        raise ScanValidationError("scan-result.json status is not completed")
    if result.get("scan_errors") != []:
        raise ScanValidationError("scan-result.json contains scan_errors")

    if expected_lineage_format is not None:
        normalized_format = str(expected_lineage_format).lower()
        if normalized_format not in SUPPORTED_LINEAGE_FORMATS:
            raise ScanValidationError(
                "expected_lineage_format must be one of: "
                + ", ".join(sorted(SUPPORTED_LINEAGE_FORMATS))
            )
        if result.get("lineage_format") != normalized_format:
            raise ScanValidationError("scan-result.json lineage_format does not match expected format")

    lineage_path = result.get("lineage_path")
    if not isinstance(lineage_path, str) or not lineage_path.strip():
        raise ScanValidationError("scan-result.json lineage_path is missing")
    lineage_artifact = _lineage_artifact_path(output_path, lineage_path)
    if not lineage_artifact.exists():
        raise ScanValidationError(f"lineage artifact is missing: {lineage_path}")

    if entry_file is not None:
        entry_path = resolve_entry_file(entry_file, root)
        expected_target = _relative_path(entry_path, root)
        if result.get("entry_file_path") != expected_target or result.get("target") != expected_target:
            raise ScanValidationError("scan-result.json was produced for a different entry file")
        if result.get("entry_file_hash") != _file_sha256(entry_path):
            raise ScanValidationError("scan-result.json is stale for the current entry file")

    return result
