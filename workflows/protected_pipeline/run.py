from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(".etiq") / "artifacts" / "protected_pipeline"
PROTECTED_ENTRY = "workflows/protected_pipeline/workflow.py"

sys.path.insert(0, str(REPO_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the protected workflow through Etiq.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Workspace-local output directory. Defaults to {DEFAULT_OUTPUT_DIR.as_posix()}.",
    )
    parser.add_argument(
        "--lineage-format",
        choices=("json", "dot"),
        default="json",
        help="Lineage artifact format to write.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing artifacts without running Etiq.",
    )
    return parser


def _failure(status: str, exc: Exception, *, result_path: str | None = None) -> int:
    print(
        json.dumps(
            {
                "result_path": result_path,
                "scan_errors": [{"type": type(exc).__name__, "message": str(exc)}],
                "status": status,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 1


def _success(result: dict[str, Any]) -> int:
    print(json.dumps({"result_path": result["result_path"], "status": result["status"]}))
    return 0


def main() -> int:
    args = _parser().parse_args()

    try:
        from etiq_runner.validation import validate_scan_result
        from etiq_runner.wrapper import run_scan
    except Exception as exc:  # noqa: BLE001 - CLI should return structured setup failures.
        return _failure("scanner_failed", exc)

    if args.validate_only:
        try:
            result = validate_scan_result(
                REPO_ROOT,
                args.output_dir,
                PROTECTED_ENTRY,
                expected_lineage_format=args.lineage_format,
            )
        except Exception as exc:  # noqa: BLE001 - validation errors are user-facing.
            return _failure("validation_failed", exc, result_path="scan-result.json")
        return _success(result)

    result = run_scan(
        workspace_root=REPO_ROOT,
        entry_file=PROTECTED_ENTRY,
        output_dir=args.output_dir,
        lineage_format=args.lineage_format,
    )
    if result.get("status") != "completed":
        print(
            json.dumps(
                {
                    "result_path": result.get("result_path"),
                    "scan_errors": result.get("scan_errors", []),
                    "status": result.get("status", "scanner_failed"),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1

    try:
        validate_scan_result(
            REPO_ROOT,
            args.output_dir,
            PROTECTED_ENTRY,
            expected_lineage_format=args.lineage_format,
        )
    except Exception as exc:  # noqa: BLE001 - validation errors are user-facing.
        return _failure("validation_failed", exc, result_path=result.get("result_path"))

    return _success(result)


if __name__ == "__main__":
    raise SystemExit(main())
