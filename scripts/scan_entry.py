from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENTRY = "examples/structured_data/iris_lineage_test.py"
DEFAULT_OUTPUT_DIR = ".etiq/artifacts/structured_data"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Etiq against a workspace-relative Python entry file."
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=REPO_ROOT,
        help="Workspace root. Defaults to the repository root.",
    )
    parser.add_argument(
        "--entry",
        default=DEFAULT_ENTRY,
        help=f"Workspace-relative Python entry file to scan. Defaults to {DEFAULT_ENTRY}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Workspace-local output directory. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--lineage-format",
        choices=("json", "dot"),
        default="json",
        help="Lineage artifact format to write.",
    )
    parser.add_argument(
        "--mock-openai",
        action="store_true",
        help="Run openai_unstructured_example.py in deterministic mock mode.",
    )
    return parser


def _load_run_scan():
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from etiq_runner.wrapper import run_scan
    except ImportError as import_error:
        raise RuntimeError(
            "Unable to import the Etiq wrapper. Install dependencies with "
            "`python -m pip install -r requirements.txt`. "
            f"Import error: {import_error}"
        ) from import_error
    return run_scan


def main() -> int:
    args = _parser().parse_args()

    if args.mock_openai:
        os.environ["ETIQ_OPENAI_EXAMPLE_MODE"] = "mock"

    try:
        run_scan = _load_run_scan()
        result = run_scan(
            workspace_root=args.workspace_root,
            entry_file=args.entry,
            output_dir=args.output_dir,
            lineage_format=args.lineage_format,
        )
    except Exception as exc:  # noqa: BLE001 - last-resort CLI failure report.
        print(
            json.dumps(
                {
                    "status": "scanner_failed",
                    "scan_errors": [{"type": type(exc).__name__, "message": str(exc)}],
                    "result_path": None,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1

    if result.get("result_path"):
        print(json.dumps({"result_path": result["result_path"], "status": result["status"]}))
    else:
        print(json.dumps({"result_path": None, "status": result["status"]}), file=sys.stderr)

    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
