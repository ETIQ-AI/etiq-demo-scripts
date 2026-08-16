from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


DEFAULT_ENTRY = "library_functions_examples/iris_lineage_test.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an Etiq library-functions scan for a workspace-relative Python entry file."
    )
    parser.add_argument(
        "--entry",
        default=DEFAULT_ENTRY,
        help=f"Workspace-relative Python entry file to scan. Defaults to {DEFAULT_ENTRY}.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional workspace-relative path for writing the full structured scan summary.",
    )
    parser.add_argument(
        "--mock-openai",
        action="store_true",
        help="Run openai_unstructured_example.py in deterministic mock mode.",
    )
    return parser


def _load_wrapper_functions():
    try:
        from .etiq_agent_wrapper import build_scan_summary, write_summary
    except ImportError as relative_error:
        try:
            from etiq_agent_wrapper import build_scan_summary, write_summary
        except ImportError as absolute_error:
            raise RuntimeError(
                "Unable to import the Etiq wrapper. Install dependencies with "
                "`python -m pip install -r requirements.txt`. "
                f"Import error: {absolute_error}"
            ) from relative_error
    return build_scan_summary, write_summary


def main() -> int:
    args = _parser().parse_args()

    if args.mock_openai:
        os.environ["ETIQ_OPENAI_EXAMPLE_MODE"] = "mock"

    try:
        build_scan_summary, write_summary = _load_wrapper_functions()
        summary = build_scan_summary(args.entry)
    except Exception as exc:  # noqa: BLE001 - CLI should report scanner/read failures clearly.
        print(
            json.dumps(
                {
                    "target_file": args.entry,
                    "scan_errors": [str(exc)],
                    "captured_objects": {
                        "states": [],
                        "dataframes": [],
                        "models": [],
                        "agents": [],
                        "unstructured": [],
                    },
                    "counts": {
                        "states": 0,
                        "dataframes": 0,
                        "models": 0,
                        "agents": 0,
                        "unstructured": 0,
                    },
                    "source_evidence": [],
                    "deliberate_error_detected": False,
                    "lineage_json": "",
                    "lineage_dot": "",
                    "lineage_generated": {"json": False, "dot": False},
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1

    if args.json_output:
        written_path = write_summary(summary, args.json_output)
        try:
            display_path = str(written_path.relative_to(Path.cwd()))
        except ValueError:
            display_path = str(written_path)
        summary = {
            **summary,
            "json_output": display_path,
        }

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
