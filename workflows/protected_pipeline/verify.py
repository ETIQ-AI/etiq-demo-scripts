from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_ENTRY = "workflows/protected_pipeline/workflow.py"
RUNNER_ENTRY = "workflows/protected_pipeline/run.py"
OUTPUT_DIR = Path(".etiq") / "artifacts" / "protected_pipeline"
GUARD_ERROR = "This workflow must be run through the Etiq wrapper."

sys.path.insert(0, str(REPO_ROOT))


def _run_direct_guard_check() -> None:
    env = os.environ.copy()
    env.pop("RUNNING_UNDER_ETIQ", None)
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / PROTECTED_ENTRY)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode == 0 or GUARD_ERROR not in completed.stderr:
        raise RuntimeError("direct execution guard did not fail with the expected message")
    print("PASS direct execution is blocked")


def _run_forced_runner() -> None:
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / RUNNER_ENTRY)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "forced runner failed")
    pointer = json.loads(completed.stdout)
    if pointer != {"result_path": "scan-result.json", "status": "completed"}:
        raise RuntimeError(f"unexpected runner pointer: {pointer}")
    print("PASS forced runner completed through Etiq")


def _validate_artifacts() -> None:
    from etiq_runner.validation import validate_scan_result

    result = validate_scan_result(REPO_ROOT, OUTPUT_DIR, PROTECTED_ENTRY)
    print("PASS validation accepts the current scan artifacts")

    lineage_path = REPO_ROOT / OUTPUT_DIR / result["lineage_path"]
    if not lineage_path.exists():
        raise RuntimeError(f"lineage artifact is missing: {lineage_path}")
    json.loads((REPO_ROOT / OUTPUT_DIR / "scan-result.json").read_text(encoding="utf-8"))
    if result["lineage_format"] == "json":
        json.loads(lineage_path.read_text(encoding="utf-8"))
    print("PASS scan-result.json and lineage.json are valid")


def main() -> int:
    try:
        _run_direct_guard_check()
        _run_forced_runner()
        _validate_artifacts()
    except Exception as exc:  # noqa: BLE001 - self-check should be direct for humans.
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
