from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_OUTPUT_ROOT = REPO_ROOT / ".etiq" / "test_outputs"
sys.path.insert(0, str(REPO_ROOT))

try:
    import etiq_runner.wrapper as wrapper
    from etiq_runner.validation import ScanValidationError, validate_scan_result
    from etiq_runner.wrapper import (
        EntryFileError,
        resolve_entry_file,
        run_scan,
        write_summary,
    )
    from scripts.generate_run_report import generate_report
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without dependencies.
    raise unittest.SkipTest(f"Etiq demo dependencies are not installed: {exc}") from exc


IRIS_ENTRY = "examples/structured_data/iris_lineage_test.py"
OPENAI_ENTRY = "examples/agentic_unstructured/openai_unstructured_example.py"
PROTECTED_ENTRY = "workflows/protected_pipeline/workflow.py"
RUNNER_ENTRY = "workflows/protected_pipeline/run.py"
SCAN_CLI_ENTRY = "scripts/scan_entry.py"


def captured_names(result: dict, object_kind: str) -> set[str]:
    return set(result["captured_objects"][object_kind])


def evidence_for(result: dict, name: str) -> list[dict]:
    return [
        item
        for item in result["source_evidence"]
        if name in item.get("names", [])
    ]


def source_contains(result: dict, name: str, expected_text: str) -> bool:
    return any(expected_text in (item.get("source") or "") for item in evidence_for(result, name))


def workspace_relative(path: str | Path) -> str:
    return Path(path).resolve().relative_to(REPO_ROOT).as_posix()


class FakeNode:
    def as_string(self) -> str:
        return "df = make_dataframe()"

    def scope(self) -> object:
        return object()


class FakeState:
    names = {"df"}
    line_no = 1
    node = FakeNode()


class FakePartialResult:
    scan_errors = None

    def list_dataframes(self) -> list[str]:
        return ["df"]

    def list_models(self) -> list[str]:
        return []

    def list_agents(self) -> list[str]:
        return []

    def get_dataframes(self) -> list[FakeState]:
        return [FakeState()]

    def get_models(self) -> list[object]:
        return []

    def get_agent_states(self) -> list[object]:
        return []

    def get_unstructured_states(self) -> list[object]:
        return []

    def create_full_lineage_graph(self, graph_format: str = "dot") -> str:
        raise RuntimeError("lineage API unavailable")


class FakeScanner:
    def scan_code(self, code_str: str) -> FakePartialResult:
        return FakePartialResult()


class FakeCompleteResult(FakePartialResult):
    def create_full_lineage_graph(self, graph_format: str = "dot") -> str:
        return '{"objects": []}' if graph_format == "json" else "digraph {}"


class GuardRecordingScanner:
    seen_guard_value: str | None = None

    def scan_code(self, code_str: str) -> FakeCompleteResult:
        self.seen_guard_value = os.environ.get("RUNNING_UNDER_ETIQ")
        return FakeCompleteResult()


class EtiqRunnerTests(unittest.TestCase):
    def make_output_dir(self) -> tempfile.TemporaryDirectory[str]:
        base = TEST_OUTPUT_ROOT
        base.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=base)

    def make_report_base_dir(self) -> Path:
        base = TEST_OUTPUT_ROOT / "report_generator_tests" / self._testMethodName
        shutil.rmtree(base, ignore_errors=True)
        base.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(base, ignore_errors=True))
        return base

    def write_report_fixture(
        self,
        output_dir: str | Path,
        *,
        entry_file: str = PROTECTED_ENTRY,
        status: str = "completed",
        scan_errors: list | None = None,
        lineage_path: str | None = "lineage.json",
        entry_hash: str | None = None,
    ) -> dict:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if entry_hash is None:
            entry_hash = hashlib.sha256((REPO_ROOT / entry_file).read_bytes()).hexdigest()
        result = {
            "status": status,
            "scan_errors": [] if scan_errors is None else scan_errors,
            "lineage_path": lineage_path,
            "lineage_format": Path(lineage_path).suffix.lstrip(".") if lineage_path else "json",
            "entry_file_path": entry_file,
            "target": entry_file,
            "entry_file_hash": entry_hash,
            "result_path": "scan-result.json",
            "counts": {
                "dataframes": 1,
                "models": 0,
                "agents": 0,
                "unstructured": 0,
            },
            "captured_objects": {
                "dataframes": ["df"],
                "models": [],
                "agents": [],
                "unstructured": [],
            },
            "limitations": [],
        }
        (output_path / "scan-result.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    def report_job(self, name: str, output_dir: str | Path, *, kind: str = "protected", entry_file: str = PROTECTED_ENTRY) -> dict:
        return {
            "name": name,
            "kind": kind,
            "command": ["py", "-3.11", "dummy.py"],
            "output_dir": workspace_relative(output_dir),
            "entry_file": entry_file,
            "expected_lineage_format": "json",
        }

    def test_generate_run_report_marks_valid_protected_artifacts_as_pass(self) -> None:
        base_dir = self.make_report_base_dir()
        output_dir = base_dir / "protected_pass"
        report_dir = base_dir / "reports"
        self.write_report_fixture(output_dir)
        (output_dir / "lineage.json").write_text('{"objects": []}', encoding="utf-8")

        report_path = generate_report(
            REPO_ROOT,
            jobs=[self.report_job("protected_pipeline", output_dir)],
            report_dir=workspace_relative(report_dir),
            check_direct_guard=False,
        )
        report = report_path.read_text(encoding="utf-8")

        self.assertEqual("latest-run-report.md", report_path.name)
        self.assertIn("| protected_pipeline | <span style=\"color:green\">PASS</span>", report)
        self.assertIn("[scan-result](../protected_pass/scan-result.json)", report)
        self.assertIn("validation: <span style=\"color:green\">PASS</span>", report)
        self.assertIn("dataframe states: 1; unique dataframe names: df", report)
        self.assertNotIn("freshness:", report)

    def test_generate_run_report_marks_missing_scan_as_not_run(self) -> None:
        base_dir = self.make_report_base_dir()
        output_dir = base_dir / "not_run"
        report_dir = base_dir / "reports"
        report_path = generate_report(
            REPO_ROOT,
            jobs=[self.report_job("missing_job", output_dir)],
            report_dir=workspace_relative(report_dir),
            check_direct_guard=False,
        )
        report = report_path.read_text(encoding="utf-8")

        self.assertIn("| missing_job | <span style=\"color:gray\">NOT RUN</span>", report)
        self.assertIn("No valid Etiq evidence found because scan-result.json is missing.", report)
        self.assertIn("No acceptable Etiq evidence was produced for this job.", report)

    def test_generate_run_report_marks_missing_lineage_as_fail(self) -> None:
        base_dir = self.make_report_base_dir()
        output_dir = base_dir / "missing_lineage"
        report_dir = base_dir / "reports"
        self.write_report_fixture(output_dir)
        report_path = generate_report(
            REPO_ROOT,
            jobs=[self.report_job("missing_lineage_job", output_dir)],
            report_dir=workspace_relative(report_dir),
            check_direct_guard=False,
        )
        report = report_path.read_text(encoding="utf-8")

        self.assertIn("| missing_lineage_job | <span style=\"color:red\">FAIL</span>", report)
        self.assertIn("Lineage artifact is missing.", report)
        self.assertIn("validation: <span style=\"color:red\">FAIL</span>", report)

    def test_generate_run_report_does_not_reuse_stale_lineage_without_current_pointer(self) -> None:
        base_dir = self.make_report_base_dir()
        output_dir = base_dir / "stale_lineage_without_pointer"
        report_dir = base_dir / "reports"
        self.write_report_fixture(
            output_dir,
            status="scanner_failed",
            scan_errors=[{"type": "RuntimeError", "message": "scan failed"}],
            lineage_path=None,
        )
        (output_dir / "lineage.json").write_text('{"objects": ["stale"]}', encoding="utf-8")

        report_path = generate_report(
            REPO_ROOT,
            jobs=[self.report_job("stale_lineage_job", output_dir)],
            report_dir=workspace_relative(report_dir),
            check_direct_guard=False,
        )
        report = report_path.read_text(encoding="utf-8")

        self.assertIn("| stale_lineage_job | <span style=\"color:red\">FAIL</span>", report)
        self.assertIn("lineage unavailable", report)
        self.assertIn("scan-result.json does not identify a lineage artifact.", report)
        self.assertNotIn("[lineage](../stale_lineage_without_pointer/lineage.json)", report)

    def test_generate_run_report_marks_non_completed_status_as_fail(self) -> None:
        base_dir = self.make_report_base_dir()
        output_dir = base_dir / "partial_status"
        report_dir = base_dir / "reports"
        self.write_report_fixture(output_dir, entry_file=IRIS_ENTRY, status="partial")
        (output_dir / "lineage.json").write_text('{"objects": []}', encoding="utf-8")
        report_path = generate_report(
            REPO_ROOT,
            jobs=[
                self.report_job(
                    "structured_data_example",
                    output_dir,
                    kind="example",
                    entry_file=IRIS_ENTRY,
                )
            ],
            report_dir=workspace_relative(report_dir),
            check_direct_guard=False,
        )
        report = report_path.read_text(encoding="utf-8")

        self.assertIn("| structured_data_example | <span style=\"color:red\">FAIL</span>", report)
        self.assertIn("`status`: partial", report)
        self.assertIn("scan-result.json status is 'partial', not 'completed'.", report)

    def test_generate_run_report_marks_stale_protected_hash_as_fail(self) -> None:
        base_dir = self.make_report_base_dir()
        output_dir = base_dir / "stale_hash"
        report_dir = base_dir / "reports"
        self.write_report_fixture(output_dir, entry_hash="stale")
        (output_dir / "lineage.json").write_text('{"objects": []}', encoding="utf-8")
        report_path = generate_report(
            REPO_ROOT,
            jobs=[self.report_job("stale_protected_job", output_dir)],
            report_dir=workspace_relative(report_dir),
            check_direct_guard=False,
        )
        report = report_path.read_text(encoding="utf-8")

        self.assertIn("| stale_protected_job | <span style=\"color:red\">FAIL</span>", report)
        self.assertIn("scan-result.json is stale for the current entry file", report)
        self.assertIn("Protected validation failed", report)
        self.assertNotIn("freshness:", report)

    def test_direct_execution_of_protected_workflow_fails_with_guard_error(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(REPO_ROOT / PROTECTED_ENTRY)],
            check=False,
            capture_output=True,
            cwd=REPO_ROOT,
            text=True,
        )

        self.assertNotEqual(0, completed.returncode)
        self.assertIn("This workflow must be run through the Etiq wrapper.", completed.stderr)

    def test_supported_runner_executes_protected_workflow_and_writes_artifacts(self) -> None:
        with self.make_output_dir() as output_dir:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / RUNNER_ENTRY),
                    "--output-dir",
                    output_dir,
                    "--lineage-format",
                    "json",
                ],
                check=False,
                capture_output=True,
                cwd=REPO_ROOT,
                text=True,
            )
            pointer = json.loads(completed.stdout)
            result_path = Path(output_dir) / "scan-result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            lineage_exists = (Path(output_dir) / result["lineage_path"]).exists()

        self.assertEqual("", completed.stderr)
        self.assertEqual(0, completed.returncode)
        self.assertEqual({"result_path": "scan-result.json", "status": "completed"}, pointer)
        self.assertEqual(PROTECTED_ENTRY, result["entry_file_path"])
        self.assertEqual(PROTECTED_ENTRY, result["target"])
        self.assertTrue(result["entry_file_hash"])
        self.assertEqual([], result["scan_errors"])
        self.assertTrue(lineage_exists)

    def test_runner_reports_structured_failure_when_etiq_import_is_unavailable(self) -> None:
        code = (
            "import importlib.abc, json, sys;"
            "\nfrom workflows.protected_pipeline import run;"
            "\nclass BlockEtiq(importlib.abc.MetaPathFinder):"
            "\n    def find_spec(self, fullname, path=None, target=None):"
            "\n        if fullname.startswith('etiq_copilot'):"
            "\n            raise ModuleNotFoundError(\"blocked etiq_copilot\")"
            "\n        return None"
            "\nsys.meta_path.insert(0, BlockEtiq());"
            "\nsys.argv = ['run.py', '--output-dir', '.etiq/test_outputs/no_etiq_import'];"
            "\nraise SystemExit(run.main())"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            cwd=REPO_ROOT,
            text=True,
        )
        failure = json.loads(completed.stderr)

        self.assertEqual(1, completed.returncode)
        self.assertEqual("", completed.stdout)
        self.assertEqual("scanner_failed", failure["status"])
        self.assertEqual("scan-result.json", failure["result_path"])
        self.assertIn("blocked etiq_copilot", failure["scan_errors"][0]["message"])
        self.assertTrue((REPO_ROOT / ".etiq/test_outputs/no_etiq_import/scan-result.json").exists())

    def test_protected_runner_preserves_result_path_on_handled_scan_failure(self) -> None:
        from workflows.protected_pipeline import run as protected_run

        result = {
            "result_path": "scan-result.json",
            "scan_errors": [{"type": "EntryFileError", "message": "bad entry"}],
            "status": "invalid_input",
        }
        stderr = io.StringIO()
        with self.make_output_dir() as output_dir:
            with mock.patch.object(sys, "argv", ["run.py", "--output-dir", output_dir]):
                with mock.patch("etiq_runner.wrapper.run_scan", return_value=result):
                    with mock.patch("sys.stderr", stderr):
                        rc = protected_run.main()

        failure = json.loads(stderr.getvalue())
        self.assertEqual(1, rc)
        self.assertEqual("scan-result.json", failure["result_path"])
        self.assertEqual("invalid_input", failure["status"])

    def test_wrapper_import_does_not_require_etiq_copilot(self) -> None:
        code = (
            "import importlib.abc, json, sys;"
            "\nclass BlockEtiq(importlib.abc.MetaPathFinder):"
            "\n    def find_spec(self, fullname, path=None, target=None):"
            "\n        if fullname.startswith('etiq_copilot'):"
            "\n            raise ModuleNotFoundError(\"blocked etiq_copilot\")"
            "\n        return None"
            "\nsys.meta_path.insert(0, BlockEtiq());"
            "\nimport etiq_runner.wrapper as wrapper;"
            "\nprint(json.dumps({"
            "'imported': True, "
            "'scanner_cached': wrapper.DebuggerCodeScanner is not None"
            "}, sort_keys=True))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            cwd=REPO_ROOT,
            text=True,
        )

        self.assertEqual("", completed.stderr)
        self.assertEqual(0, completed.returncode)
        self.assertEqual(
            {"imported": True, "scanner_cached": False},
            json.loads(completed.stdout),
        )

    def test_wrapper_restores_pre_existing_etiq_guard_after_scan(self) -> None:
        previous_guard = os.environ.get("RUNNING_UNDER_ETIQ")
        os.environ["RUNNING_UNDER_ETIQ"] = "pre-existing"
        scanner = GuardRecordingScanner()
        try:
            with self.make_output_dir() as output_dir:
                with mock.patch.object(wrapper, "DebuggerCodeScanner", return_value=scanner):
                    result = run_scan(REPO_ROOT, IRIS_ENTRY, output_dir, "json")
        finally:
            if previous_guard is None:
                os.environ.pop("RUNNING_UNDER_ETIQ", None)
            else:
                os.environ["RUNNING_UNDER_ETIQ"] = previous_guard

        self.assertEqual("completed", result["status"])
        self.assertEqual("1", scanner.seen_guard_value)
        self.assertEqual(previous_guard if previous_guard is not None else None, os.environ.get("RUNNING_UNDER_ETIQ"))

    def test_wrapper_removes_etiq_guard_after_scan_when_previously_unset(self) -> None:
        previous_guard = os.environ.pop("RUNNING_UNDER_ETIQ", None)
        scanner = GuardRecordingScanner()
        try:
            with self.make_output_dir() as output_dir:
                with mock.patch.object(wrapper, "DebuggerCodeScanner", return_value=scanner):
                    result = run_scan(REPO_ROOT, IRIS_ENTRY, output_dir, "json")
                guard_after_scan = os.environ.get("RUNNING_UNDER_ETIQ")
        finally:
            if previous_guard is not None:
                os.environ["RUNNING_UNDER_ETIQ"] = previous_guard

        self.assertEqual("completed", result["status"])
        self.assertEqual("1", scanner.seen_guard_value)
        self.assertIsNone(guard_after_scan)

    def test_iris_scan_writes_structured_result_and_json_lineage(self) -> None:
        with self.make_output_dir() as output_dir:
            result = run_scan(REPO_ROOT, IRIS_ENTRY, output_dir, "json")
            result_path = Path(output_dir) / "scan-result.json"
            lineage_path = Path(output_dir) / "lineage.json"

            self.assertEqual("completed", result["status"])
            self.assertEqual("1.0", result["schema_version"])
            self.assertEqual([], result["scan_errors"])
            self.assertEqual(IRIS_ENTRY, result["target"])
            self.assertEqual("lineage.json", result["lineage_path"])
            self.assertTrue(result_path.exists())
            self.assertTrue(lineage_path.exists())
            self.assertNotIn("lineage_json", result)
            self.assertNotIn("lineage_dot", result)
            self.assertIn('"objects"', lineage_path.read_text(encoding="utf-8"))

            written_result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result, written_result)

            required_dataframes = {
                "clean_measurements_df",
                "deliberate_empty_features",
                "final_report_df",
                "iris_df",
                "iris_with_species_df",
                "species_lookup_df",
                "species_summary_df",
                "wide_petal_df",
            }
            self.assertLessEqual(required_dataframes, captured_names(result, "dataframes"))
            self.assertEqual(set(), captured_names(result, "models"))
            self.assertTrue(result["deliberate_error_detected"])
            self.assertTrue(source_contains(result, "deliberate_empty_features", "not-a-real-species"))
            self.assertTrue(source_contains(result, "final_report_df", "species_summary_df.assign"))

    def test_openai_mock_scan_writes_same_shape_and_captures_agentic_states(self) -> None:
        previous_mode = os.environ.get("ETIQ_OPENAI_EXAMPLE_MODE")
        os.environ["ETIQ_OPENAI_EXAMPLE_MODE"] = "mock"
        try:
            with self.make_output_dir() as output_dir:
                result = run_scan(REPO_ROOT, OPENAI_ENTRY, output_dir, "json")
                lineage_exists = (Path(output_dir) / result["lineage_path"]).exists()
        finally:
            if previous_mode is None:
                os.environ.pop("ETIQ_OPENAI_EXAMPLE_MODE", None)
            else:
                os.environ["ETIQ_OPENAI_EXAMPLE_MODE"] = previous_mode

        self.assertEqual("completed", result["status"])
        self.assertEqual([], result["scan_errors"])
        self.assertEqual(OPENAI_ENTRY, result["target"])
        self.assertEqual("lineage.json", result["lineage_path"])
        self.assertTrue(lineage_exists)
        self.assertLessEqual(
            {"documents_df", "selected_documents_df"},
            captured_names(result, "dataframes"),
        )
        self.assertIn("agent", captured_names(result, "agents"))
        self.assertLessEqual(
            {
                "customer_question",
                "document_context",
                "user_prompt",
                "result",
                "response",
                "normalized_response_text",
                "final_answer_record",
            },
            captured_names(result, "unstructured"),
        )
        self.assertTrue(source_contains(result, "user_prompt", "Answer the customer question"))
        self.assertTrue(source_contains(result, "result", "agent.run_sync(user_prompt)"))

    def test_invalid_entry_outside_workspace_writes_invalid_input_result(self) -> None:
        with self.make_output_dir() as output_dir:
            result = run_scan(REPO_ROOT, "../README.md", output_dir, "json")
            written_result = json.loads((Path(output_dir) / "scan-result.json").read_text())

        self.assertEqual("invalid_input", result["status"])
        self.assertEqual(result, written_result)
        self.assertIn("stay inside workspace", result["scan_errors"][0])

    def test_non_python_entry_writes_invalid_input_result(self) -> None:
        with self.make_output_dir() as output_dir:
            result = run_scan(REPO_ROOT, "README.md", output_dir, "json")

        self.assertEqual("invalid_input", result["status"])
        self.assertIn("must be a Python file", result["scan_errors"][0])

    def test_output_directory_outside_workspace_is_rejected_without_writing_there(self) -> None:
        outside_workspace = REPO_ROOT.parent / "outside_scan_output"
        result = run_scan(REPO_ROOT, IRIS_ENTRY, outside_workspace, "json")

        self.assertEqual("invalid_input", result["status"])
        self.assertNotIn("result_path", result)
        self.assertIn("output_dir must stay inside workspace", result["scan_errors"][0])

    def test_unsupported_lineage_format_writes_invalid_input_result(self) -> None:
        with self.make_output_dir() as output_dir:
            result = run_scan(REPO_ROOT, IRIS_ENTRY, output_dir, "svg")
            written_result = json.loads((Path(output_dir) / "scan-result.json").read_text())

        self.assertEqual("invalid_input", result["status"])
        self.assertEqual(result, written_result)
        self.assertIn("lineage_format must be one of", result["scan_errors"][0])

    def test_cli_handled_failure_writes_scan_result_and_returns_nonzero(self) -> None:
        with self.make_output_dir() as output_dir:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / SCAN_CLI_ENTRY),
                    "--workspace-root",
                    str(REPO_ROOT),
                    "--entry",
                    "README.md",
                    "--output-dir",
                    output_dir,
                    "--lineage-format",
                    "json",
                ],
                check=False,
                capture_output=True,
                cwd=REPO_ROOT,
                text=True,
            )
            pointer = json.loads(completed.stdout)
            written_result = json.loads((Path(output_dir) / "scan-result.json").read_text())

        self.assertEqual(1, completed.returncode)
        self.assertEqual({"result_path": "scan-result.json", "status": "invalid_input"}, pointer)
        self.assertEqual("invalid_input", written_result["status"])
        self.assertEqual("scan-result.json", written_result["result_path"])
        self.assertNotIn("lineage_json", written_result)
        self.assertNotIn("lineage_dot", written_result)

    def test_validate_scan_result_rejects_missing_stale_and_non_completed_results(self) -> None:
        with self.make_output_dir() as output_dir:
            with self.assertRaisesRegex(ScanValidationError, "missing"):
                validate_scan_result(REPO_ROOT, output_dir, IRIS_ENTRY)

        with self.make_output_dir() as output_dir:
            with mock.patch.object(
                wrapper,
                "DebuggerCodeScanner",
                return_value=GuardRecordingScanner(),
            ):
                result = run_scan(REPO_ROOT, IRIS_ENTRY, output_dir, "json")
            self.assertEqual("completed", validate_scan_result(REPO_ROOT, output_dir, IRIS_ENTRY)["status"])

            result_path = Path(output_dir) / "scan-result.json"
            result["status"] = "partial"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ScanValidationError, "not completed"):
                validate_scan_result(REPO_ROOT, output_dir, IRIS_ENTRY)

        with self.make_output_dir() as output_dir:
            with mock.patch.object(
                wrapper,
                "DebuggerCodeScanner",
                return_value=GuardRecordingScanner(),
            ):
                result = run_scan(REPO_ROOT, IRIS_ENTRY, output_dir, "json")

            result_path = Path(output_dir) / "scan-result.json"
            result["entry_file_hash"] = "stale"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ScanValidationError, "stale"):
                validate_scan_result(REPO_ROOT, output_dir, IRIS_ENTRY)

    def test_validate_scan_result_rejects_expected_lineage_format_mismatch(self) -> None:
        with self.make_output_dir() as output_dir:
            with mock.patch.object(
                wrapper,
                "DebuggerCodeScanner",
                return_value=GuardRecordingScanner(),
            ):
                result = run_scan(REPO_ROOT, IRIS_ENTRY, output_dir, "dot")

            self.assertEqual("dot", result["lineage_format"])
            self.assertEqual("completed", validate_scan_result(
                REPO_ROOT,
                output_dir,
                IRIS_ENTRY,
                expected_lineage_format="dot",
            )["status"])
            with self.assertRaisesRegex(ScanValidationError, "lineage_format"):
                validate_scan_result(
                    REPO_ROOT,
                    output_dir,
                    IRIS_ENTRY,
                    expected_lineage_format="json",
                )

    def test_validate_scan_result_rejects_unknown_expected_lineage_format(self) -> None:
        with self.make_output_dir() as output_dir:
            lineage_path = Path(output_dir) / "lineage.json"
            lineage_path.write_text('{"objects": []}', encoding="utf-8")
            entry_hash = hashlib.sha256((REPO_ROOT / IRIS_ENTRY).read_bytes()).hexdigest()
            (Path(output_dir) / "scan-result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "scan_errors": [],
                        "lineage_path": "lineage.json",
                        "lineage_format": "json",
                        "entry_file_path": IRIS_ENTRY,
                        "target": IRIS_ENTRY,
                        "entry_file_hash": entry_hash,
                        "result_path": "scan-result.json",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ScanValidationError, "expected_lineage_format"):
                validate_scan_result(
                    REPO_ROOT,
                    output_dir,
                    IRIS_ENTRY,
                    expected_lineage_format="svg",
                )

    def test_runner_validate_only_rejects_lineage_format_mismatch(self) -> None:
        with self.make_output_dir() as output_dir:
            lineage_path = Path(output_dir) / "lineage.dot"
            lineage_path.write_text("digraph {}", encoding="utf-8")
            entry_hash = hashlib.sha256((REPO_ROOT / PROTECTED_ENTRY).read_bytes()).hexdigest()
            result_path = Path(output_dir) / "scan-result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "scan_errors": [],
                        "lineage_path": "lineage.dot",
                        "lineage_format": "dot",
                        "entry_file_path": PROTECTED_ENTRY,
                        "target": PROTECTED_ENTRY,
                        "entry_file_hash": entry_hash,
                        "result_path": "scan-result.json",
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / RUNNER_ENTRY),
                    "--validate-only",
                    "--output-dir",
                    output_dir,
                    "--lineage-format",
                    "json",
                ],
                check=False,
                capture_output=True,
                cwd=REPO_ROOT,
                text=True,
            )
            failure = json.loads(completed.stderr)

        self.assertEqual(1, completed.returncode)
        self.assertEqual("", completed.stdout)
        self.assertEqual("validation_failed", failure["status"])
        self.assertIn("lineage_format", failure["scan_errors"][0]["message"])

    def test_validate_scan_result_rejects_scan_errors_lineage_and_wrong_target(self) -> None:
        with self.make_output_dir() as output_dir:
            with mock.patch.object(
                wrapper,
                "DebuggerCodeScanner",
                return_value=GuardRecordingScanner(),
            ):
                result = run_scan(REPO_ROOT, IRIS_ENTRY, output_dir, "json")

            result_path = Path(output_dir) / "scan-result.json"

            result["scan_errors"] = [{"type": "RuntimeError", "message": "bad scan"}]
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ScanValidationError, "contains scan_errors"):
                validate_scan_result(REPO_ROOT, output_dir, IRIS_ENTRY)

            result["scan_errors"] = []
            result["lineage_path"] = ""
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ScanValidationError, "lineage_path is missing"):
                validate_scan_result(REPO_ROOT, output_dir, IRIS_ENTRY)

            result["lineage_path"] = "missing-lineage.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ScanValidationError, "lineage artifact is missing"):
                validate_scan_result(REPO_ROOT, output_dir, IRIS_ENTRY)

            result["lineage_path"] = "lineage.json"
            result["target"] = OPENAI_ENTRY
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ScanValidationError, "different entry file"):
                validate_scan_result(REPO_ROOT, output_dir, IRIS_ENTRY)

    def test_lineage_api_failure_preserves_partial_result_and_limitation(self) -> None:
        with self.make_output_dir() as output_dir:
            with mock.patch.object(wrapper, "DebuggerCodeScanner", return_value=FakeScanner()):
                result = run_scan(REPO_ROOT, IRIS_ENTRY, output_dir, "json")

        self.assertEqual("partial", result["status"])
        self.assertEqual([], result["scan_errors"])
        self.assertIn("df", captured_names(result, "dataframes"))
        self.assertIsNone(result["lineage_path"])
        self.assertTrue(any("lineage API unavailable" in item for item in result["limitations"]))

    def test_target_failure_and_scanner_failure_are_distinct(self) -> None:
        with self.make_output_dir() as output_dir:
            target_file = Path(output_dir) / "bad_syntax.py"
            target_file.write_text("def broken(:\n    pass\n", encoding="utf-8")
            target_entry = Path(output_dir).relative_to(REPO_ROOT) / "bad_syntax.py"
            target_result = run_scan(REPO_ROOT, target_entry, output_dir, "json")

        with self.make_output_dir() as output_dir:
            with mock.patch.object(
                wrapper,
                "DebuggerCodeScanner",
                side_effect=ImportError("scanner unavailable"),
            ):
                scanner_result = run_scan(REPO_ROOT, IRIS_ENTRY, output_dir, "json")

        self.assertEqual("target_failed", target_result["status"])
        self.assertEqual("scanner_failed", scanner_result["status"])

    def test_resolve_entry_file_reports_clear_workspace_errors(self) -> None:
        with self.assertRaisesRegex(EntryFileError, "not found inside workspace"):
            resolve_entry_file("examples/structured_data/does_not_exist.py")

        with self.assertRaisesRegex(EntryFileError, "must be a Python file"):
            resolve_entry_file("README.md")

    def test_write_summary_rejects_output_outside_workspace(self) -> None:
        outside_workspace = REPO_ROOT.parent / "outside_summary.json"

        with self.assertRaisesRegex(EntryFileError, "must stay inside workspace"):
            write_summary({}, outside_workspace)


if __name__ == "__main__":
    unittest.main()
