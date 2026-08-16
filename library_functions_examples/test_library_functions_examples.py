from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLE_DIR))

try:
    from etiq_agent_wrapper import (
        EntryFileError,
        build_scan_summary,
        resolve_entry_file,
        write_summary,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without dependencies.
    raise unittest.SkipTest(f"Etiq demo dependencies are not installed: {exc}") from exc


IRIS_ENTRY = "library_functions_examples/iris_lineage_test.py"
OPENAI_ENTRY = "library_functions_examples/openai_unstructured_example.py"


def captured_names(summary: dict, object_kind: str) -> set[str]:
    return set(summary["captured_objects"][object_kind])


def evidence_for(summary: dict, name: str) -> list[dict]:
    return [
        item
        for item in summary["source_evidence"]
        if name in item.get("names", [])
    ]


def source_contains(summary: dict, name: str, expected_text: str) -> bool:
    return any(expected_text in (item.get("source") or "") for item in evidence_for(summary, name))


class LibraryFunctionsExampleTests(unittest.TestCase):
    def test_iris_scan_summary_reports_semantic_capture(self) -> None:
        summary = build_scan_summary(IRIS_ENTRY)

        self.assertEqual([], summary["scan_errors"])
        self.assertEqual(IRIS_ENTRY, summary["target_file"])
        self.assertTrue(summary["deliberate_error_detected"])
        self.assertTrue(summary["lineage_generated"]["json"])
        self.assertTrue(summary["lineage_generated"]["dot"])
        self.assertIn('"objects"', summary["lineage_json"])
        self.assertIn("digraph", summary["lineage_dot"])

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
        self.assertLessEqual(required_dataframes, captured_names(summary, "dataframes"))
        self.assertEqual(set(), captured_names(summary, "models"))
        self.assertTrue(source_contains(summary, "deliberate_empty_features", "not-a-real-species"))
        self.assertTrue(source_contains(summary, "final_report_df", "species_summary_df.assign"))

    def test_openai_example_mock_scan_reports_agentic_unstructured_capture(self) -> None:
        previous_mode = os.environ.get("ETIQ_OPENAI_EXAMPLE_MODE")
        os.environ["ETIQ_OPENAI_EXAMPLE_MODE"] = "mock"
        try:
            summary = build_scan_summary(OPENAI_ENTRY)
        finally:
            if previous_mode is None:
                os.environ.pop("ETIQ_OPENAI_EXAMPLE_MODE", None)
            else:
                os.environ["ETIQ_OPENAI_EXAMPLE_MODE"] = previous_mode

        self.assertEqual([], summary["scan_errors"])
        self.assertEqual(OPENAI_ENTRY, summary["target_file"])
        self.assertTrue(summary["lineage_generated"]["json"])
        self.assertTrue(summary["lineage_generated"]["dot"])

        self.assertLessEqual(
            {"documents_df", "selected_documents_df"},
            captured_names(summary, "dataframes"),
        )
        self.assertIn("agent", captured_names(summary, "agents"))
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
            captured_names(summary, "unstructured"),
        )
        self.assertTrue(source_contains(summary, "user_prompt", "Answer the customer question"))
        self.assertTrue(source_contains(summary, "result", "agent.run_sync(user_prompt)"))

    def test_resolve_entry_file_reports_clear_workspace_errors(self) -> None:
        with self.assertRaisesRegex(EntryFileError, "not found inside workspace"):
            resolve_entry_file("library_functions_examples/does_not_exist.py")

        with self.assertRaisesRegex(EntryFileError, "must be a Python file"):
            resolve_entry_file("README.md")

    def test_write_summary_rejects_output_outside_workspace(self) -> None:
        outside_workspace = Path(__file__).resolve().parents[2] / "outside_summary.json"

        with self.assertRaisesRegex(EntryFileError, "must stay inside workspace"):
            write_summary({}, outside_workspace)


if __name__ == "__main__":
    unittest.main()
