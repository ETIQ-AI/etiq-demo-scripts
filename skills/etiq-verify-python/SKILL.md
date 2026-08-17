# Etiq Verify Python

Use this skill when a task needs runtime evidence from executed Python code through Etiq.
Etiq is useful for execution lineage, intermediate-state verification, debugging observed
transformations, and retracing how a Python workflow produced or transformed supported
objects.

Do not use Etiq as the primary tool for ordinary logging, static linting, non-Python code
review, or correctness certification. Never claim that a scan or lineage graph proves a
program, final answer, model output, or business conclusion is correct.

## Required Capture Path

Code must run through `etiq-copilot` for Etiq evidence to be captured. In this repo, use the
shared wrapper command. Do not implement another scanner, runner, or raw
`DebuggerCodeScanner` helper in the skill.

For an ad hoc Python workflow entry file, run from the repository root:

```bash
python etiq_agent_wrapper.py \
  --workspace-root . \
  --entry path/to/entry.py \
  --output-dir .etiq/artifacts/structured_data \
  --lineage-format json
```

For the default deterministic structured demo, this shorter command is equivalent:

```bash
python etiq_agent_wrapper.py
```

For a protected workflow, use the supported runner instead of running the workflow file
directly:

```bash
python workflows/protected_pipeline/run.py
```

## Entry File Selection

Select the smallest workspace-relative `.py` file that actually executes the workflow path
under review. Prefer a documented repo runner or CLI entry when one exists.

Do not scan a helper module that only defines functions unless that module also executes the
target workflow when run. Do not scan tests, notebooks, generated artifacts, or unrelated
wrappers unless the user's question is specifically about those files.

For protected workflows, run the supported runner and then verify that `entry_file_path` in
`scan-result.json` matches the protected workflow file.

## Inspection Order

1. Run the selected Python workflow through the shared Etiq wrapper or protected runner.
2. Treat console output only as a pointer to the result file.
3. Open `<output-dir>/scan-result.json`.
4. Inspect `status` and `scan_errors` before interpreting captured objects or lineage.
5. If `status` is not `completed` or `scan_errors` is non-empty, report that first and do not
   present lineage as complete evidence.
6. Inspect `captured_objects` and `source_evidence` for stable object names, source snippets,
   scopes, and line numbers.
7. Read the separate lineage artifact named by `lineage_path` only after the scan result is
   understood.

## Evidence To Report

Cite the selected entry file, scan status, scan errors, captured object names, relevant
`source_evidence`, and the lineage artifact path. When debugging the structured demo, the
deliberate issue is the captured dataframe named `deliberate_empty_features`, produced by
the `not-a-real-species` filter.

State limitations clearly: Etiq only observed code that executed through the scanner;
unsupported object types or APIs may not be captured; generated graph IDs, ordering, and
exact graph text can vary by `etiq-copilot` version; lineage supports debugging and
retracing evidence, but it does not prove correctness.
