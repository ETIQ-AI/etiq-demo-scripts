# etiq-demo-scripts

Small runnable examples for exploring Etiq scans, lineage artifacts, and a protected
workflow pattern that forces agents through the Etiq runner.

## Install

Use Python 3.10-3.13 for Etiq. In this workspace, Python 3.11 is the known-good choice:

```bash
py -3.11 -m pip install -r requirements.txt
```

## Main Paths

- Agent-facing generic wrapper scans: `etiq_agent_wrapper.py`
- Underlying generic scan CLI: `scripts/scan_entry.py`
- Structured-data example: `examples/structured_data/`
- Agentic unstructured example: `examples/agentic_unstructured/`
- Forced protected workflow: `workflows/protected_pipeline/run.py`
- Human-friendly run report: `scripts/generate_run_report.py`
- Durable docs: `docs/`
- Generated artifacts: `.etiq/artifacts/`

## Generic Scan

Run the deterministic structured-data example from the repository root:

```bash
py -3.11 etiq_agent_wrapper.py
```

The command prints only a small pointer object on stdout:

```json
{"result_path": "scan-result.json", "status": "..."}
```

Read the full result from `.etiq/artifacts/structured_data/scan-result.json`. Lineage is
written separately as `.etiq/artifacts/structured_data/lineage.json` or
`.etiq/artifacts/structured_data/lineage.dot`, depending on `--lineage-format`.

## Forced Runner

Run the protected workflow through Etiq:

```bash
py -3.11 workflows/protected_pipeline/run.py
```

Direct execution of `workflows/protected_pipeline/workflow.py` is intentionally blocked.
The supported runner sets `RUNNING_UNDER_ETIQ=1` only while invoking Etiq, writes
`scan-result.json` plus the requested lineage artifact, then validates that the result is
completed, current for the entry file, and free of scan errors.

For the one-command self-check:

```bash
py -3.11 workflows/protected_pipeline/verify.py
```

## Run Report

After Codex, Claude Code, Cursor, or another agent works on this repo, generate the
human-friendly "what happened?" report:

```bash
py -3.11 scripts/generate_run_report.py
```

Open `.etiq/reports/latest-run-report.md`. The report links to each known job's
`scan-result.json` and lineage artifact, and uses PASS/WARN/FAIL/NOT RUN labels so a
human can see whether the Etiq gate passed.

Example report excerpt:

```markdown
# Etiq Run Report

Generated: 2026-08-17 13:39 +01:00
Workspace: <repo-root>

## Summary

| Job | Gate | Etiq Evidence | Status | Artifacts |
| --- | --- | --- | --- | --- |
| protected_pipeline | <span style="color:green">PASS</span> | <span style="color:green">FOUND</span> | completed | [scan-result](../artifacts/protected_pipeline/scan-result.json), [lineage](../artifacts/protected_pipeline/lineage.json) |
| structured_data_example | <span style="color:green">PASS</span> | <span style="color:green">FOUND</span> | completed | [scan-result](../artifacts/structured_data/scan-result.json), [lineage](../artifacts/structured_data/lineage.json) |
| agentic_unstructured_example_mock | <span style="color:green">PASS</span> | <span style="color:green">FOUND</span> | completed | [scan-result](../artifacts/agentic_unstructured_example_mock/scan-result.json), [lineage](../artifacts/agentic_unstructured_example_mock/lineage.json) |

## structured_data_example

Gate: <span style="color:green">PASS</span>
Command: `py -3.11 etiq_agent_wrapper.py`
Output: `.etiq/artifacts/structured_data/`

Evidence:
- `status`: completed
- `scan_errors`: none
- `entry_file_path`: examples/structured_data/iris_lineage_test.py
- dataframe states: 8; unique dataframe names: clean_measurements_df, deliberate_empty_features, final_report_df, iris_df, iris_with_species_df, species_lookup_df, species_summary_df, wide_petal_df
- model states: 0; unique model names: none
- agent states: 0; unique agent names: none
- unstructured states: 2; unique unstructured names: iris, measurement_columns

Assessment:
- The available Etiq evidence is acceptable for this job.
```

## Docs

- `docs/WRAPPER_USAGE.md`: ad hoc scans through the wrapper.
- `docs/FORCED_RUNNER.md`: protected workflow pattern.
- `docs/USING_AGENT_WITH_ETIQ.md`: using Codex, Claude Code, or another agent with Etiq.
- `docs/IRIS_CODEX_ETIQ_EXAMPLE.md`: concrete Iris pipeline prompt and artifact checks.
- `docs/REPOSITORY_ORGANIZATION.md`: current layout and tracked-vs-generated rules.
