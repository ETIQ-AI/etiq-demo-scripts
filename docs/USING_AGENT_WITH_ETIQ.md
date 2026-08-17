# Using An Agent With Etiq

Use this when asking Codex, Claude Code, Cursor, or another coding agent to build or modify
a Python workflow that must be observed by Etiq.

## Quick Flow

1. Tell the agent to use the forced Etiq runner.
2. Make sure the workflow file cannot be run directly.
3. Let the agent run only the supported runner command.
4. Generate the run report and inspect the linked `scan-result.json` and lineage artifacts.

## Enforcement Model

Agent instructions are not enforcement. Codex, Claude Code, or another agent can still try
to run other commands. Treat Etiq as an acceptance gate: the work is not done until the
supported runner completes and produces fresh, valid artifacts.

The gate is:

```bash
py -3.11 workflows/protected_pipeline/run.py
```

Accept the work only if validation passes and the resulting `scan-result.json` and lineage
artifact match the current workflow.

## Prompt To Paste Into A New Agent Chat

```text
Use the forced Etiq runner pattern in this repo.

Put or update workflow logic in:
workflows/protected_pipeline/workflow.py

Do not run that workflow file directly. It should fail if run directly:
py -3.11 workflows/protected_pipeline/workflow.py

Run the workflow only through:
py -3.11 workflows/protected_pipeline/run.py

After running it, inspect:
.etiq/artifacts/protected_pipeline/scan-result.json

Then generate the human-friendly run report:
py -3.11 scripts/generate_run_report.py

Confirm status is completed, scan_errors is [], lineage_path exists, and
entry_file_path is workflows/protected_pipeline/workflow.py. Summarize the captured
objects, lineage artifact path, and `.etiq/reports/latest-run-report.md` in your final
response.
```

## Commands To Check Yourself

Direct execution should fail:

```bash
py -3.11 workflows/protected_pipeline/workflow.py
```

Supported runner should pass:

```bash
py -3.11 workflows/protected_pipeline/run.py
```

One-command self-check:

```bash
py -3.11 workflows/protected_pipeline/verify.py
```

Human-friendly "what happened?" report:

```bash
py -3.11 scripts/generate_run_report.py
```

## What To Inspect After

Open the generated report:

```text
.etiq/reports/latest-run-report.md
```

It links to each known job's `scan-result.json` and lineage artifact. Use its
PASS/WARN/FAIL/NOT RUN labels to see whether the Etiq gate passed after Codex, Claude Code,
Cursor, or another agent worked on the repo.

Open:

```text
.etiq/artifacts/protected_pipeline/scan-result.json
```

Check:

- `status` is `completed`.
- `scan_errors` is `[]`.
- `entry_file_path` is `workflows/protected_pipeline/workflow.py`.
- `entry_file_hash` is present.
- `lineage_path` points to an existing file.
- `captured_objects.dataframes` includes the expected dataframes.
- `source_evidence` shows source snippets for the captured objects.

For lineage, open the file named by `lineage_path`, usually:

```text
.etiq/artifacts/protected_pipeline/lineage.json
```

If you want Graphviz-compatible lineage, run with `--lineage-format dot` and inspect
`lineage.dot`.

## Treat As Failed If

- direct execution of `workflow.py` succeeds;
- the runner exits nonzero;
- `scan_errors` is not empty;
- `status` is not `completed`;
- `lineage_path` is missing or points to a missing file;
- `entry_file_hash` is stale after workflow changes.

For stronger enforcement, put the supported runner command in CI and reject any change where
the Etiq gate fails.
