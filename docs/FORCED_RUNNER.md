# Forced Runner

Use this when a workflow must not be run directly by Codex, Claude Code, CI, or a human.

## Files

- `workflows/protected_pipeline/run.py` is the supported command.
- `workflows/protected_pipeline/workflow.py` contains the workflow and refuses direct
  execution.
- `workflows/protected_pipeline/verify.py` is a user-facing self-check.
- `etiq_runner/wrapper.py` sets `RUNNING_UNDER_ETIQ=1` only during Etiq scanning.
- `etiq_runner/validation.py` validates the saved scan artifact and lineage artifact.

## Supported Command

From the repository root:

```bash
py -3.11 workflows/protected_pipeline/run.py
```

That command runs `workflows/protected_pipeline/workflow.py` once through Etiq, writes
artifacts, and validates the result.

By default, forced-runner artifacts are written to:

```text
.etiq/artifacts/protected_pipeline/
```

## Direct Execution Must Fail

This command is intentionally unsupported:

```bash
py -3.11 workflows/protected_pipeline/workflow.py
```

It should fail with:

```text
This workflow must be run through the Etiq wrapper.
```

## Guard Boundary

The local guard uses the public environment variable `RUNNING_UNDER_ETIQ=1`. This prevents
accidental direct runs, including ordinary agent or shell attempts to execute the workflow
file directly. It is not a security boundary: a caller can bypass this local guard by
manually setting that environment variable.

The stronger enforcement is the supported command plus validation. CI should run the
supported command and reject missing, stale, partial, failed, or mismatched scan artifacts.

## Validation Rules

`run.py` validates that:

- `scan-result.json` exists.
- `status == "completed"`.
- `scan_errors == []`.
- `lineage_path` points to an existing lineage artifact.
- `lineage_format` matches the requested format.
- `entry_file_path` and `entry_file_hash` match the current protected workflow.

## Self-Check

Run:

```bash
py -3.11 workflows/protected_pipeline/verify.py
```

Expected output:

```text
PASS direct execution is blocked
PASS forced runner completed through Etiq
PASS validation accepts the current scan artifacts
PASS scan-result.json and lineage.json are valid
```
