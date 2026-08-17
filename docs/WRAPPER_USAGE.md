# Wrapper Usage

Use this when you want to run a Python entry file through Etiq without enforcing the
protected workflow pattern.

## Files

- `etiq_runner/wrapper.py` wraps `DebuggerCodeScanner`.
- `etiq_runner/validation.py` validates saved scan artifacts.
- `etiq_agent_wrapper.py` is the agent-facing compatibility command.
- `scripts/scan_entry.py` is the underlying CLI for scanning any workspace-relative
  Python file and is delegated to by `etiq_agent_wrapper.py`.
- `.etiq/artifacts/structured_data/` is the default structured-data output directory.

## Command

From the repository root:

```bash
py -3.11 etiq_agent_wrapper.py \
  --workspace-root . \
  --entry examples/structured_data/iris_lineage_test.py \
  --output-dir .etiq/artifacts/structured_data \
  --lineage-format json
```

## Output Contract

The command prints only a small pointer object on stdout. Read the actual results from:

```text
<output-dir>/scan-result.json
<output-dir>/lineage.json
```

or, for DOT lineage:

```text
<output-dir>/lineage.dot
```

Always inspect `status` and `scan_errors` before using captured objects or lineage.
