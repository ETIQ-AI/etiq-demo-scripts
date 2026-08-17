# Repository Organization

The project has two supported Etiq paths:

- Generic wrapper scans: `etiq_agent_wrapper.py`, delegated to `scripts/scan_entry.py`.
- Forced protected workflow scans: `workflows/protected_pipeline/run.py`.

## Layout

```text
etiq_runner/
  wrapper.py
  validation.py

workflows/
  protected_pipeline/
    workflow.py
    run.py
    verify.py

examples/
  structured_data/
    iris_lineage_test.py
    expected_outputs/
  agentic_unstructured/
    openai_unstructured_example.py
    expected_outputs/

scripts/
  scan_entry.py

etiq_agent_wrapper.py

tests/
  test_etiq_runner.py

docs/
  FORCED_RUNNER.md
  WRAPPER_USAGE.md
  USING_AGENT_WITH_ETIQ.md
  REPOSITORY_ORGANIZATION.md
```

## Tracked Files

Track source code, tests, docs, and curated example outputs. The curated lineage examples
stay in:

```text
examples/structured_data/expected_outputs/
examples/agentic_unstructured/expected_outputs/
```

## Ignored Files

Generated scan outputs should stay out of source folders and live under:

```text
.etiq/artifacts/
.etiq/test_outputs/
```

Keep `codex_instructions/` and `instructions_between_agents/` as ignored coordination
context rather than primary user documentation.
