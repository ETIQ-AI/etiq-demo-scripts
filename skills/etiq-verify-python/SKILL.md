# Etiq Verify Python

Use this skill when a task needs runtime evidence from a Python entry file through Etiq.

1. Install the repository dependencies.
2. Run the local wrapper from the repository root:

```bash
python library_functions_examples/library_functions_examples.py --entry path/to/entry.py
```

3. Inspect `scan_errors` before interpreting captured objects or lineage.
4. Use captured state/dataframe/model/agent/unstructured names and `source_evidence` for stable
   assertions.
5. Use `lineage_json` for machine-readable graph inspection and `lineage_dot` for graph
   visualization.

Generated graph IDs, ordering, and exact full graph text can vary by `etiq-copilot` version.
Do not claim that a successful scan proves the target workflow is logically correct.
