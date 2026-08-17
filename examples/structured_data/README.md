# Structured Data Example

This example scans a deterministic Iris dataframe pipeline with no network access, no
secrets, and no paid model credentials.

Run from the repository root:

```bash
py -3.11 etiq_agent_wrapper.py
```

That command scans `examples/structured_data/iris_lineage_test.py` through
`etiq_runner/wrapper.py`, writes `.etiq/artifacts/structured_data/scan-result.json`, and
writes a separate lineage artifact referenced by `lineage_path`.

To choose a different output directory:

```bash
py -3.11 etiq_agent_wrapper.py --output-dir .etiq/artifacts/structured_data
```

Use `--lineage-format json` for machine-readable graph output and `--lineage-format dot`
for Graphviz-compatible output. The graph text is not embedded in `scan-result.json`.

## Iris Target

`iris_lineage_test.py` uses the built-in scikit-learn Iris dataset as a local source and
then runs a pandas-only data pipeline. It converts measurements to a dataframe, joins
species labels, filters complete rows, creates a deliberately bad empty intermediate named
`deliberate_empty_features`, filters wider-petal rows, aggregates by species, and produces
`final_report_df`.

The default scan should capture dataframe lineage only. `list_models()` should return an
empty list for this target.

Curated semantic examples and lineage outputs are stored in `expected_outputs/`. Treat
generated graph IDs, ordering, and exact full graph text as version-specific.
