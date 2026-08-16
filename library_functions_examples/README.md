# Library Functions Example

This folder demonstrates using the `etiq-copilot` Python library directly. The default
command scans a deterministic Iris pipeline with no network access, no secrets, and no
paid model credentials.

Run from the repository root:

```bash
python library_functions_examples/library_functions_examples.py
```

That command scans `library_functions_examples/iris_lineage_test.py` through
`DebuggerCodeScanner` and prints a structured JSON summary containing `target_file`,
`scan_errors`, captured state/dataframe/model/agent/unstructured names, counts, source
evidence, `deliberate_error_detected`, `lineage_json`, and `lineage_dot`.

Some `etiq-copilot` versions do not expose every optional API. When JSON lineage,
agent listing, or unstructured-state listing is unavailable, the wrapper leaves the
corresponding field empty and records the discrepancy in `optional_api_notes`.

To write the full JSON summary to a file:

```bash
python library_functions_examples/library_functions_examples.py --json-output library_functions_examples/latest_scan_summary.json
```

To scan a different workspace-relative entry file:

```bash
python library_functions_examples/library_functions_examples.py --entry library_functions_examples/iris_lineage_test.py
```

## Iris Target

`iris_lineage_test.py` uses the built-in scikit-learn Iris dataset as a local source and
then runs a pandas-only data pipeline. It converts measurements to a dataframe, joins
species labels, filters complete rows, creates a deliberately bad empty intermediate named
`deliberate_empty_features`, filters wider-petal rows, aggregates by species, and produces
`final_report_df`.

The default Iris scan should capture dataframe lineage only. `list_models()` should return
an empty list for this target.

Inspect `expected_scan_result.json`, `expected_openai_unstructured_scan_result.json`,
and `expected_lineage.json` for stable semantic examples. Treat generated graph IDs,
ordering, and exact full graph text as version-specific.

Curated lineage outputs for this target are saved as `iris_lineage.dot` and
`iris_lineage.png`.

## OpenAI/Agentic Unstructured Target

The OpenAI example is separate from the default Iris run. It follows Etiq's documented
agentic workflow shape: a `pydantic-ai` agent receives policy-document context, builds a
named prompt, runs the agent, and leaves named unstructured values such as `user_prompt`,
`result`, `result.output`, `response`, and `normalized_response_text` for Etiq to capture.

For a live OpenAI-backed agent call, install the dependencies, set `OPENAI_API_KEY` in
your shell environment, and run:

```bash
python library_functions_examples/library_functions_examples.py --entry library_functions_examples/openai_unstructured_example.py
```

Live runs require network access and may be subject to normal API billing and rate limits.
Do not hard-code API keys, private prompts, response IDs, or raw private model output in
source control. Set `OPENAI_MODEL` if you want a model other than `openai:gpt-4.1-mini`.

For local scanner verification without credentials or network access, use the explicit mock
flag:

```bash
python library_functions_examples/library_functions_examples.py --entry library_functions_examples/openai_unstructured_example.py --mock-openai
```

The mock path still runs through a `pydantic-ai` agent with `FunctionModel`, matching the
documented capture route while avoiding a live OpenAI request.

Curated mock-mode lineage outputs for this target are saved as
`openai_unstructured_lineage.dot` and `openai_unstructured_lineage.png`.

## Scanner Setup Rules

Install dependencies first:

```bash
python -m pip install -r requirements.txt
```

Configure required environment variables outside source control, for example
`OPENAI_API_KEY` for the live OpenAI agent path. Run the scanner against the Python entry
file that actually starts the workflow you want to observe. Keep prompts, responses, and
other unstructured values in named variables when you need Etiq to capture them where
supported.

Etiq's default parser config captures pandas dataframes/series and supported agent types
such as `pydantic-ai`. Only edit parser config when adding another agent framework. Do not
register primitive types such as `str`, `dict`, or `list`; the Etiq docs warn that doing so
can capture nearly everything and create very large lineage graphs.

Always inspect `scan_errors` before interpreting captured values or lineage. A successful
scan gives runtime evidence about observed objects and source locations; it does not prove
that the final result is correct.
