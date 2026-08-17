# Iris Pipeline With Codex And Etiq

Use this as a concrete example when you want Codex to build a Python data pipeline and
show artifact evidence that it ran through Etiq.

## Goal

Ask Codex to use the Iris dataset, create a pipeline with data transformations, and avoid
model training or prediction. The run must go through the forced Etiq runner.

## What You Need First

From the repository root:

```bash
py -3.11 -m pip install -r requirements.txt
```

The protected workflow files already exist:

```text
workflows/protected_pipeline/workflow.py
workflows/protected_pipeline/run.py
```

Generated Etiq artifacts will be written to:

```text
.etiq/artifacts/protected_pipeline/
```

## Prompt To Give Codex

Paste this into a new Codex chat:

```text
Build a protected Etiq-observed Python data pipeline using the Iris dataset.

Requirements:
- Use the forced Etiq runner pattern in this repo.
- Put the workflow logic in workflows/protected_pipeline/workflow.py.
- Use the Iris dataset from sklearn.datasets.
- Create dataframe transformations only.
- Do not train, fit, predict, or evaluate any model.
- Include useful named pandas dataframes, such as raw iris data, species lookup,
  labelled measurements, filtered rows, grouped summaries, and a final report dataframe.
- Do not run workflows/protected_pipeline/workflow.py directly.
- Run only py -3.11 workflows/protected_pipeline/run.py.

After running, inspect .etiq/artifacts/protected_pipeline/scan-result.json.
Confirm status is completed, scan_errors is [], lineage_path exists, and entry_file_path
is workflows/protected_pipeline/workflow.py. Summarize the captured dataframes and lineage
artifact path in your final response.
```

## What Codex Should Run

The gate command is:

```bash
py -3.11 workflows/protected_pipeline/run.py
```

Direct execution should fail:

```bash
py -3.11 workflows/protected_pipeline/workflow.py
```

If direct execution succeeds, the workflow is not protected correctly.

## What To Check After Codex Finishes

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
- `captured_objects.dataframes` includes the expected dataframe names.
- `captured_objects.models` is empty.
- `source_evidence` includes source snippets for the dataframe transformations.

Then open the lineage artifact named by `lineage_path`, usually:

```text
.etiq/artifacts/protected_pipeline/lineage.json
```

Use the lineage file to see how the dataframes relate to each other.

## One-Command Human Check

Run:

```bash
py -3.11 workflows/protected_pipeline/verify.py
```

This confirms direct execution is blocked, the forced runner completes through Etiq,
validation accepts the artifacts, and the lineage file exists.

## Acceptance Rule

Treat Etiq as the gate. Do not accept the agent's work just because it says the pipeline
ran. Accept it only when the supported runner passes and the artifacts show a fresh,
completed Etiq scan with no scan errors.
