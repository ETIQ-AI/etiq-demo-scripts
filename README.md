# etiq-demo-scripts

Small runnable examples for exploring Etiq.

## Library Functions Demo

Install the local example dependencies:

```bash
python -m pip install -r requirements.txt
```

Use Python 3.10-3.13 for Etiq. On Windows with multiple interpreters, this may be:

```bash
py -3.11 -m pip install -r requirements.txt
py -3.11 library_functions_examples/library_functions_examples.py
```

Run the deterministic Iris lineage scan from the repository root:

```bash
python library_functions_examples/library_functions_examples.py
```

The command executes `library_functions_examples/iris_lineage_test.py` through
`etiq-copilot`, reports scan errors first, and prints a structured JSON summary with
captured object names, source evidence, and lineage strings where the installed package
version supports them.

Use `etiq-copilot>=2.3.1` for the agentic unstructured example; that version exposes
`list_agents()`, `get_agent_states()`, `get_unstructured_states()`, and JSON lineage output.

Run the OpenAI-backed pydantic-ai example with live credentials:

```bash
python library_functions_examples/library_functions_examples.py --entry library_functions_examples/openai_unstructured_example.py
```

For local scanner verification without a live OpenAI request, pass `--mock-openai` explicitly.
