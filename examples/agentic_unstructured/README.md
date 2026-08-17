# Agentic Unstructured Example

This example scans a `pydantic-ai` workflow that builds a named prompt, runs an agent, and
leaves named unstructured values for Etiq to capture.

For a live OpenAI-backed agent call, install dependencies, set `OPENAI_API_KEY`, and run:

```bash
py -3.11 etiq_agent_wrapper.py \
  --entry examples/agentic_unstructured/openai_unstructured_example.py
```

For local scanner verification without credentials or network access, use mock mode:

```bash
py -3.11 etiq_agent_wrapper.py \
  --entry examples/agentic_unstructured/openai_unstructured_example.py \
  --mock-openai
```

Curated example outputs are stored in `expected_outputs/`.
