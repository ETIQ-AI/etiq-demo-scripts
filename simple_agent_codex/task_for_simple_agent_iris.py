from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from simple_agent.simple_agent import AgentConfig, run_pipeline_agent
except ImportError:
    from simple_agent import AgentConfig, run_pipeline_agent


def _default_output_script() -> Path:
    return Path(__file__).with_name("iris_generated_pipeline.py")


class TaskSpec(BaseModel):
    task_id: str
    title: str
    dataset_name: list[str] = Field(default_factory=list)
    dataset_loader: str | None = None
    acceptance_criteria: list[str] = Field(default_factory=list)
    prompt: str


def build_iris_task_prompt() -> str:
    return "\n".join(
        [
            "Build a complete runnable Python script for a simple supervised data science task on the iris dataset.",
            "Use sklearn.datasets.load_iris() as the dataset source. Do not download any external data.",
            "Use pandas for the main dataset artifacts.",
            "Create one main non-empty root dataframe that contains the iris features and target.",
            "Do not create an earlier empty placeholder dataframe before the main dataset dataframe.",
            "Create a reproducible train/test split using a fixed random_state.",
            "Create downstream artifacts for training and testing data, including feature and target artifacts.",
            "Train a baseline classifier on the training artifacts.",
            "Create predictions and a non-empty metrics dataframe that contains at least an accuracy metric.",
            "Keep the important artifacts assigned to named variables so they can be captured by lineage tooling.",
            "Return only the Python script.",
        ]
    )


def build_iris_task_spec() -> TaskSpec:
    return TaskSpec(
        task_id="iris_baseline_pipeline_v1",
        title="Iris Baseline Pipeline",
        dataset_name=["iris"],
        dataset_loader="sklearn.datasets.load_iris()",
        acceptance_criteria=[
            "The generated script executes without runtime errors.",
            "etiq_copilot captures dataframe artifacts from the script.",
            "The first dataframe artifact is the main non-empty dataset dataframe rather than an empty placeholder.",
            "All checked artifacts in the reachable lineage are non-empty.",
            "The generated pipeline includes predictions and a non-empty metrics dataframe.",
        ],
        prompt=build_iris_task_prompt(),
    )


IRIS_TASK = build_iris_task_spec()


def build_iris_agent_config(
    model: str,
    *,
    max_rewrites: int = 5,
    output_script: Path | None = None,
) -> AgentConfig:
    return AgentConfig(
        task=IRIS_TASK.prompt,
        model=model,
        max_rewrites=max_rewrites,
        output_script=output_script or _default_output_script(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the iris task setup or launch the simple_agent on that task."
    )
    parser.add_argument("--model", help="Model name to use when launching the agent.")
    parser.add_argument(
        "--max-rewrites",
        type=int,
        default=5,
        help="Rewrite budget to use if --model is provided.",
    )
    parser.add_argument(
        "--output-script",
        type=Path,
        default=_default_output_script(),
        help="Output script path to use if --model is provided.",
    )
    parser.add_argument(
        "--print-task",
        action="store_true",
        help="Print the iris task spec before exiting or before launching the agent.",
    )
    args = parser.parse_args()

    if args.print_task or not args.model:
        print("Iris Task Spec")
        print(IRIS_TASK.model_dump_json(indent=2))

    if not args.model:
        return

    config = build_iris_agent_config(
        args.model,
        max_rewrites=args.max_rewrites,
        output_script=args.output_script,
    )
    outcome = run_pipeline_agent(config)
    print()
    print("Agent Outcome")
    print(outcome.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
