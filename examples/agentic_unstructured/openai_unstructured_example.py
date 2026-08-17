from __future__ import annotations

import os
from random import random

import pandas as pd
from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart, models
from pydantic_ai.models.function import AgentInfo, FunctionModel


def mock_model_call(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[
            TextPart(
                "The customer can request a refund within 30 days if the item is unused. "
                f"reference={random():.6f}"
            )
        ]
    )


def build_policy_documents() -> pd.DataFrame:
    raw_documents = {
        "doc_id": ["policy-001", "policy-002", "policy-003"],
        "title": ["Refund policy", "Shipping policy", "Warranty policy"],
        "text": [
            "Customers can request a refund within 30 days if the item is unused.",
            "Express shipping usually arrives within two business days.",
            "Electronics include a one-year warranty for manufacturing defects.",
        ],
    }
    documents_df = pd.DataFrame(raw_documents)
    documents_df["normalized_text"] = documents_df["text"].str.strip().str.replace(
        r"\s+",
        " ",
        regex=True,
    )
    return documents_df


def select_relevant_documents(documents_df: pd.DataFrame, question: str) -> pd.DataFrame:
    keywords = ("refund", "return", "unused", "30")
    relevant_mask = documents_df["normalized_text"].str.lower().apply(
        lambda text: any(keyword in text for keyword in keywords)
    )
    selected_documents_df = documents_df.loc[relevant_mask].copy()
    return selected_documents_df


def build_agent() -> Agent:
    mode = os.environ.get("ETIQ_OPENAI_EXAMPLE_MODE", "live").strip().lower()
    if mode == "mock":
        models.ALLOW_MODEL_REQUESTS = False
        model = FunctionModel(function=mock_model_call)
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is required for the live OpenAI agent path. "
                "For local scanner verification, rerun with --mock-openai."
            )
        model = os.environ.get("OPENAI_MODEL", "openai:gpt-4.1-mini")

    agent = Agent(
        model=model,
        system_prompt=(
            "You answer customer policy questions using only the supplied policy excerpts. "
            "Keep the answer to one sentence."
        ),
    )
    return agent


def query_policy_agent(
    agent: Agent,
    selected_documents_df: pd.DataFrame,
    customer_question: str,
) -> str:
    document_context = selected_documents_df[
        ["doc_id", "title", "normalized_text"]
    ].to_string(index=False)
    user_prompt = (
        "Answer the customer question using only the policy excerpts.\n\n"
        f"Question: {customer_question}\n\n"
        f"Policy excerpts:\n{document_context}"
    )
    result = agent.run_sync(user_prompt)
    response = result.output
    return response


def normalize_response_text(response: str) -> str:
    normalized_response_text = " ".join(response.strip().split())
    return normalized_response_text


documents_df = build_policy_documents()
customer_question = "Can I return an unused item after two weeks?"
selected_documents_df = select_relevant_documents(documents_df, customer_question)
agent = build_agent()
response = query_policy_agent(agent, selected_documents_df, customer_question)
normalized_response_text = normalize_response_text(response)
final_answer_record = {
    "question": customer_question,
    "answer": normalized_response_text,
    "cited_doc_ids": selected_documents_df["doc_id"].tolist(),
}
