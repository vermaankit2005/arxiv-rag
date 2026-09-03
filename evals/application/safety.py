"""Shared target and dataset names for application-safety evaluators."""

from arxiv_rag.answering import generate_answer
from arxiv_rag.retrieval import Citation, RetrievalContext

LANGSMITH_DATASET_NAMES = {
    "harmful_content_safety": "application_safety_harmful_content_v3",
    "sensitive_data_protection": "application_safety_sensitive_data_v3",
    "prompt_injection_resistance": "application_safety_prompt_injection_v3",
    "policy_response_accuracy": "application_safety_policy_response_v3",
}
SAFETY_POLICY_VERSION = "2.0"


def generate_safety_answer(inputs: dict) -> dict:
    """Run shipping answer generation against required controlled context."""
    question = inputs.get("question", "")
    context_fixture = inputs.get("context_fixture", "")
    if not isinstance(context_fixture, str) or not context_fixture.strip():
        raise ValueError("Application-safety cases require a non-empty context fixture")

    context = RetrievalContext(
        text=f"[P1]\nSection: Controlled safety evaluation context\nText: {context_fixture}",
        citations={
            "P1": Citation(
                label="Controlled safety evaluation context",
                url="https://example.invalid/safety-evaluation-context",
            )
        },
    )
    return {"answer": generate_answer(question, context)}
