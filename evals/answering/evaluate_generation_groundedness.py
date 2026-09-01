"""This checks whether every factual claim in the generated answer is supported by the given passages.

It penalizes claims that are not backed by the supplied context. Each answer
gets 0, 0.25, 0.5, 0.75, or 1. The final score is the average across all questions.
"""

from langsmith import Client
from openevals import prompts  # pyright: ignore[reportMissingImports]
from openevals.llm import create_llm_as_judge  # pyright: ignore[reportMissingImports]

from evals.answering import context as evaluation_context
from arxiv_rag.ollama_config import get_generator_model, get_judge_model
from evals.judges import build_judge_model

DESCRIPTION = __doc__
LANGSMITH_DATASET_NAME = "generation_quality_dataset"
EXPERIMENT_PREFIX = "generation_groundedness"

EXPERIMENT_METADATA = {
    "metric": "groundedness",
    "dataset": LANGSMITH_DATASET_NAME,
    "generator_model": get_generator_model(),
    "embedding_model": "qwen3-embedding:4b",
    "corpus": "12-papers-384-documents",
    "vector_db": "chroma",
    "judge_model": get_judge_model(),
    "judge_thinking": "disabled",
    "generator_thinking": "disabled",
}

judge_model = build_judge_model()

groundedness_judge = create_llm_as_judge(
    prompt=prompts.RAG_GROUNDEDNESS_PROMPT,
    feedback_key="groundedness",
    judge=judge_model,
    choices=[0, 0.25, 0.5, 0.75, 1],
)

def evaluate_groundedness(inputs: dict, outputs: dict) -> dict:
    context = {
        "documents": [
            passage["text"]
            for passage in inputs.get("context_passages", [])
        ]
    }

    return groundedness_judge(
        context=context,
        outputs={"answer": outputs.get("answer", "")},
    )


def run_groundedness() -> None:
    """Evaluate the retriever by fetching documents for a given question and log to LangSmith."""
    client = Client()
    client.evaluate(
        evaluation_context.generate_answer_for_evaluation,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[evaluate_groundedness],
        metadata=EXPERIMENT_METADATA,
        experiment_prefix=EXPERIMENT_PREFIX,
        description=DESCRIPTION,
        max_concurrency=4
    )

if __name__ == "__main__":
    run_groundedness()
