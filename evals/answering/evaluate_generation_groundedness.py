from langsmith import Client
from openevals import prompts  # pyright: ignore[reportMissingImports]
from openevals.llm import create_llm_as_judge  # pyright: ignore[reportMissingImports]

from arxiv_rag.retrieval import RetrievalContext
from evals.answering import context as evaluation_context
from evals.answering.judges import build_judge_model

LANGSMITH_DATASET_NAME = "generation_quality_dataset"
EXPERIMENT_PREFIX = "generation_groundedness"
JUDGE_MODEL_NAME = "gemma4:26b"

EXPERIMENT_METADATA = {
    "metric": "groundedness",
    "dataset": LANGSMITH_DATASET_NAME,
    "generator_model": "gemma4:26b",
    "embedding_model": "qwen3-embedding:4b",
    "corpus": "12-papers-384-documents",
    "vector_db": "chroma",
    "judge_model": JUDGE_MODEL_NAME,
}

judge_model = build_judge_model(JUDGE_MODEL_NAME)

groundedness_judge = create_llm_as_judge(
    prompt=prompts.RAG_GROUNDEDNESS_PROMPT,
    feedback_key="groundedness",
    judge=judge_model,
    choices=[0, 0.25, 0.5, 0.75, 1],
)

def _build_passages(context_passages: list[dict]) -> RetrievalContext:
    """Build a list of passages with text and citation information."""
    return evaluation_context.build_retrieval_context(context_passages, preserve_passage_ids=False)


def generate_answer_for_evaluation(inputs: dict) -> dict:
    """Generate an answer for a given question and log to LangSmith."""
    return evaluation_context.generate_answer_for_evaluation(inputs, _build_passages)

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
        generate_answer_for_evaluation,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[evaluate_groundedness],
        metadata=EXPERIMENT_METADATA,
        experiment_prefix=EXPERIMENT_PREFIX,
        description=(
            "Evaluate the groundedness of generated answers by comparing them to reference evidence units."
        ),
        max_concurrency=4
    )

if __name__ == "__main__":
    run_groundedness()
