from collections.abc import Callable
import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama  # pyright: ignore[reportMissingImports]
from langsmith import Client
from openevals.llm import create_llm_as_judge  # pyright: ignore[reportMissingImports]

from evals.answering.evaluate_generation_citation_support import generate_answer_for_citation_support

load_dotenv()

LANGSMITH_DATASET_NAME = "generation_quality_dataset"
EXPERIMENT_PREFIX = "generation_correctness"
JUDGE_MODEL_NAME = "qwen3.8:27b"
ALLOWED_SCORES = {0, 0.25, 0.5, 0.75, 1}
EXPERIMENT_METADATA = {
    "metric": "correctness",
    "dataset": LANGSMITH_DATASET_NAME,
    "generator_model": "qwen3.8:27b",
    "judge_model": JUDGE_MODEL_NAME,
}

CORRECTNESS_PROMPT = """
You are checking whether a generated answer is factually correct for a question.

Use only the supplied required facts and their supporting passages. Penalize
incorrect values, contradictions, and statements that distort the expected
answer. Do not penalize an answer only because it omits a required fact;
completeness is evaluated separately. Ignore citation-marker formatting.

Question and frozen reference:
{inputs}

Generated answer:
{outputs}

Return one of these scores:
- 1: every substantive statement answering the question is correct.
- 0.75: correct overall, with one minor factual imprecision.
- 0.5: a meaningful mix of correct and incorrect information.
- 0.25: mostly incorrect, but contains a small correct part.
- 0: entirely incorrect or directly contradicts the reference.
"""

Judge = Callable[..., dict]

judge_model = ChatOllama(
    model=JUDGE_MODEL_NAME,
    base_url=os.environ["OLLAMA_BASE_URL"],
    temperature=0,
)

correctness_judge = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    judge=judge_model,
    choices=[0, 0.25, 0.5, 0.75, 1],
)


def _build_fact_references(context_passages: list[dict], required_facts: list[dict]) -> list[dict]:
    """Attach each frozen required fact to its frozen supporting passage text."""
    passages_by_id = {passage["id"]: passage["text"] for passage in context_passages}
    references = []

    for required_fact in required_facts:
        supporting_ids = required_fact.get("supporting_passage_ids", [])
        unknown_ids = sorted(set(supporting_ids) - passages_by_id.keys())
        if unknown_ids:
            unknown = ", ".join(unknown_ids)
            raise ValueError(f"Required fact {required_fact.get('id', '<unknown>')} used unknown passage IDs: {unknown}")

        references.append({
            "id": required_fact["id"],
            "fact": required_fact["fact"],
            "supporting_passages": [
                {"id": passage_id, "text": passages_by_id[passage_id]}
                for passage_id in supporting_ids
            ],
        })

    return references


def evaluate_correctness(
    inputs: dict, outputs: dict, reference_outputs: dict, judge: Judge = correctness_judge
) -> dict:
    """Judge factual correctness without treating omitted facts as incorrect."""
    references = _build_fact_references(
        inputs.get("context_passages", []),
        reference_outputs.get("required_facts", []),
    )
    if not references:
        raise ValueError("Correctness evaluation requires at least one required fact")

    result = judge(
        inputs={
            "question": inputs.get("question", ""),
            "required_facts": references,
        },
        outputs={"answer": outputs.get("answer", "")},
    )
    score = result.get("score")
    if score not in ALLOWED_SCORES:
        raise ValueError(f"Correctness judge returned an invalid score: {score!r}")

    return {
        "key": "correctness",
        "score": score,
        "comment": result.get("comment", "Correctness was judged against the frozen required facts."),
    }


def run_correctness() -> None:
    """Run correctness against the frozen generation dataset in LangSmith."""
    client = Client()
    client.evaluate(
        generate_answer_for_citation_support,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[evaluate_correctness],
        metadata=EXPERIMENT_METADATA,
        experiment_prefix=EXPERIMENT_PREFIX,
        description=(
            "Judge whether each generated answer is factually consistent with the "
            "frozen required facts and their supporting passages. Scores are restricted "
            "to 0, 0.25, 0.5, 0.75, or 1. Missing facts are not penalized here because "
            "they are measured by completeness."
        ),
    )


if __name__ == "__main__":
    run_correctness()
