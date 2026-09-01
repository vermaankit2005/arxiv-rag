"""This checks whether the generated answer is factually correct.

It uses the required facts and their supporting passages. Missing facts are not
counted against the score here. Each answer gets 0, 0.25, 0.5, 0.75, or 1. The
final score is the average across all questions.
"""

from langsmith import Client
from openevals.llm import create_llm_as_judge  # pyright: ignore[reportMissingImports]

from evals.answering import context as evaluation_context
from arxiv_rag.ollama_config import get_generator_model, get_judge_model
from evals.judges import build_judge_model
from evals.answering.references import build_fact_references

DESCRIPTION = __doc__
LANGSMITH_DATASET_NAME = "generation_quality_dataset"
EXPERIMENT_PREFIX = "generation_correctness"
ALLOWED_SCORES = {0, 0.25, 0.5, 0.75, 1}
EXPERIMENT_METADATA = {
    "metric": "correctness",
    "dataset": LANGSMITH_DATASET_NAME,
    "generator_model": get_generator_model(),
    "judge_model": get_judge_model(),
    "judge_thinking": "disabled",
    "generator_thinking": "disabled",
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

judge_model = build_judge_model()

correctness_judge = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    judge=judge_model,
    choices=[0, 0.25, 0.5, 0.75, 1],
)


def _build_fact_references(context_passages: list[dict], required_facts: list[dict]) -> list[dict]:
    return build_fact_references(context_passages, required_facts)


def evaluate_correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Judge factual correctness without treating omitted facts as incorrect."""
    references = _build_fact_references(inputs.get("context_passages", []), reference_outputs.get("required_facts", []))
    if not references:
        raise ValueError("Correctness evaluation requires at least one required fact")

    result = correctness_judge(
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
        evaluation_context.generate_answer_for_evaluation,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[evaluate_correctness],
        metadata=EXPERIMENT_METADATA,
        experiment_prefix=EXPERIMENT_PREFIX,
        description=DESCRIPTION,
        max_concurrency=4
    )

if __name__ == "__main__":
    run_correctness()
