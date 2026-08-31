from langsmith import Client
from openevals.llm import create_llm_as_judge  # pyright: ignore[reportMissingImports]

from evals.answering import context as evaluation_context
from evals.answering.judges import build_judge_model
from evals.answering.references import build_fact_references

LANGSMITH_DATASET_NAME = "generation_quality_dataset"
EXPERIMENT_PREFIX = "generation_completeness"
JUDGE_MODEL_NAME = "gemma4:26b"
EXPERIMENT_METADATA = {
    "metric": "completeness",
    "dataset": LANGSMITH_DATASET_NAME,
    "generator_model": "gemma4:26b",
    "judge_model": JUDGE_MODEL_NAME,
}

COMPLETENESS_PROMPT = """
You are checking whether a generated answer covers one required fact.

The answer covers the fact only when it communicates the complete factual
meaning, either exactly or by a clear paraphrase. A citation marker alone does
not cover a fact. Use the supporting passages only to interpret the required
fact. Do not use outside knowledge.

Question, required fact, and supporting passages:
{inputs}

Generated answer:
{outputs}
"""

judge_model = build_judge_model(JUDGE_MODEL_NAME)

completeness_judge = create_llm_as_judge(
    prompt=COMPLETENESS_PROMPT,
    feedback_key="required_fact_coverage",
    judge=judge_model,
    choices=[False, True],
)

def _build_fact_references(context_passages: list[dict], required_facts: list[dict]) -> list[dict]:
    return build_fact_references(context_passages, required_facts)


def evaluate_completeness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Return covered frozen required facts divided by all required facts."""
    references = _build_fact_references(
        inputs.get("context_passages", []),
        reference_outputs.get("required_facts", []),
    )
    if not references:
        raise ValueError("Completeness evaluation requires at least one required fact")

    covered_count = 0
    for reference in references:
        result = completeness_judge(
            inputs={
                "question": inputs.get("question", ""),
                "required_fact": reference,
            },
            outputs={"answer": outputs.get("answer", "")},
        )
        covered = result.get("score")
        if covered not in (True, False, 0, 1):
            raise ValueError(f"Completeness judge returned an invalid score: {covered!r}")
        covered_count += 1 if covered else 0

    total_count = len(references)
    return {
        "key": "completeness",
        "score": covered_count / total_count,
        "comment": f"{covered_count}/{total_count} required facts were covered.",
    }


def run_completeness() -> None:
    """Run completeness against the frozen generation dataset in LangSmith."""
    client = Client()
    client.evaluate(
        evaluation_context.generate_answer_for_evaluation,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[evaluate_completeness],
        metadata=EXPERIMENT_METADATA,
        experiment_prefix=EXPERIMENT_PREFIX,
        description=(
            "Check each frozen required fact independently against the generated answer. "
            "The score is the number of covered required facts divided by the total "
            "number of required facts."
        ),
        max_concurrency=4
    )


if __name__ == "__main__":
    run_completeness()
