from langsmith import Client
from openevals.llm import create_llm_as_judge

from evals.answering.judges import build_judge_model
from evals.pipeline import context as evaluation_context

LANGSMITH_DATASET_NAME = "pipeline_required_fact_coverage_dataset"
EXPERIMENT_PREFIX = "pipeline_required_fact_coverage"
JUDGE_MODEL_NAME = "gemma4:26b"
EXPERIMENT_METADATA = {
    "metric": "required_fact_coverage",
    "evaluation_level": "pipeline",
    "dataset": LANGSMITH_DATASET_NAME,
    "retriever_top_k": 5,
    "generator_model": "gemma4:26b",
    "judge_model": JUDGE_MODEL_NAME,
}

REQUIRED_FACT_COVERAGE_PROMPT = """
You are checking whether the final answer from a retrieval-augmented application
covers one required fact.

The fact is covered only when the answer communicates its complete and correct
factual meaning, either exactly or by a clear paraphrase. A citation marker alone
does not cover a fact. Do not use outside knowledge.

Question and required fact:
{inputs}

Final answer:
{outputs}
"""

judge_model = build_judge_model(JUDGE_MODEL_NAME)
required_fact_coverage_judge = create_llm_as_judge(
    prompt=REQUIRED_FACT_COVERAGE_PROMPT,
    feedback_key="required_fact_coverage",
    judge=judge_model,
    choices=[False, True],
)

def evaluate_required_fact_coverage(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Return covered hidden required facts divided by all required facts."""
    required_facts = reference_outputs.get("required_facts", [])
    if not required_facts:
        raise ValueError("Pipeline coverage evaluation requires at least one required fact")

    covered_count = 0

    for required_fact in required_facts:

        result = required_fact_coverage_judge(
            inputs={
                "question": inputs.get("question", ""),
                "required_fact": required_fact,
            },
            outputs={"answer": outputs.get("answer", "")},
        )

        covered = result.get("score")
        if covered not in (True, False, 0, 1):
            raise ValueError(
                f"Required-fact coverage judge returned an invalid score: {covered!r}"
            )

        covered_count += 1 if covered else 0

    total_count = len(required_facts)
    return {
        "key": "required_fact_coverage",
        "score": covered_count / total_count,
        "comment": f"{covered_count}/{total_count} required facts were covered.",
    }


def run_required_fact_coverage() -> None:
    """Run live retrieval and generation against the separate pipeline dataset."""
    client = Client()

    client.evaluate(
        evaluation_context.generate_pipeline_answer_for_evaluation,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[evaluate_required_fact_coverage],
        metadata=EXPERIMENT_METADATA,
        experiment_prefix=EXPERIMENT_PREFIX,
        description=(
            "Run each question through live production retrieval and answer generation, "
            "then check each hidden required fact independently. The score is covered "
            "required facts divided by all required facts."
        ),
    )


if __name__ == "__main__":
    run_required_fact_coverage()
