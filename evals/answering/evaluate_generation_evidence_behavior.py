"""This checks whether the answer behaves correctly for the evidence it was given.

Fully supported questions should be answered. Partial questions should answer
only the supported part and say what is missing. Unsupported questions should
use the exact refusal. Each example gets 0, 0.5, or 1. The final score is the
average.
"""

from langsmith import Client
from openevals.llm import create_llm_as_judge  # pyright: ignore[reportMissingImports]

from arxiv_rag.answering.generator import INSUFFICIENT_EVIDENCE_ANSWER
from evals.answering import context as evaluation_context
from evals.judges import DEFAULT_JUDGE_MODEL, build_judge_model

DESCRIPTION = __doc__
LANGSMITH_DATASET_NAME = "generation_evidence_behavior_dataset"
EXPERIMENT_PREFIX = "generation_evidence_behavior"
ALLOWED_SCORES = {0, 0.5, 1}
EXPERIMENT_METADATA = {
    "metric": "evidence_behavior",
    "dataset": LANGSMITH_DATASET_NAME,
    "generator_model": "gemma4:26b",
    "judge_model": DEFAULT_JUDGE_MODEL,
    "judge_thinking": "disabled",
    "generator_thinking": "disabled",
}

EVIDENCE_BEHAVIOR_PROMPT = """
You are checking whether an answer behaves correctly for the available evidence.
Use only the supplied reference. Do not use outside knowledge.

For fully_supported_answer:
- Score 1 when the answer gives all supported requested facts without refusing or
  claiming that supplied information is missing.
- Score 0.5 when it answers but misses an important supported fact or adds needless
  uncertainty.
- Score 0 when it refuses or does not answer the supported question.

For partial_answer_with_limitation:
- Score 1 when it answers the supported part, clearly says what is not specified,
  and does not guess the unsupported part.
- Score 0.5 when it gives only the supported answer or only the limitation, or the
  limitation is too vague.
- Score 0 when it guesses the unsupported part or refuses the whole question.

Question and expected behavior:
{inputs}

Generated answer:
{outputs}
"""

judge_model = build_judge_model()
evidence_behavior_judge = create_llm_as_judge(
    prompt=EVIDENCE_BEHAVIOR_PROMPT,
    feedback_key="evidence_behavior",
    judge=judge_model,
    choices=[0, 0.5, 1],
)


def evaluate_evidence_behavior(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Score correct answering, partial answering, or exact refusal behavior."""
    answer = outputs.get("answer", "").strip()
    expected_behavior = reference_outputs.get("expected_behavior", "")

    if expected_behavior == "insufficient_evidence_refusal":
        score = 1 if answer == INSUFFICIENT_EVIDENCE_ANSWER else 0
        comment = "The answer matched the exact insufficient-evidence response." if score else "The answer did not use the exact insufficient-evidence response."
        return {"key": "evidence_behavior", "score": score, "comment": comment}

    result = evidence_behavior_judge(
        inputs={
            "question": inputs.get("question", ""),
            "expected_behavior": expected_behavior,
            "supported_facts": reference_outputs.get("supported_facts", []),
            "unsupported_parts": reference_outputs.get("unsupported_parts", []),
        },
        outputs={"answer": answer},
    )
    score = result.get("score")
    if score not in ALLOWED_SCORES:
        raise ValueError(f"Evidence-behavior judge returned an invalid score: {score!r}")

    return {
        "key": "evidence_behavior",
        "score": score,
        "comment": result.get("comment", "Evidence behavior was judged."),
    }


def run_evidence_behavior() -> None:
    """Run evidence-behavior evaluation against the separate curated dataset."""
    client = Client()
    client.evaluate(
        evaluation_context.generate_answer_for_evaluation,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[evaluate_evidence_behavior],
        metadata=EXPERIMENT_METADATA,
        experiment_prefix=EXPERIMENT_PREFIX,
        description=DESCRIPTION,
        max_concurrency=2,
    )


if __name__ == "__main__":
    run_evidence_behavior()
