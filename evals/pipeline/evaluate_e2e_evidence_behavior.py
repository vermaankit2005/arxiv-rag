"""This runs live retrieval and generation, then checks whether the answer behaves correctly for the evidence that was actually retrieved.

It should answer when the evidence is enough, limit itself when only part is
supported, and refuse when nothing is supported. Each example gets 0, 0.5, or 1.
The final score is the average.
"""

from langsmith import Client
from openevals.llm import create_llm_as_judge  # pyright: ignore[reportMissingImports]

from arxiv_rag.answering.generator import INSUFFICIENT_EVIDENCE_ANSWER
from evals.judges import DEFAULT_JUDGE_MODEL, build_judge_model
from evals.pipeline import context as evaluation_context

DESCRIPTION = __doc__
LANGSMITH_DATASET_NAME = "pipeline_evidence_behavior_dataset"
EXPERIMENT_PREFIX = "pipeline_evidence_behavior"
EXPERIMENT_METADATA = {
    "metric": "pipeline_evidence_behavior",
    "dataset": LANGSMITH_DATASET_NAME,
    "generator_model": "gemma4:26b",
    "judge_model": DEFAULT_JUDGE_MODEL,
    "judge_thinking": "disabled",
    "generator_thinking": "disabled",
}

EVIDENCE_BEHAVIOR_PROMPT = f"""
You are checking whether a retrieval-augmented application responds appropriately
to the evidence it actually retrieved.

First determine whether the retrieved passages support all, some, or none of the
information requested by the question. Then judge the final answer:

- When all requested information is supported, the answer should answer it.
- When only some is supported, the answer should give only the supported part and
  clearly state what the evidence does not specify.
- When none is supported, the answer should be exactly:
  {INSUFFICIENT_EVIDENCE_ANSWER}
- The answer must not invent details unsupported by the retrieved passages.

Score 1 when the behavior is fully correct, 0.5 when it is mixed or the limitation
is unclear, and 0 when it chooses the wrong behavior or invents unsupported facts.
Use only the retrieved passages. Do not use outside knowledge.

Question and retrieved passages:
{{inputs}}

Final answer:
{{outputs}}
"""

judge_model = build_judge_model()
evidence_behavior_judge = create_llm_as_judge(
    prompt=EVIDENCE_BEHAVIOR_PROMPT,
    feedback_key="pipeline_evidence_behavior",
    judge=judge_model,
    choices=[0, 0.5, 1],
)


def evaluate_evidence_behavior(inputs: dict, outputs: dict) -> dict:
    """Judge answer, partial-answer, or refusal behavior against live evidence."""
    result = evidence_behavior_judge(
        inputs={
            "question": inputs.get("question", ""),
            "retrieved_passages": outputs.get("retrieved_context", ""),
        },
        outputs={"answer": outputs.get("answer", "")},
    )
    score = result.get("score")
    if score not in (0, 0.5, 1):
        raise ValueError(f"Evidence-behavior judge returned an invalid score: {score!r}")

    return {
        "key": "evidence_behavior",
        "score": score,
        "comment": result.get("comment", "Pipeline evidence behavior was judged."),
    }


def run_evidence_behavior() -> None:
    """Run evidence behavior against passages returned by live retrieval."""
    client = Client()
    client.evaluate(
        evaluation_context.generate_pipeline_answer_and_passages_for_evaluation,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[evaluate_evidence_behavior],
        metadata=EXPERIMENT_METADATA,
        experiment_prefix=EXPERIMENT_PREFIX,
        description=DESCRIPTION,
        max_concurrency=1,
    )


if __name__ == "__main__":
    run_evidence_behavior()
