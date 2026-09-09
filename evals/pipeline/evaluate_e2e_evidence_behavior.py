"""This runs live retrieval and generation, then checks whether the answer behaves correctly for the evidence that was actually retrieved.

It should answer when the evidence is enough, limit itself when only part is
supported, and refuse when nothing is supported. Each example gets 0, 0.5, or 1.
The final score is the average.
"""

from langsmith import Client
from openevals.llm import create_llm_as_judge  # pyright: ignore[reportMissingImports]

from arxiv_rag.answering.generator import INSUFFICIENT_EVIDENCE_ANSWER
from arxiv_rag.model_provider import get_generator_model_name, get_judge_model_name
from evals.judges import build_judge_model
from evals.pipeline import context as evaluation_context

DESCRIPTION = __doc__
LANGSMITH_DATASET_NAME = "pipeline_evidence_behavior_dataset"
EXPERIMENT_PREFIX = "pipeline_evidence_behavior"
EVALUATOR_VERSION = "pipeline-evidence-behavior-v2"
EXPERIMENT_METADATA = {
    "metric": "pipeline_evidence_behavior",
    "dataset": LANGSMITH_DATASET_NAME,
    "generator_model": get_generator_model_name(),
    "judge_model": get_judge_model_name(),
    "judge_thinking": "disabled",
    "generator_thinking": "disabled",
    "evaluator_version": EVALUATOR_VERSION,
}

# The judge names the behavior; this file turns that name into the score. Asking
# for a word rather than one of 0, 0.5, or 1 is what a constrained decoder can
# reliably produce.
BEHAVIOR_SCORES = {"correct": 1.0, "mixed": 0.5, "wrong": 0.0}

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

Report "correct" when the behavior is fully correct, "mixed" when it is partly
right or the limitation is unclear, and "wrong" when it chooses the wrong
behavior or invents unsupported facts.
Use only the retrieved passages. Do not use outside knowledge.

Question and retrieved passages:
{{inputs}}

Final answer:
{{outputs}}
"""

EVIDENCE_BEHAVIOR_SCHEMA = {
    "title": "EvidenceBehaviorResult",
    "type": "object",
    "properties": {
        "support_level": {"type": "string", "enum": ["all", "some", "none"]},
        "behavior": {"type": "string", "enum": list(BEHAVIOR_SCORES)},
        "reason": {"type": "string"},
    },
    "required": ["support_level", "behavior", "reason"],
    "additionalProperties": False,
}

judge_model = build_judge_model()
evidence_behavior_judge = create_llm_as_judge(
    prompt=EVIDENCE_BEHAVIOR_PROMPT,
    feedback_key="pipeline_evidence_behavior",
    judge=judge_model,
    output_schema=EVIDENCE_BEHAVIOR_SCHEMA,
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
    behavior = result.get("behavior")
    if behavior not in BEHAVIOR_SCORES:
        raise ValueError(
            f"Evidence-behavior judge returned no usable behavior. Got keys "
            f"{sorted(result)} with behavior {behavior!r}."
        )

    support_level = result.get("support_level", "unknown")
    reason = result.get("reason", "Pipeline evidence behavior was judged.")
    return {
        "key": "evidence_behavior",
        "score": BEHAVIOR_SCORES[behavior],
        "comment": f"Evidence supports {support_level}; behavior {behavior}. {reason}",
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
