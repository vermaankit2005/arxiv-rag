"""Run live retrieval and generation, then judge the final answer as a complete user-facing response.

The judge sees the question, final answer, and hidden source-curated required facts,
but not the runtime retrieved context. It scores correctness, relevance,
completeness, clarity, and conciseness together from 0 to 1.
"""

from typing import cast

from langsmith import Client
from openevals.llm import create_llm_as_judge  # pyright: ignore[reportMissingImports]

from arxiv_rag.model_provider import get_generator_model_name, get_judge_model_name
from evals.judges import build_judge_model
from evals.pipeline import context as evaluation_context

DESCRIPTION = __doc__
LANGSMITH_DATASET_NAME = "pipeline_required_fact_coverage_dataset"
EXPERIMENT_PREFIX = "pipeline_answer_quality"
ANSWER_QUALITY_RUBRIC_VERSION = "holistic-v1"
QUALITY_SCORES = {"excellent": 1.0, "good": 0.75, "mixed": 0.5, "poor": 0.25, "unusable": 0.0}
EXPERIMENT_METADATA = {
    "metric": "answer_quality",
    "evaluation_level": "pipeline",
    "evaluation_focus": "holistic user-facing answer quality",
    "rubric_version": ANSWER_QUALITY_RUBRIC_VERSION,
    "dataset": LANGSMITH_DATASET_NAME,
    "generator_model": get_generator_model_name(),
    "judge_model": get_judge_model_name(),
    "judge_thinking": "disabled",
    "generator_thinking": "disabled",
    "runtime_context_visible_to_judge": False,
}

ANSWER_QUALITY_PROMPT = """
You are evaluating the final answer from a retrieval-augmented research
assistant as an independent expert reader.

Judge how well the answer responds to the question overall:
- Correctness: its substantive claims are accurate and do not contradict the
  authoritative source-curated required facts.
- Relevance: it directly answers the question without unrelated material.
- Completeness: it covers the important information needed for a useful answer.
- Clarity: it is understandable and explains the ideas coherently.
- Conciseness: it avoids needless repetition or detail while remaining useful.

Treat the supplied required facts as authoritative for this paper-specific
question. You may use general domain knowledge to recognize clear factual errors
or assess explanatory quality, but do not override or penalize a paper-specific
fact merely because it is absent from your outside knowledge. Ignore citation
marker formatting. Do not assess whether retrieved passages support citations;
that is measured separately.

Question and hidden source-curated required facts:
{inputs}

Final answer:
{outputs}

Return one overall rating:
- excellent: correct, direct, complete, clear, and appropriately concise.
- good: useful overall, with only a minor weakness in one dimension.
- mixed: partly useful, but has a meaningful correctness, relevance,
  completeness, clarity, or conciseness problem.
- poor: only a small useful portion and major overall problems.
- unusable: non-responsive or fundamentally incorrect.
"""

ANSWER_QUALITY_SCHEMA = {
    "title": "AnswerQualityResult",
    "type": "object",
    "properties": {
        "rating": {"type": "string", "enum": list(QUALITY_SCORES)},
        "reason": {"type": "string"},
    },
    "required": ["rating", "reason"],
    "additionalProperties": False,
}

judge_model = build_judge_model()
answer_quality_judge = create_llm_as_judge(
    prompt=ANSWER_QUALITY_PROMPT,
    feedback_key="answer_quality",
    judge=judge_model,
    output_schema=ANSWER_QUALITY_SCHEMA,
)


def evaluate_answer_quality(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Judge the final answer holistically without exposing runtime context."""
    required_facts = reference_outputs.get("required_facts", [])
    if not required_facts:
        raise ValueError("Pipeline answer-quality evaluation requires at least one required fact")

    result = cast(dict, answer_quality_judge(
        inputs={
            "question": inputs.get("question", ""),
            "required_facts": required_facts,
        },
        outputs={"answer": outputs.get("answer", "")},
    ))
    rating = result.get("rating")
    if rating not in QUALITY_SCORES:
        raise ValueError(f"Answer-quality judge returned an invalid rating: {rating!r}")

    return {
        "key": "answer_quality",
        "score": QUALITY_SCORES[rating],
        "comment": f"Overall rating: {rating}. {result.get('reason', 'Overall answer quality was judged.')}",
    }


def run_answer_quality() -> None:
    """Run holistic answer quality against live pipeline answers."""
    client = Client()
    client.evaluate(
        evaluation_context.generate_pipeline_answer_and_passages_for_evaluation,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[evaluate_answer_quality],
        metadata=EXPERIMENT_METADATA,
        experiment_prefix=EXPERIMENT_PREFIX,
        description=DESCRIPTION,
        max_concurrency=1,
    )


if __name__ == "__main__":
    run_answer_quality()
