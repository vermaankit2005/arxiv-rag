"""Run live retrieval and generation, then check fact-level citation support.

A citation belongs to a factual claim or closely related fact group, not to every
sentence. Multiple citations in one group are judged together. The final score is
fully supported citation groups divided by all citation groups, averaged across
the questions by LangSmith.
"""

from langsmith import Client

from arxiv_rag.model_provider import get_generator_model_name, get_judge_model_name
from evals.fact_citation import (
    EVALUATOR_VERSION,
    build_fact_citation_judge,
    evaluate_fact_citations,
)
from evals.pipeline import context as evaluation_context

DESCRIPTION = __doc__
LANGSMITH_DATASET_NAME = "pipeline_required_fact_coverage_dataset"
EXPERIMENT_PREFIX = "pipeline_fact_citation"
EXPERIMENT_METADATA = {
    "metric": "fact_citation",
    "evaluation_level": "pipeline",
    "dataset": LANGSMITH_DATASET_NAME,
    "generator_model": get_generator_model_name(),
    "judge_model": get_judge_model_name(),
    "judge_thinking": "disabled",
    "generator_thinking": "disabled",
    "evaluator_version": EVALUATOR_VERSION,
    "citation_unit": "citation_group",
    "citation_scope": "natural_answer_structure",
    "multi_citation_support": "joint_evidence",
}

fact_citation_judge = build_fact_citation_judge()


def _build_passage_details(outputs: dict) -> dict[str, dict]:
    passages = outputs.get("retrieved_passages", {})
    details = {
        citation_id: {"id": citation_id, "text": text}
        for citation_id, text in passages.items()
    }
    for block in outputs.get("retrieved_context", "").split("\n\n---\n\n"):
        if "\nSection: " not in block or "\nText: " not in block:
            continue
        marker, remainder = block.split("\nSection: ", 1)
        section, text = remainder.split("\nText: ", 1)
        citation_id = marker.strip().removeprefix("[").removesuffix("]")
        if citation_id in details:
            details[citation_id] = {
                "id": citation_id,
                "section_path": section.split(" > "),
                "text": text,
            }
    return details


def evaluate_fact_citation(inputs: dict, outputs: dict) -> dict:
    """Judge citation groups against the passages returned by live retrieval."""
    return evaluate_fact_citations(
        answer=outputs.get("answer", ""),
        passages_by_id=_build_passage_details(outputs),
        fact_judge=fact_citation_judge,
    )


def run_fact_citation() -> None:
    """Run fact-level citation support against retrieved runtime passages."""
    client = Client()
    client.evaluate(
        evaluation_context.generate_pipeline_answer_and_passages_for_evaluation,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[evaluate_fact_citation],
        metadata=EXPERIMENT_METADATA,
        experiment_prefix=EXPERIMENT_PREFIX,
        description=DESCRIPTION,
        max_concurrency=1,
    )


if __name__ == "__main__":
    run_fact_citation()
