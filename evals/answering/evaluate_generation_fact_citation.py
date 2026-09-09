"""Check whether each cited factual claim is supported by its frozen evidence.

A citation belongs to a factual claim or closely related fact group, not to every
sentence. Multiple citations in one group are judged together. The final score is
fully supported citation groups divided by all citation groups, averaged across
the questions by LangSmith.
"""

from langsmith import Client

from arxiv_rag.model_provider import get_generator_model_name, get_judge_model_name
from evals.answering import context as evaluation_context
from evals.fact_citation import (
    EVALUATOR_VERSION,
    build_fact_citation_judge,
    evaluate_fact_citations,
)

DESCRIPTION = __doc__
LANGSMITH_DATASET_NAME = "generation_quality_dataset"
EXPERIMENT_PREFIX = "generation_fact_citation"
EXPERIMENT_METADATA = {
    "metric": "fact_citation",
    "evaluation_level": "answer_generation",
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


def evaluate_fact_citation(inputs: dict, outputs: dict) -> dict:
    """Judge citation groups against the frozen passages supplied to generation."""
    passages_by_id = {
        passage["id"]: passage
        for passage in inputs.get("context_passages", [])
    }
    return evaluate_fact_citations(
        answer=outputs.get("answer", ""),
        passages_by_id=passages_by_id,
        fact_judge=fact_citation_judge,
    )


def run_fact_citation() -> None:
    """Run fact-level citation support against the frozen generation dataset."""
    client = Client()
    client.evaluate(
        evaluation_context.generate_answer_for_evaluation,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[evaluate_fact_citation],
        metadata=EXPERIMENT_METADATA,
        experiment_prefix=EXPERIMENT_PREFIX,
        description=DESCRIPTION,
        max_concurrency=1,
    )


if __name__ == "__main__":
    run_fact_citation()
