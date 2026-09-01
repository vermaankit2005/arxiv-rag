"""Evaluation inventories for priority and full regression runs."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationSpec:
    """Everything needed to execute one existing LangSmith evaluation."""

    name: str
    dataset_name: str
    target: Any
    evaluators: tuple[Any, ...]
    feedback_keys: tuple[str, ...]
    metadata: dict
    experiment_prefix: str
    description: str | None
    expected_examples: int
    max_concurrency: int = 1
    use_balanced_subset: bool = False


BALANCED_GENERATION_EXAMPLE_IDS = (
    "gen-001",
    "gen-004",
    "gen-005",
    "gen-007",
    "gen-010",
    "gen-012",
    "gen-014",
    "gen-015",
    "gen-018",
    "gen-019",
    "gen-021",
    "gen-024",
)


def priority_specs() -> tuple[EvaluationSpec, ...]:
    """Return the urgent metrics, using a fixed balanced 12-case subset where possible."""
    from evals.answering import context as answering_context
    from evals.answering import (
        evaluate_generation_citation_support as generation_citation,
    )
    from evals.answering import evaluate_generation_naturalness as naturalness
    from evals.application import evaluate_policy_response_accuracy as policy_response
    from evals.application import evaluate_sensitive_data_protection as sensitive_data
    from evals.application import safety

    return (
        EvaluationSpec(
            name="generation_naturalness",
            dataset_name=naturalness.LANGSMITH_DATASET_NAME,
            target=answering_context.generate_answer_for_evaluation,
            evaluators=(naturalness.evaluate_naturalness,),
            feedback_keys=("naturalness",),
            metadata=naturalness.EXPERIMENT_METADATA,
            experiment_prefix=naturalness.EXPERIMENT_PREFIX,
            description=naturalness.DESCRIPTION,
            expected_examples=12,
            max_concurrency=2,
            use_balanced_subset=True,
        ),
        EvaluationSpec(
            name="application_sensitive_data",
            dataset_name=sensitive_data.DATASET_NAME,
            target=safety.generate_safety_answer,
            evaluators=(sensitive_data.evaluate_sensitive_data_protection,),
            feedback_keys=(sensitive_data.METRIC_NAME,),
            metadata=sensitive_data.EXPERIMENT_METADATA,
            experiment_prefix=sensitive_data.EXPERIMENT_PREFIX,
            description=sensitive_data.DESCRIPTION,
            expected_examples=3,
        ),
        EvaluationSpec(
            name="application_policy_response",
            dataset_name=policy_response.DATASET_NAME,
            target=safety.generate_safety_answer,
            evaluators=(policy_response.evaluate_policy_response_accuracy,),
            feedback_keys=(policy_response.METRIC_NAME,),
            metadata=policy_response.EXPERIMENT_METADATA,
            experiment_prefix=policy_response.EXPERIMENT_PREFIX,
            description=policy_response.DESCRIPTION,
            expected_examples=6,
        ),
        EvaluationSpec(
            name="generation_citation_support",
            dataset_name=generation_citation.LANGSMITH_DATASET_NAME,
            target=answering_context.generate_answer_for_evaluation,
            evaluators=(generation_citation.evaluate_citation_support,),
            feedback_keys=("citation_support",),
            metadata=generation_citation.EXPERIMENT_METADATA,
            experiment_prefix=generation_citation.EXPERIMENT_PREFIX,
            description=generation_citation.DESCRIPTION,
            expected_examples=12,
            max_concurrency=3,
            use_balanced_subset=True,
        ),
    )


def full_specs() -> tuple[EvaluationSpec, ...]:
    """Return every active, non-deprecated component, pipeline, and application eval."""
    from evals.answering import context as answering_context
    from evals.answering import (
        evaluate_generation_citation_support as generation_citation,
    )
    from evals.answering import evaluate_generation_completeness as completeness
    from evals.answering import evaluate_generation_correctness as correctness
    from evals.answering import (
        evaluate_generation_evidence_behavior as generation_evidence,
    )
    from evals.answering import evaluate_generation_groundedness as groundedness
    from evals.answering import evaluate_generation_naturalness as naturalness
    from evals.application import evaluate_harmful_content_safety as harmful_content
    from evals.application import evaluate_policy_response_accuracy as policy_response
    from evals.application import (
        evaluate_prompt_injection_resistance as prompt_injection,
    )
    from evals.application import evaluate_sensitive_data_protection as sensitive_data
    from evals.application import safety
    from evals.loading import evaluate_html_content_retention as html_retention
    from evals.loading import (
        evaluate_passage_anchor_validity_and_probe_recall as passage_recall,
    )
    from evals.pipeline import context as pipeline_context
    from evals.pipeline import evaluate_e2e_citation_support as pipeline_citation
    from evals.pipeline import evaluate_e2e_evidence_behavior as pipeline_evidence
    from evals.pipeline import evaluate_e2e_required_fact_coverage as pipeline_coverage
    from evals.retriever import (
        evaluate_retriever_document_precision as document_precision,
    )
    from evals.retriever import evaluate_retriever_evidence_recall as evidence_recall
    from evals.retriever import evaluate_retriever_mean_reciprocal_rank as mrr

    generation_specs = (
        EvaluationSpec(
            "generation_groundedness",
            groundedness.LANGSMITH_DATASET_NAME,
            answering_context.generate_answer_for_evaluation,
            (groundedness.evaluate_groundedness,),
            ("groundedness",),
            groundedness.EXPERIMENT_METADATA,
            groundedness.EXPERIMENT_PREFIX,
            groundedness.DESCRIPTION,
            24,
            4,
        ),
        EvaluationSpec(
            "generation_citation_support",
            generation_citation.LANGSMITH_DATASET_NAME,
            answering_context.generate_answer_for_evaluation,
            (generation_citation.evaluate_citation_support,),
            ("citation_support",),
            generation_citation.EXPERIMENT_METADATA,
            generation_citation.EXPERIMENT_PREFIX,
            generation_citation.DESCRIPTION,
            24,
            3,
        ),
        EvaluationSpec(
            "generation_correctness",
            correctness.LANGSMITH_DATASET_NAME,
            answering_context.generate_answer_for_evaluation,
            (correctness.evaluate_correctness,),
            ("correctness",),
            correctness.EXPERIMENT_METADATA,
            correctness.EXPERIMENT_PREFIX,
            correctness.DESCRIPTION,
            24,
            4,
        ),
        EvaluationSpec(
            "generation_completeness",
            completeness.LANGSMITH_DATASET_NAME,
            answering_context.generate_answer_for_evaluation,
            (completeness.evaluate_completeness,),
            ("completeness",),
            completeness.EXPERIMENT_METADATA,
            completeness.EXPERIMENT_PREFIX,
            completeness.DESCRIPTION,
            24,
            4,
        ),
        EvaluationSpec(
            "generation_naturalness",
            naturalness.LANGSMITH_DATASET_NAME,
            answering_context.generate_answer_for_evaluation,
            (naturalness.evaluate_naturalness,),
            ("naturalness",),
            naturalness.EXPERIMENT_METADATA,
            naturalness.EXPERIMENT_PREFIX,
            naturalness.DESCRIPTION,
            24,
            2,
        ),
        EvaluationSpec(
            "generation_evidence_behavior",
            generation_evidence.LANGSMITH_DATASET_NAME,
            answering_context.generate_answer_for_evaluation,
            (generation_evidence.evaluate_evidence_behavior,),
            ("evidence_behavior",),
            generation_evidence.EXPERIMENT_METADATA,
            generation_evidence.EXPERIMENT_PREFIX,
            generation_evidence.DESCRIPTION,
            9,
            2,
        ),
    )
    application_specs = (
        EvaluationSpec(
            "application_harmful_content",
            harmful_content.DATASET_NAME,
            safety.generate_safety_answer,
            (harmful_content.evaluate_harmful_content_safety,),
            (harmful_content.METRIC_NAME,),
            harmful_content.EXPERIMENT_METADATA,
            harmful_content.EXPERIMENT_PREFIX,
            harmful_content.DESCRIPTION,
            3,
        ),
        EvaluationSpec(
            "application_sensitive_data",
            sensitive_data.DATASET_NAME,
            safety.generate_safety_answer,
            (sensitive_data.evaluate_sensitive_data_protection,),
            (sensitive_data.METRIC_NAME,),
            sensitive_data.EXPERIMENT_METADATA,
            sensitive_data.EXPERIMENT_PREFIX,
            sensitive_data.DESCRIPTION,
            3,
        ),
        EvaluationSpec(
            "application_prompt_injection",
            prompt_injection.DATASET_NAME,
            safety.generate_safety_answer,
            (prompt_injection.evaluate_prompt_injection_resistance,),
            (prompt_injection.METRIC_NAME,),
            prompt_injection.EXPERIMENT_METADATA,
            prompt_injection.EXPERIMENT_PREFIX,
            prompt_injection.DESCRIPTION,
            4,
        ),
        EvaluationSpec(
            "application_policy_response",
            policy_response.DATASET_NAME,
            safety.generate_safety_answer,
            (policy_response.evaluate_policy_response_accuracy,),
            (policy_response.METRIC_NAME,),
            policy_response.EXPERIMENT_METADATA,
            policy_response.EXPERIMENT_PREFIX,
            policy_response.DESCRIPTION,
            6,
        ),
    )
    return (
        EvaluationSpec(
            "loading_anchor_and_recall",
            passage_recall.LANGSMITH_DATASET_NAME,
            passage_recall.load_passages_for_evaluation,
            (passage_recall.evaluate_passage_anchor_validity_and_probe_recall,),
            ("anchor_coverage", "text_recall"),
            passage_recall.EXPERIMENT_METADATA,
            passage_recall.EXPERIMENT_PREFIX,
            passage_recall.DESCRIPTION,
            12,
        ),
        EvaluationSpec(
            "loading_content_retention",
            html_retention.LANGSMITH_DATASET_NAME,
            html_retention.load_paper_for_content_retention,
            (html_retention.evaluate_html_content_retention,),
            ("html_block_coverage", "html_word_retention"),
            html_retention.EXPERIMENT_METADATA,
            html_retention.EXPERIMENT_PREFIX,
            html_retention.DESCRIPTION,
            12,
        ),
        EvaluationSpec(
            "retriever_evidence_recall",
            evidence_recall.LANGSMITH_DATASET_NAME,
            evidence_recall.fetch_docs_for_evaluation,
            (evidence_recall.evaluate_evidence_recall,),
            ("evidence_recall",),
            evidence_recall.EXPERIMENT_METADATA,
            evidence_recall.EXPERIMENT_PREFIX,
            evidence_recall.DESCRIPTION,
            24,
        ),
        EvaluationSpec(
            "retriever_mrr",
            mrr.LANGSMITH_DATASET_NAME,
            mrr.fetch_docs_for_evaluation,
            (mrr.evaluate_mrr,),
            ("mrr_at_5",),
            mrr.EXPERIMENT_METADATA,
            mrr.EXPERIMENT_PREFIX,
            mrr.DESCRIPTION,
            24,
        ),
        EvaluationSpec(
            "retriever_document_precision",
            document_precision.LANGSMITH_DATASET_NAME,
            document_precision.fetch_docs_for_evaluation,
            (document_precision.evaluate_document_precision,),
            ("document_precision_at_5",),
            document_precision.EXPERIMENT_METADATA,
            document_precision.EXPERIMENT_PREFIX,
            document_precision.DESCRIPTION,
            24,
        ),
        *generation_specs,
        EvaluationSpec(
            "pipeline_required_fact_coverage",
            pipeline_coverage.LANGSMITH_DATASET_NAME,
            pipeline_context.generate_pipeline_answer_for_evaluation,
            (pipeline_coverage.evaluate_required_fact_coverage,),
            ("required_fact_coverage",),
            pipeline_coverage.EXPERIMENT_METADATA,
            pipeline_coverage.EXPERIMENT_PREFIX,
            pipeline_coverage.DESCRIPTION,
            24,
        ),
        EvaluationSpec(
            "pipeline_citation_support",
            pipeline_citation.LANGSMITH_DATASET_NAME,
            pipeline_context.generate_pipeline_answer_and_passages_for_evaluation,
            (pipeline_citation.evaluate_citation_support,),
            ("citation_support",),
            pipeline_citation.EXPERIMENT_METADATA,
            pipeline_citation.EXPERIMENT_PREFIX,
            pipeline_citation.DESCRIPTION,
            24,
        ),
        EvaluationSpec(
            "pipeline_evidence_behavior",
            pipeline_evidence.LANGSMITH_DATASET_NAME,
            pipeline_context.generate_pipeline_answer_and_passages_for_evaluation,
            (pipeline_evidence.evaluate_evidence_behavior,),
            ("evidence_behavior",),
            pipeline_evidence.EXPERIMENT_METADATA,
            pipeline_evidence.EXPERIMENT_PREFIX,
            pipeline_evidence.DESCRIPTION,
            9,
        ),
        *application_specs,
    )
