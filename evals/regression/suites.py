"""Small, explicit inventories for the priority and full regression suites."""

GENERATION_SUBSET = (
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


def _spec(
    name,
    module,
    target,
    evaluator,
    feedback_keys,
    expected,
    concurrency=1,
    subset_ids=(),
):
    """Create the plain dictionary consumed by the runner."""
    return {
        "name": name,
        "dataset": getattr(module, "LANGSMITH_DATASET_NAME", None)
        or module.DATASET_NAME,
        "target": target,
        "evaluator": evaluator,
        "feedback_keys": feedback_keys,
        "metadata": module.EXPERIMENT_METADATA,
        "prefix": module.EXPERIMENT_PREFIX,
        "description": module.DESCRIPTION,
        "expected": expected,
        "concurrency": concurrency,
        "subset_ids": subset_ids,
    }


def priority_specs() -> list[dict]:
    """Return the four urgent evaluations."""
    from evals.answering import context as answering_context
    from evals.answering import evaluate_generation_citation_support as citation
    from evals.answering import evaluate_generation_naturalness as naturalness
    from evals.application import evaluate_policy_response_accuracy as policy
    from evals.application import evaluate_sensitive_data_protection as sensitive
    from evals.application import safety

    generate = answering_context.generate_answer_for_evaluation
    return [
        _spec(
            "generation_naturalness",
            naturalness,
            generate,
            naturalness.evaluate_naturalness,
            ("naturalness",),
            12,
            2,
            GENERATION_SUBSET,
        ),
        _spec(
            "application_sensitive_data",
            sensitive,
            safety.generate_safety_answer,
            sensitive.evaluate_sensitive_data_protection,
            (sensitive.METRIC_NAME,),
            10,
        ),
        _spec(
            "application_policy_response",
            policy,
            safety.generate_safety_answer,
            policy.evaluate_policy_response_accuracy,
            (policy.METRIC_NAME,),
            10,
        ),
        _spec(
            "generation_citation_support",
            citation,
            generate,
            citation.evaluate_citation_support,
            ("citation_support",),
            12,
            3,
            GENERATION_SUBSET,
        ),
    ]


def full_specs() -> list[dict]:
    """Return every active evaluation against its full frozen dataset."""
    from evals.answering import context as answering_context
    from evals.answering import evaluate_generation_citation_support as citation
    from evals.answering import evaluate_generation_completeness as completeness
    from evals.answering import evaluate_generation_correctness as correctness
    from evals.answering import (
        evaluate_generation_evidence_behavior as generation_evidence,
    )
    from evals.answering import evaluate_generation_groundedness as groundedness
    from evals.answering import evaluate_generation_naturalness as naturalness
    from evals.application import evaluate_harmful_content_safety as harmful
    from evals.application import evaluate_policy_response_accuracy as policy
    from evals.application import evaluate_prompt_injection_resistance as injection
    from evals.application import evaluate_sensitive_data_protection as sensitive
    from evals.application import safety
    from evals.loading import evaluate_html_content_retention as retention
    from evals.loading import (
        evaluate_passage_anchor_validity_and_probe_recall as passage_recall,
    )
    from evals.pipeline import context as pipeline_context
    from evals.pipeline import evaluate_e2e_citation_support as pipeline_citation
    from evals.pipeline import evaluate_e2e_evidence_behavior as pipeline_evidence
    from evals.pipeline import evaluate_e2e_required_fact_coverage as pipeline_coverage
    from evals.retriever import evaluate_retriever_document_precision as precision
    from evals.retriever import evaluate_retriever_evidence_recall as recall
    from evals.retriever import evaluate_retriever_mean_reciprocal_rank as mrr

    generate = answering_context.generate_answer_for_evaluation
    pipeline_generate = pipeline_context.generate_pipeline_answer_for_evaluation
    pipeline_generate_with_passages = (
        pipeline_context.generate_pipeline_answer_and_passages_for_evaluation
    )

    return [
        _spec(
            "loading_anchor_and_recall",
            passage_recall,
            passage_recall.load_passages_for_evaluation,
            passage_recall.evaluate_passage_anchor_validity_and_probe_recall,
            ("anchor_coverage", "text_recall"),
            12,
        ),
        _spec(
            "loading_content_retention",
            retention,
            retention.load_paper_for_content_retention,
            retention.evaluate_html_content_retention,
            ("html_block_coverage", "html_word_retention"),
            12,
        ),
        _spec(
            "retriever_evidence_recall",
            recall,
            recall.fetch_docs_for_evaluation,
            recall.evaluate_evidence_recall,
            ("evidence_recall",),
            24,
        ),
        _spec(
            "retriever_mrr",
            mrr,
            mrr.fetch_docs_for_evaluation,
            mrr.evaluate_mrr,
            ("mrr_at_5",),
            24,
        ),
        _spec(
            "retriever_document_precision",
            precision,
            precision.fetch_docs_for_evaluation,
            precision.evaluate_document_precision,
            ("document_precision_at_5",),
            24,
        ),
        _spec(
            "generation_groundedness",
            groundedness,
            generate,
            groundedness.evaluate_groundedness,
            ("groundedness",),
            24,
            4,
        ),
        _spec(
            "generation_citation_support",
            citation,
            generate,
            citation.evaluate_citation_support,
            ("citation_support",),
            24,
            3,
        ),
        _spec(
            "generation_correctness",
            correctness,
            generate,
            correctness.evaluate_correctness,
            ("correctness",),
            24,
            4,
        ),
        _spec(
            "generation_completeness",
            completeness,
            generate,
            completeness.evaluate_completeness,
            ("completeness",),
            24,
            4,
        ),
        _spec(
            "generation_naturalness",
            naturalness,
            generate,
            naturalness.evaluate_naturalness,
            ("naturalness",),
            24,
            2,
        ),
        _spec(
            "generation_evidence_behavior",
            generation_evidence,
            generate,
            generation_evidence.evaluate_evidence_behavior,
            ("evidence_behavior",),
            9,
            2,
        ),
        _spec(
            "pipeline_required_fact_coverage",
            pipeline_coverage,
            pipeline_generate,
            pipeline_coverage.evaluate_required_fact_coverage,
            ("required_fact_coverage",),
            24,
        ),
        _spec(
            "pipeline_citation_support",
            pipeline_citation,
            pipeline_generate_with_passages,
            pipeline_citation.evaluate_citation_support,
            ("citation_support",),
            24,
        ),
        _spec(
            "pipeline_evidence_behavior",
            pipeline_evidence,
            pipeline_generate_with_passages,
            pipeline_evidence.evaluate_evidence_behavior,
            ("evidence_behavior",),
            9,
        ),
        _spec(
            "application_harmful_content",
            harmful,
            safety.generate_safety_answer,
            harmful.evaluate_harmful_content_safety,
            (harmful.METRIC_NAME,),
            10,
        ),
        _spec(
            "application_sensitive_data",
            sensitive,
            safety.generate_safety_answer,
            sensitive.evaluate_sensitive_data_protection,
            (sensitive.METRIC_NAME,),
            10,
        ),
        _spec(
            "application_prompt_injection",
            injection,
            safety.generate_safety_answer,
            injection.evaluate_prompt_injection_resistance,
            (injection.METRIC_NAME,),
            10,
        ),
        _spec(
            "application_policy_response",
            policy,
            safety.generate_safety_answer,
            policy.evaluate_policy_response_accuracy,
            (policy.METRIC_NAME,),
            10,
        ),
    ]
