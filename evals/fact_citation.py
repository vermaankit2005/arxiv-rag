"""Check each citation group against its naturally attached factual claim.

All passages cited in the group are judged together using the complete answer.
"""

import re
from collections.abc import Callable

from openevals.llm import create_llm_as_judge  # pyright: ignore[reportMissingImports]

from arxiv_rag.answering.generator import CITATION_ID_PATTERN, CITATION_MARKER_PATTERN
from evals.judges import build_judge_model

EVALUATOR_VERSION = "fact-citation-v3"
CITATION_GROUP_PATTERN = re.compile(rf"(?:{CITATION_MARKER_PATTERN.pattern}\s*)+")

FACT_CITATION_PROMPT = """
You are checking one citation group in a complete answer.

The citation being checked is wrapped in `<TARGET_CITATION>` tags. Identify only
the factual claim or closely related claim group naturally attached to that marked
citation. Read Markdown normally: a citation after a list item supports that item,
and a citation after closely related sentences may support that group.
Ignore headings, introductory labels, transitions, formatting, and clearly
introduced analogies unless they make a factual claim covered by the citation.

Consider all cited passages together. Mark the citation group supported only when
they support every factual claim naturally attached to it. Check precise wording
and quantities. Detailed evidence overrides a loose summary when they conflict.
Judge only citation support; never require this claim group to answer the complete
user question. Use no outside knowledge.

Complete answer and citation group to check:
{inputs}

Cited evidence passages:
{outputs}
"""

FACT_CITATION_SCHEMA = {
    "title": "FactCitationResult",
    "type": "object",
    "properties": {
        "claim": {"type": "string"},
        "supported": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["claim", "supported", "reason"],
    "additionalProperties": False,
}


def extract_citation_groups(answer: str) -> list[tuple[str, ...]]:
    """Return adjacent citation IDs as ordered groups."""
    return [
        tuple(dict.fromkeys(CITATION_ID_PATTERN.findall(match.group())))
        for match in CITATION_GROUP_PATTERN.finditer(answer)
    ]


def build_fact_citation_judge() -> Callable:
    """Build the structured judge shared by both evaluation levels."""
    return create_llm_as_judge(
        prompt=FACT_CITATION_PROMPT,
        feedback_key="fact_citation_judgement",
        judge=build_judge_model(),
        output_schema=FACT_CITATION_SCHEMA,
    )


def _format_evidence(passage: dict) -> dict:
    return {
        "paper": passage.get("arxiv_id") or passage.get("label", ""),
        "section": " > ".join(passage.get("section_path", [])),
        "location": passage.get("location", ""),
        "text": passage.get("text", ""),
    }


def evaluate_fact_citations(
    answer: str,
    passages_by_id: dict[str, dict],
    fact_judge: Callable,
) -> dict:
    """Return fully supported citation groups divided by all citation groups."""
    citation_matches = list(CITATION_GROUP_PATTERN.finditer(answer))
    citation_groups = [
        tuple(dict.fromkeys(CITATION_ID_PATTERN.findall(match.group())))
        for match in citation_matches
    ]
    if not citation_groups:
        return {
            "key": "fact_citation",
            "score": 0.0,
            "comment": "The answer contained no citation groups.",
            "metadata": {
                "evaluator_version": EVALUATOR_VERSION,
                "citation_group_count": 0,
            },
        }

    unknown_ids = sorted(
        {
            citation_id
            for citation_group in citation_groups
            for citation_id in citation_group
            if citation_id not in passages_by_id
        }
    )
    if unknown_ids:
        raise ValueError(f"Answer used unknown citation IDs: {', '.join(unknown_ids)}")

    group_results = []
    for group_number, (citation_ids, citation_match) in enumerate(
        zip(citation_groups, citation_matches, strict=True), start=1
    ):
        marked_answer = (
            answer[:citation_match.start()]
            + "<TARGET_CITATION>"
            + citation_match.group()
            + "</TARGET_CITATION>"
            + answer[citation_match.end():]
        )
        result = fact_judge(
            inputs={
                "answer_with_target": marked_answer,
                "citation_ids": list(citation_ids),
            },
            outputs={
                "evidence_passages": [
                    _format_evidence(passages_by_id[citation_id])
                    for citation_id in citation_ids
                ]
            },
        )
        if result.get("supported") not in (True, False):
            raise ValueError(
                f"Fact-citation judge returned an invalid result: {result!r}"
            )
        group_results.append(
            {
                "group_number": group_number,
                "citation_ids": list(citation_ids),
                **result,
            }
        )

    supported_count = sum(
        1 for group_result in group_results if group_result["supported"]
    )
    group_count = len(citation_groups)
    return {
        "key": "fact_citation",
        "score": supported_count / group_count,
        "comment": (
            f"{supported_count}/{group_count} citation groups were fully supported."
        ),
        "metadata": {
            "evaluator_version": EVALUATOR_VERSION,
            "citation_group_count": group_count,
            "supported_group_count": supported_count,
            "group_results": group_results,
        },
    }
