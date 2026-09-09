"""Check whether each citation group supports its attached factual claim."""

import re
from collections.abc import Callable

from openevals.llm import create_llm_as_judge  # pyright: ignore[reportMissingImports]

from arxiv_rag.answering.generator import CITATION_ID_PATTERN, CITATION_MARKER_PATTERN
from evals.judges import build_judge_model

EVALUATOR_VERSION = "fact-citation-v3"
CITATION_GROUP_PATTERN = re.compile(rf"(?:{CITATION_MARKER_PATTERN.pattern}\s*)+")

FACT_CITATION_PROMPT = """
Check whether the cited passages support the factual claim attached to the citation
wrapped in <TARGET_CITATION> tags. Consider the cited passages together and use
only their evidence. Return supported=true only if they support the entire claim.

Complete answer:
{inputs}

Cited passages:
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
    return create_llm_as_judge(
        prompt=FACT_CITATION_PROMPT,
        feedback_key="fact_citation_judgement",
        judge=build_judge_model(),
        output_schema=FACT_CITATION_SCHEMA,
    )


def evaluate_fact_citations(answer: str, passages_by_id: dict[str, dict], fact_judge: Callable) -> dict:
    """Return the share of citation groups fully supported by their passages."""
    matches = list(CITATION_GROUP_PATTERN.finditer(answer))
    groups = extract_citation_groups(answer)

    if not groups:
        return _evaluation_result(0, [])

    cited_ids = {citation_id for group in groups for citation_id in group}
    unknown_ids = sorted(cited_ids - passages_by_id.keys())
    if unknown_ids:
        raise ValueError(f"Answer used unknown citation IDs: {', '.join(unknown_ids)}")

    results = []
    for number, (match, citation_ids) in enumerate(zip(matches, groups, strict=True), start=1):
        marked_answer = (
            f"{answer[:match.start()]}<TARGET_CITATION>{match.group()}"
            f"</TARGET_CITATION>{answer[match.end():]}"
        )
        result = fact_judge(
            inputs={"answer_with_target": marked_answer},
            outputs={
                "evidence_passages": [
                    passages_by_id[citation_id].get("text", "")
                    for citation_id in citation_ids
                ]
            },
        )
        if not isinstance(result.get("supported"), bool):
            raise ValueError(f"Fact-citation judge returned an invalid result: {result!r}")

        results.append({
            "group_number": number,
            "citation_ids": list(citation_ids),
            **result,
        })

    supported_count = sum(result["supported"] for result in results)
    return _evaluation_result(supported_count, results)


def _evaluation_result(supported_count: int, group_results: list[dict]) -> dict:
    group_count = len(group_results)
    return {
        "key": "fact_citation",
        "score": supported_count / group_count if group_count else 0.0,
        "comment": (
            f"{supported_count}/{group_count} citation groups were fully supported."
            if group_count
            else "The answer contained no citation groups."
        ),
        "metadata": {
            "evaluator_version": EVALUATOR_VERSION,
            "citation_group_count": group_count,
            "supported_group_count": supported_count,
            "group_results": group_results,
        },
    }
