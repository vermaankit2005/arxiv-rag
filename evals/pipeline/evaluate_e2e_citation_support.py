import re

from langsmith import Client
from openevals.llm import create_llm_as_judge  # pyright: ignore[reportMissingImports]

from evals.answering.judges import build_judge_model
from evals.pipeline import context as evaluation_context

LANGSMITH_DATASET_NAME = "pipeline_required_fact_coverage_dataset"
EXPERIMENT_PREFIX = "pipeline_citation_support"
JUDGE_MODEL_NAME = "gpt-oss:20b"
EXPERIMENT_METADATA = {
    "metric": "pipeline_citation_support",
    "dataset": LANGSMITH_DATASET_NAME,
    "generator_model": "gemma4:26b",
    "judge_model": JUDGE_MODEL_NAME,
}

CITATION_PATTERN = re.compile(r"\[(P\d+)\]")
CITATION_GROUP_PATTERN = re.compile(r"(?:\[P\d+\]\s*)+")
STATEMENT_BOUNDARY_PATTERN = re.compile(r"[.!?](?=\s)|\n+")

CITATION_SUPPORT_PROMPT = """
You are checking whether one cited source passage supports one factual statement.

A passage supports the statement when the complete factual meaning is stated in
that passage or follows directly from it. A passage does not support the statement
when it is merely related, contradicts the statement, or supports only part of it.
Use only the supplied passage. Do not use outside knowledge.

Question and statement:
{inputs}

Cited passage:
{outputs}
"""

judge_model = build_judge_model(JUDGE_MODEL_NAME)

citation_support_judge = create_llm_as_judge(
    prompt=CITATION_SUPPORT_PROMPT,
    feedback_key="citation_support_pair",
    judge=judge_model,
    choices=[False, True],
)


def _extract_statement_citation_pairs(answer: str) -> list[tuple[str, str]]:
    """Attach each citation marker to the factual statement immediately before it."""
    pairs = []
    previous_group_end = 0

    for citation_group in CITATION_GROUP_PATTERN.finditer(answer):

        statement_text = answer[previous_group_end:citation_group.start()]
        boundaries = list(STATEMENT_BOUNDARY_PATTERN.finditer(statement_text))

        for boundary in reversed(boundaries):
            if statement_text[boundary.end():].strip():
                statement_text = statement_text[boundary.end():]
                break

        statement = statement_text.strip(" \t\r\n-*#>.;:")
        if not statement:
            raise ValueError("Could not find a statement before a citation marker")

        citation_ids = dict.fromkeys(CITATION_PATTERN.findall(citation_group.group()))
        pairs.extend((statement, citation_id) for citation_id in citation_ids)
        previous_group_end = citation_group.end()

    return pairs


def evaluate_citation_support(inputs: dict, outputs: dict) -> dict:
    """Return the share of cited statement-passage pairs supported by that passage."""
    answer = outputs.get("answer", "")
    pairs = _extract_statement_citation_pairs(answer)
    if not pairs:
        return {
            "key": "citation_support",
            "score": 0.0,
            "comment": "The answer contained no statement-citation pairs.",
        }
    passages_by_id = outputs.get("retrieved_passages", {})

    unknown_ids = sorted({citation_id for _, citation_id in pairs} - passages_by_id.keys())
    if unknown_ids:
        unknown = ", ".join(unknown_ids)
        raise ValueError(f"Answer used unknown citation IDs: {unknown}")

    supported_count = 0

    for statement, citation_id in pairs:

        result = citation_support_judge(
            inputs={
                "question": inputs.get("question", ""),
                "statement": statement,
            },
            outputs={"passage": passages_by_id[citation_id]},
        )

        supported = result.get("score")
        if supported not in (True, False, 0, 1):
            raise ValueError(f"Citation-support judge returned an invalid score: {supported!r}")

        supported_count += 1 if supported else 0

    score = supported_count / len(pairs)
    summary = f"{supported_count}/{len(pairs)} statement-citation pairs were supported."
    return {
        "key": "citation_support",
        "score": score,
        "comment": summary,
    }


def run_citation_support() -> None:
    """Run citation support against the retrieved passages."""
    client = Client()
    client.evaluate(
        evaluation_context.generate_pipeline_answer_and_passages_for_evaluation,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[evaluate_citation_support],
        metadata=EXPERIMENT_METADATA,
        experiment_prefix=EXPERIMENT_PREFIX,
        description=(
            "For every citation attached to a generated statement, judge whether "
            "that exact retrieved passage supports the complete statement. The score "
            "is supported statement-passage pairs divided by all cited pairs."
        ),
        max_concurrency=1,
    )


if __name__ == "__main__":
    run_citation_support()
