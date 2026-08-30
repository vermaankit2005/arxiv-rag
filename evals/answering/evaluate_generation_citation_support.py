import os
import re
from collections.abc import Callable

from dotenv import load_dotenv
from langchain_ollama import ChatOllama  # pyright: ignore[reportMissingImports]
from langsmith import Client
from openevals.llm import create_llm_as_judge  # pyright: ignore[reportMissingImports]

from arxiv_rag import answering
from arxiv_rag.retrieval import Citation, RetrievalContext

load_dotenv()

LANGSMITH_DATASET_NAME = "generation_quality_dataset"
EXPERIMENT_PREFIX = "generation_citation_support"
JUDGE_MODEL_NAME = "qwen3.8:27b"
EXPERIMENT_METADATA = {
    "metric": "citation_support",
    "dataset": LANGSMITH_DATASET_NAME,
    "generator_model": "qwen3.8:27b",
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

Judge = Callable[..., dict]

judge_model = ChatOllama(
    model=JUDGE_MODEL_NAME,
    base_url=os.environ["OLLAMA_BASE_URL"],
    temperature=0,
)

citation_support_judge = create_llm_as_judge(
    prompt=CITATION_SUPPORT_PROMPT,
    feedback_key="citation_support_pair",
    judge=judge_model,
    choices=[False, True],
)


def _build_section_breadcrumbs(section_path: list[str]) -> str:
    if not section_path:
        return "Unsectioned"
    return " > ".join(section_path)


def _build_retrieval_context(context_passages: list[dict]) -> RetrievalContext:
    """Build generation context while preserving the dataset's stable passage IDs."""
    citations = {}
    formatted_passages = []

    for passage in context_passages:
        citation_id = passage["id"]
        section_breadcrumbs = _build_section_breadcrumbs(passage["section_path"])

        url = f"https://arxiv.org/html/{passage['arxiv_id']}{passage['location']}"

        citations[citation_id] = Citation(
            label=f"{passage['arxiv_id']} — {section_breadcrumbs}",
            url=url,
        )
        formatted_passages.append(
            f"[{citation_id}]\n"
            f"Section: {section_breadcrumbs}\n"
            f"Text: {passage['text']}"
        )

    return RetrievalContext(
        text="\n\n---\n\n".join(formatted_passages),
        citations=citations,
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


def generate_answer_for_citation_support(inputs: dict) -> dict:
    """Generate an answer from the frozen context using shipping answer code."""
    question = inputs.get("question", "")
    context_passages = inputs.get("context_passages", [])
    retrieval_context = _build_retrieval_context(context_passages)
    answer = answering.generate_answer(question, retrieval_context)
    return {"answer": answer}


def evaluate_citation_support(
    inputs: dict, outputs: dict, judge: Judge = citation_support_judge
) -> dict:
    """Return the share of cited statement-passage pairs supported by that passage."""
    answer = outputs.get("answer", "")
    pairs = _extract_statement_citation_pairs(answer)
    if not pairs:
        return {
            "key": "citation_support",
            "score": 0.0,
            "comment": "The answer contained no statement-citation pairs.",
        }

    passages_by_id = {
        passage["id"]: passage["text"]
        for passage in inputs.get("context_passages", [])
    }

    unknown_ids = sorted({citation_id for _, citation_id in pairs} - passages_by_id.keys())
    if unknown_ids:
        unknown = ", ".join(unknown_ids)
        raise ValueError(f"Answer used unknown citation IDs: {unknown}")

    supported_count = 0

    for statement, citation_id in pairs:

        result = judge(
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
    """Run citation support against the frozen generation dataset in LangSmith."""
    client = Client()
    client.evaluate(
        generate_answer_for_citation_support,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[evaluate_citation_support],
        metadata=EXPERIMENT_METADATA,
        experiment_prefix=EXPERIMENT_PREFIX,
        description=(
            "For every citation attached to a generated statement, judge whether "
            "that exact frozen passage supports the complete statement. The score "
            "is supported statement-passage pairs divided by all cited pairs."
        ),
    )


if __name__ == "__main__":
    run_citation_support()
