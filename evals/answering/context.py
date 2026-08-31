from collections.abc import Callable

from arxiv_rag import answering
from arxiv_rag.retrieval import Citation, RetrievalContext


def build_section_breadcrumbs(section_path: list[str]) -> str:
    if not section_path:
        return "Unsectioned"
    return " > ".join(section_path)


def build_retrieval_context(context_passages: list[dict], preserve_passage_ids: bool) -> RetrievalContext:
    citations = {}
    formatted_passages = []

    for passage in context_passages:
        citation_id = passage["id"] if preserve_passage_ids else f"P{len(citations) + 1}"
        section_breadcrumbs = build_section_breadcrumbs(passage["section_path"])
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


def generate_answer_for_evaluation(inputs: dict, context_builder: Callable[[list[dict]], RetrievalContext]) -> dict:
    question = inputs.get("question", "")
    context_passages = inputs.get("context_passages", [])
    retrieval_context = context_builder(context_passages)
    answer = answering.generate_answer(question, retrieval_context)
    return {"answer": answer}
