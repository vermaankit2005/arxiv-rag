import re

from arxiv_rag.answering.chat_model import ChatModel, get_chat_model
from arxiv_rag.logging import get_logger
from arxiv_rag.retrieval import RetrievalContext

log = get_logger(__name__)

INSUFFICIENT_EVIDENCE_ANSWER = "I don't know the answer based on the provided evidence."
CITATION_ID_PATTERN = re.compile(r"P\d+")
CITATION_MARKER_PATTERN = re.compile(r"\[P\d+(?:\s*,\s*P\d+)*\]")
URL_PATTERN = re.compile(r"https?://", re.IGNORECASE)


def citation_ids_in_text(text: str) -> list[str]:
    """Return passage IDs from [P1] and [P1, P2] markers, in order of first appearance."""
    ids = []
    seen = set()

    for marker in CITATION_MARKER_PATTERN.finditer(text):
        for citation_id in CITATION_ID_PATTERN.findall(marker.group()):
            if citation_id not in seen:
                seen.add(citation_id)
                ids.append(citation_id)

    return ids


def _build_prompt(question: str, context: RetrievalContext) -> str:
    return (
        "Answer the question using only the supplied source passages.\n\n"
        "Rules:\n"
        "- Answer directly and use clear Markdown.\n"
        "- Use short paragraphs, headings only when useful, and bullets for real lists.\n"
        "- Do not repeat the same point in different words.\n"
        "- Put a passage ID such as [P1] immediately after every factual sentence.\n"
        "- When one sentence needs more than one passage, write separate markers with a space: [P1] [P2].\n"
        "- Never combine IDs in one pair of brackets. Do not write [P1, P2] or [P1,P2].\n"
        "- Use only passage IDs that appear in the supplied passages.\n"
        "- Do not write or invent URLs.\n"
        "- Never guess or fill in information that the supplied passages do not support.\n"
        "- If the passages support only part of the question, answer that part and clearly state what the evidence does not specify.\n"
        f"- If the passages support none of the requested information, reply exactly: {INSUFFICIENT_EVIDENCE_ANSWER}\n\n"
        f"Question:\n{question}\n\n"
        f"Source passages:\n{context.text}"
    )


def _validate_answer(answer: str, context: RetrievalContext) -> None:
    if answer == INSUFFICIENT_EVIDENCE_ANSWER:
        return

    if URL_PATTERN.search(answer):
        raise RuntimeError("The generated answer must not contain model-written URLs.")

    citation_ids = set(citation_ids_in_text(answer))
    if not citation_ids:
        raise RuntimeError("The generated answer must contain at least one citation.")

    unknown_ids = citation_ids - context.citations.keys()
    if unknown_ids:
        unknown = ", ".join(sorted(unknown_ids))
        raise RuntimeError(f"The generated answer used unknown citation IDs: {unknown}.")


def generate_answer(question: str, context: RetrievalContext, model: ChatModel | None = None) -> str:
    """Generate normal answer text containing validated inline passage IDs."""
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")

    if not context.text.strip() or not context.citations:
        log.warning("no evidence in context, answering with the insufficient-evidence reply")
        return INSUFFICIENT_EVIDENCE_ANSWER

    chat_model = model or get_chat_model()

    response = chat_model.invoke(_build_prompt(question, context))
    answer = response.content.strip()

    _validate_answer(answer, context)

    cited = ", ".join(citation_ids_in_text(answer))
    log.info("generated answer from %d passages, cited %s", len(context.citations), cited)
    return answer
