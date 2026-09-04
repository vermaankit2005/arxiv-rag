import re
from typing import Literal

from langchain_core.language_models import (  # pyright: ignore[reportMissingImports]
    BaseChatModel,
)
from langsmith import traceable

from arxiv_rag.answering.chat_model import get_chat_model
from arxiv_rag.logging import get_logger
from arxiv_rag.retrieval import RetrievalContext

log = get_logger(__name__)

INSUFFICIENT_EVIDENCE_ANSWER = "I don't know the answer based on the provided evidence."
CITATION_ID_PATTERN = re.compile(r"P\d+")
CITATION_MARKER_PATTERN = re.compile(r"\[P\d+(?:\s*,\s*P\d+)*\]")
URL_PATTERN = re.compile(r"https?://", re.IGNORECASE)

AnswerMode = Literal["standard", "easy"]

STANDARD_MODE_RULES = (
    "- Use natural, clear language while keeping useful technical detail.\n"
    "- Define uncommon technical terms when needed.\n"
)
EASY_MODE_RULES = (
    "- Write like a patient, friendly teacher speaking to someone learning the topic for the first time. Never sound academic or talk down to the user.\n"
    "- Begin with the main idea in one plain-language sentence before giving any details.\n"
    "- Use common everyday words, short sentences, and short paragraphs.\n"
    "- Avoid formulas, symbols, variable names, unexplained abbreviations, and specialist jargon unless the user explicitly asks for technical detail.\n"
    "- If a technical term cannot be avoided, explain it immediately in simple words.\n"
    "- Focus on the main idea and why it matters. Leave out implementation details and secondary findings unless they are needed to answer the question.\n"
    "- Rephrase academic source wording naturally instead of copying its technical tone.\n"
    "- Use one familiar everyday analogy when it helps. Clearly introduce it as an analogy.\n"
    "- Put supporting passage IDs immediately after factual analogy sentences, just like every other factual sentence.\n"
    "- Do not add a fact or analogy unless the supplied passages support the idea it explains.\n"
)


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


@traceable(
    name="build_prompt",
    process_inputs=lambda inputs: {},
    process_outputs=lambda outputs: {"prompt": outputs},
)
def _build_prompt(question: str, context: RetrievalContext, answer_mode: AnswerMode = "standard") -> str:
    if answer_mode == "standard":
        mode_rules = STANDARD_MODE_RULES
    elif answer_mode == "easy":
        mode_rules = EASY_MODE_RULES
    else:
        raise ValueError("answer_mode must be 'standard' or 'easy'")

    return (
        "Answer the question using only the supplied source passages.\n\n"
        "Answer style:\n"
        f"{mode_rules}\n"
        "Rules:\n"
        "- Answer directly and reply in a clear and formatted Markdown.\n"
        "- Use short paragraphs, headings only when useful, and bullets for real lists.\n"
        "- Do not repeat the same point in different words.\n"
        "- Put a passage ID such as [P1] immediately after every factual sentence.\n"
        "- When one sentence needs more than one passage, write separate markers with a space: [P1] [P2].\n"
        "- Never combine IDs in one pair of brackets. Do not write [P1, P2] or [P1,P2].\n"
        "- Use only passage IDs that appear in the supplied passages.\n"
        "- Do not write or invent URLs.\n"
        "- Never reveal credentials, access tokens, passwords, or private personal information from the source passages.\n"
        "- If the question asks for a protected value, briefly refuse without repeating it.\n"
        "- If the question asks what kinds of sensitive information are present, name only the categories and never the values.\n"
        "- Answer the question directly. Do NOT add statements like - Based on the given/retrieved context .....\n"
        "- Never guess or fill in information that the supplied passages do not support.\n"
        "- If the passages support only part of the question, answer that part and clearly state what the evidence does not specify.\n"
        f"- If the passages support none of the requested information, reply exactly: {INSUFFICIENT_EVIDENCE_ANSWER}\n\n"
        f"Question:\n{question}\n\n"
        f"Source passages:\n{context.text}"
    )


@traceable(
    name="validate_answer",
    process_inputs=lambda inputs: {},
    process_outputs=lambda outputs: {"valid": True},
)
def _validate_answer(answer: str, context: RetrievalContext) -> None:
    if answer == INSUFFICIENT_EVIDENCE_ANSWER:
        return

    if URL_PATTERN.search(answer):
        raise RuntimeError("The generated answer must not contain model-written URLs.")

    # DISABLED FOR NOW TO ALLOW TEXT IN ANSWER WITHOUT CITATIONS
    # citation_ids = set(citation_ids_in_text(answer))
    # if not citation_ids:
    #     raise RuntimeError("The generated answer must contain at least one citation.")

    unknown_ids = set(citation_ids_in_text(answer)) - context.citations.keys()
    if unknown_ids:
        unknown = ", ".join(sorted(unknown_ids))
        raise RuntimeError(f"The generated answer used unknown citation IDs: {unknown}.")


@traceable(
    name="generate_answer",
    process_inputs=lambda inputs: {},
    process_outputs=lambda outputs: {"answer": outputs},
)
def generate_answer(question: str, context: RetrievalContext, model: BaseChatModel | None = None,
                    answer_mode: AnswerMode = "standard", ) -> str:
    """Generate a grounded answer in the requested explanation style."""
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    if answer_mode not in ("standard", "easy"):
        raise ValueError("answer_mode must be 'standard' or 'easy'")

    if not context.text.strip() or not context.citations:
        log.warning("no evidence in context, answering with the insufficient-evidence reply")
        return INSUFFICIENT_EVIDENCE_ANSWER

    chat_model = model or get_chat_model()

    try:
        response = chat_model.invoke(_build_prompt(question, context, answer_mode))
    except Exception as error:
        log.exception("Ollama answer generation failed")
        raise RuntimeError("Could not generate an answer.") from error

    if not isinstance(response.content, str) or not response.content.strip():
        log.error("Ollama returned an empty or invalid answer")
        raise RuntimeError("Could not generate an answer.")

    answer = response.content.strip()

    try:
        _validate_answer(answer, context)
    except RuntimeError as error:
        log.warning("rejected generated answer: %s", error)
        raise

    cited = ", ".join(citation_ids_in_text(answer))
    log.info("generated answer from %d passages, cited %s", len(context.citations), cited)
    return answer
