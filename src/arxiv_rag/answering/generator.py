import os
import re
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain_ollama import ChatOllama  # pyright: ignore[reportMissingImports]

from arxiv_rag.retrieval import RetrievalContext

MODEL_NAME = "qwen3.8:27b"
INSUFFICIENT_EVIDENCE_ANSWER = "I don't know the answer based on the provided evidence."
CITATION_PATTERN = re.compile(r"\[(P\d+)\]")
URL_PATTERN = re.compile(r"https?://", re.IGNORECASE)


class ChatResponse:
    content: str


class ChatModel(ABC):
    @abstractmethod
    def invoke(self, prompt: str) -> ChatResponse:
        ...

def _get_chat_model() -> ChatOllama:
    load_dotenv()
    base_url = os.environ.get("OLLAMA_BASE_URL")
    if not base_url:
        raise RuntimeError("OLLAMA_BASE_URL is not set. Add it to your .env file.")

    return ChatOllama(
        model=MODEL_NAME,
        base_url=base_url,
        temperature=0,
    )


def _build_prompt(question: str, context: RetrievalContext) -> str:
    return (
        "Answer the question using only the supplied source passages.\n\n"
        "Rules:\n"
        "- Answer directly and use clear Markdown.\n"
        "- Use short paragraphs, headings only when useful, and bullets for real lists.\n"
        "- Do not repeat the same point in different words.\n"
        "- Put a passage ID such as [P1] immediately after every factual sentence.\n"
        "- Use only passage IDs that appear in the supplied passages.\n"
        "- Use [P1] [P2] when one sentence needs more than one passage.\n"
        "- Do not write or invent URLs.\n"
        f"- If the evidence is insufficient, reply exactly: {INSUFFICIENT_EVIDENCE_ANSWER}\n\n"
        f"Question:\n{question}\n\n"
        f"Source passages:\n{context.text}"
    )


def _validate_answer(answer: str, context: RetrievalContext) -> None:
    if answer == INSUFFICIENT_EVIDENCE_ANSWER:
        return

    if URL_PATTERN.search(answer):
        raise RuntimeError("The generated answer must not contain model-written URLs.")

    citation_ids = set(CITATION_PATTERN.findall(answer))
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
        return INSUFFICIENT_EVIDENCE_ANSWER

    chat_model = model or _get_chat_model()

    response = chat_model.invoke(_build_prompt(question, context))

    answer = response.content.strip()

    _validate_answer(answer, context)

    return answer
