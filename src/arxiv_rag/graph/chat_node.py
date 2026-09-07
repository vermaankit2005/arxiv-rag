from __future__ import annotations

# pyright: reportMissingImports=false
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from arxiv_rag.answering.chat_model import get_chat_model
from arxiv_rag.answering.generator import CITATION_MARKER_PATTERN, URL_PATTERN
from arxiv_rag.graph.prompts import CHAT_SYSTEM_PROMPT
from arxiv_rag.logging import get_logger

if TYPE_CHECKING:
    from arxiv_rag.graph.workflow_graph import WorkflowGraphState

log = get_logger(__name__)


def _validate_chat_answer(answer: str) -> None:
    if URL_PATTERN.search(answer):
        raise RuntimeError("The chat answer must not contain model-written URLs.")
    if CITATION_MARKER_PATTERN.search(answer):
        raise RuntimeError("The chat answer must not contain passage markers.")


def chat_node(state: WorkflowGraphState) -> dict:
    user_prompt = f"""
    Conversation history:
        <conversation>
        {state["messages"]}
        </conversation>

    Current user message:
        <current_message>
        {state["original_question"]}
        </current_message>

    Reply to the current user message.
    """

    llm = get_chat_model()
    if llm is None:
        raise RuntimeError("Could not get chat model.")
    try:
        response = llm.invoke(
            [
                SystemMessage(content=CHAT_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
    except Exception as error:
        log.exception("Ollama answer generation failed")
        raise RuntimeError("Could not generate an answer.") from error

    answer = response.content.strip()
    try:
        _validate_chat_answer(answer)
    except RuntimeError as error:
        log.warning("Rejected chat answer: %s", error)
        raise

    messages = [
        HumanMessage(content=state["original_question"]),
        AIMessage(content=answer),
    ]

    return {
        "messages": messages,
        "answer": answer,
        "current_built_context": None,
    }
