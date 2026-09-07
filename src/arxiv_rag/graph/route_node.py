from __future__ import annotations

# pyright: reportMissingImports=false
from typing import TYPE_CHECKING, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from arxiv_rag.answering.chat_model import get_chat_model
from arxiv_rag.graph.prompts import ROUTER_SYSTEM_PROMPT
from arxiv_rag.logging import get_logger

if TYPE_CHECKING:
    from arxiv_rag.graph.workflow_graph import WorkflowGraphState

log = get_logger(__name__)


class RouterNodeOutput(BaseModel):
    """Structured output expected from the router model."""

    route: Literal["chat", "rag"] = Field(..., description="The route to take: 'chat' or 'rag'")
    answer_request: str | None = Field(..., description="The answer request only when the route is 'rag'")
    retrieval_query: str | None = Field(..., description="The topic-only search query when the route is 'rag'")
    style_override: Literal["easy"] | None = Field(
        ..., description="Use 'easy' when the user explicitly requests beginner-friendly wording"
    )


def _fallback_to_rag(state: WorkflowGraphState, reason: str) -> dict:
    question = state["original_question"]
    log.warning("Invalid router output; falling back to RAG: %s", reason)
    return {
        "route": "rag",
        "answer_request": question,
        "retrieval_query": question,
        "answer_mode": state["answer_mode"],
    }


def _valid_router_response(response: RouterNodeOutput) -> bool:
    if response.route == "rag":
        return bool(
            response.answer_request
            and response.answer_request.strip()
            and response.retrieval_query
            and response.retrieval_query.strip()
        )
    return (
        not response.answer_request
        and not response.retrieval_query
        and response.style_override is None
    )


def route_node(state: WorkflowGraphState) -> dict:
    user_prompt = f"""
    Conversation history:
        <conversation>
        {state["messages"]}
        </conversation>

    Current user message:
        <current_message>
        {state["original_question"]}
        </current_message>

    Classify the current user message.
    """
    llm = get_chat_model()

    if llm is not None:
        llm = llm.with_structured_output(RouterNodeOutput)
    else:
        raise RuntimeError("Could not get chat model.")

    try:
        response = llm.invoke(
            [
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
    except ValueError as error:
        return _fallback_to_rag(state, str(error))
    except Exception as error:
        log.exception("Ollama answer generation failed")
        raise RuntimeError("Could not generate an answer.") from error

    if not isinstance(response, RouterNodeOutput):
        return _fallback_to_rag(state, "response did not match RouterNodeOutput")
    if not _valid_router_response(response):
        return _fallback_to_rag(state, "response fields violated the route contract")

    effective_mode = (
        "easy"
        if state["answer_mode"] == "easy" or response.style_override == "easy"
        else "standard"
    )

    log.info("Original question: %s", state["original_question"])
    if response.route == "rag":
        log.info("RAG route selected. Answer request: %s", response.answer_request)
        log.info("Retrieval query: %s", response.retrieval_query)
        log.info("Answer mode: %s", effective_mode)

    return {
        "route": response.route,
        "answer_request": response.answer_request,
        "retrieval_query": response.retrieval_query,
        "answer_mode": effective_mode,
    }
