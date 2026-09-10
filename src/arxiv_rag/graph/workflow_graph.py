# State typedict for the workflow graph
# pyright: reportMissingImports=false
from collections.abc import Iterator
from typing import Literal

from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from arxiv_rag.answering import AnswerMode
from arxiv_rag.graph.answer_node import answer_node
from arxiv_rag.graph.chat_node import chat_node
from arxiv_rag.graph.prompts import CHAT_SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT
from arxiv_rag.graph.retrieval_node import retrieval_node
from arxiv_rag.graph.route_node import RouterNodeOutput, route_node
from arxiv_rag.graph.state import WorkflowGraphState
from arxiv_rag.logging import get_logger

log = get_logger(__name__)

# Keep these imports available from this module for callers that used the old layout.
__all__ = [
    "CHAT_SYSTEM_PROMPT",
    "ROUTER_SYSTEM_PROMPT",
    "RouterNodeOutput",
    "WorkflowGraphState",
    "answer_node",
    "chat_node",
    "invoke_workflow_graph",
    "retrieval_node",
    "route_edge",
    "route_node",
]

def route_edge(state: WorkflowGraphState) -> Literal["chat_node", "retrieval_node"]:
    log.debug("Routing to %s", state["route"])
    if state["route"] == "chat":
        return "chat_node"
    elif state["route"] == "rag":
        return "retrieval_node"
    else:
        raise ValueError(f"Invalid route: {state['route']}")

graph = StateGraph(WorkflowGraphState)

graph.add_node("route_node", route_node)
graph.add_node("chat_node", chat_node)
graph.add_node("retrieval_node", retrieval_node)
graph.add_node("answer_node", answer_node)

graph.add_edge(START, "route_node")
graph.add_conditional_edges(
    "route_node", route_edge, {"chat_node": "chat_node", "retrieval_node": "retrieval_node"}
)
graph.add_edge("chat_node", END)
graph.add_edge("retrieval_node", "answer_node")
graph.add_edge("answer_node", END)

workflow_graph = graph.compile(checkpointer=InMemorySaver())


def invoke_workflow_graph(question: str, thread_id: str, answer_mode: AnswerMode = "standard") -> WorkflowGraphState:

    config = {"configurable": {"thread_id": thread_id}}

    final_state = workflow_graph.invoke(
        input={
            "original_question": question,
            "route": None,
            "answer_request": None,
            "retrieval_query": None,
            "answer_mode": answer_mode,
            "current_evidence": None,
            "current_built_context": None,
            "answer": "",
        },
        config=config,
    )

    return final_state


def stream_workflow_graph(question: str, thread_id: str, answer_mode: AnswerMode = "standard") \
        -> Iterator[WorkflowGraphState | str]:
    config = {"configurable": {"thread_id": thread_id}}

    state = workflow_graph.stream(
        input={
            "original_question": question,
            "route": None,
            "answer_request": None,
            "retrieval_query": None,
            "answer_mode": answer_mode,
            "current_evidence": None,
            "current_built_context": None,
            "answer": "",
        },
        config=config,
        stream_mode=["messages", "values"],
    )
    final_state = None

    for stream_mode, data in state:
        if stream_mode == "values":
            final_state = data
            continue

        message, metadata = data

        if metadata.get("langgraph_node") not in {"chat_node", "answer_node"}:
            continue
        if not isinstance(message, AIMessageChunk):
            continue

        if isinstance(message.content, str) and message.content:
            yield message.content

    if final_state is None:
        raise RuntimeError("The workflow did not produce any output.")

    yield final_state
