# State typedict for the workflow graph
# pyright: reportMissingImports=false
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph, add_messages

from arxiv_rag.answering import AnswerMode
from arxiv_rag.graph.chat_node import chat_node
from arxiv_rag.graph.prompts import CHAT_SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT
from arxiv_rag.graph.rag_node import rag_node
from arxiv_rag.graph.route_node import RouterNodeOutput, route_node
from arxiv_rag.logging import get_logger
from arxiv_rag.retrieval import BuiltContext

log = get_logger(__name__)

# Keep these imports available from this module for callers that used the old layout.
__all__ = [
    "CHAT_SYSTEM_PROMPT",
    "ROUTER_SYSTEM_PROMPT",
    "RouterNodeOutput",
    "WorkflowGraphState",
    "chat_node",
    "invoke_workflow_graph",
    "rag_node",
    "route_edge",
    "route_node",
]

# State
class WorkflowGraphState(TypedDict):
    original_question: str
    messages: Annotated[list[BaseMessage], add_messages]

    route: Literal["chat", "rag"] | None
    answer_request: str | None
    retrieval_query: str | None

    answer_mode: Literal["standard", "easy"]
    current_evidence: BuiltContext | None

    current_built_context: BuiltContext | None
    answer: str


def route_edge(state: WorkflowGraphState) -> Literal["chat_node", "rag_node"]:
    log.debug("Routing to %s", state["route"])
    if state["route"] == "chat":
        return "chat_node"
    elif state["route"] == "rag":
        return "rag_node"
    else:
        raise ValueError(f"Invalid route: {state['route']}")

graph = StateGraph(WorkflowGraphState)

graph.add_node("route_node", route_node)
graph.add_node("chat_node", chat_node)
graph.add_node("rag_node", rag_node)

graph.add_edge(START, "route_node")
graph.add_conditional_edges(
    "route_node", route_edge, {"chat_node": "chat_node", "rag_node": "rag_node"}
)
graph.add_edge("chat_node", END)
graph.add_edge("rag_node", END)

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


if __name__ == "__main__":
    thread_id = "example_thread_id"
    answer_mode = "standard"

    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if (
                question.lower() == "exit"
                or question.lower() == "quit"
                or question.lower() == "bye"
        ):
            break

        result = invoke_workflow_graph(question, thread_id, answer_mode)
        print(result["messages"][-1].content)  # Print the last message content
