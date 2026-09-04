# State typedict for the workflow graph
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph, add_messages
from pydantic import BaseModel, Field

from arxiv_rag.answering import AnswerMode, generate_answer
from arxiv_rag.answering.chat_model import get_chat_model
from arxiv_rag.graph.prompts import CHAT_SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT
from arxiv_rag.logging import get_logger
from arxiv_rag.retrieval import BuiltContext, PaperRetriever

log = get_logger(__name__)


# State
class WorkflowGraphState(TypedDict):
    original_question: str
    messages: Annotated[list[BaseMessage], add_messages]

    route: Literal["chat", "rag"]
    rewritten_question: str | None
    retrieval_query: str | None

    answer_mode: Literal["standard", "easy"]
    current_evidence: BuiltContext | None

    current_built_context: BuiltContext | None
    answer: str


# Pydantic model for the output of the router node
class RouterNodeOutput(BaseModel):
    route: Literal["chat", "rag"] = Field(..., description="The route to take: 'chat' or 'rag'")
    rewritten_question: str | None = Field(..., description="The rewritten question only when the route is 'rag'")
    retrieval_query: str | None = Field(..., description="The topic-only search query when the route is 'rag'")
    style_override: Literal["easy"] | None = Field(..., description=("Use 'easy' when the user explicitly requests "
                                                                     "beginner-friendly wording"), )


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
    except Exception as error:
        log.exception("Ollama answer generation failed")
        raise RuntimeError("Could not generate an answer.") from error

    if not isinstance(response, RouterNodeOutput):
        log.error("Response is not of type RouterNodeOutput")
        raise RuntimeError("Could not generate an answer.")

    effective_mode = (
        "easy"
        if state["answer_mode"] == "easy" or response.style_override == "easy"
        else "standard"
    )

    log.info("Original question: %s", state["original_question"])
    if response.route == "rag":
        log.info("RAG route selected. Rewritten question: %s", response.rewritten_question)
        log.info("Retrieval query: %s", response.retrieval_query)

    return {
        "route": response.route,
        "rewritten_question": response.rewritten_question,
        "retrieval_query": response.retrieval_query,
        "answer_mode": effective_mode,
    }


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
        response = llm.invoke([SystemMessage(content=CHAT_SYSTEM_PROMPT), HumanMessage(content=user_prompt)])
    except Exception as error:
        log.exception("Ollama answer generation failed")
        raise RuntimeError("Could not generate an answer.") from error

    messages = [
        HumanMessage(content=state["original_question"]),
        AIMessage(content=response.content),
    ]

    return {
        "messages": messages,
        "answer": response.content,
        "current_built_context": None,
    }


def route_edge(state: WorkflowGraphState) -> Literal["chat_node", "rag_node"]:
    log.debug("Routing to %s", state["route"])
    if state["route"] == "chat":
        return "chat_node"
    elif state["route"] == "rag":
        return "rag_node"
    else:
        raise ValueError(f"Invalid route: {state['route']}")


def rag_node(state: WorkflowGraphState) -> dict:
    if state["rewritten_question"] is None:
        raise ValueError("rewritten_question must not be None for RAG route")
    if state["retrieval_query"] is None:
        raise ValueError("retrieval_query must not be None for RAG route")

    built = PaperRetriever().retrieve_context_with_details(state["retrieval_query"])
    answer = generate_answer(
        state["rewritten_question"], built.context, answer_mode=state["answer_mode"]
    )

    return {
        "messages": [HumanMessage(content=state["original_question"]), AIMessage(content=answer)],
        "answer": answer,
        "current_built_context": built,
    }


graph = StateGraph(WorkflowGraphState)

graph.add_node("route_node", route_node)
graph.add_node("chat_node", chat_node)
graph.add_node("rag_node", rag_node)

graph.add_edge(START, "route_node")
graph.add_conditional_edges(
    "route_node", route_edge, {"chat_node": "chat_node", "rag_node": "rag_node"}
)
graph.add_edge("chat_node", END)  # Loop back to route_node for next user message

workflow_graph = graph.compile(checkpointer=InMemorySaver())


def invoke_workflow_graph(question: str, thread_id: str, answer_mode: AnswerMode = "standard") -> WorkflowGraphState:
    config = {"configurable": {"thread_id": thread_id}}

    final_state = workflow_graph.invoke(
        input={
            "original_question": question,
            "answer_mode": answer_mode,
        },
        config=config,
    )

    return final_state


if __name__ == "__main__":
    # Example usage
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
