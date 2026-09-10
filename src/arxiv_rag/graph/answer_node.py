# pyright: reportMissingImports=false
from langchain_core.messages import AIMessage, HumanMessage

from arxiv_rag.answering import generate_answer
from arxiv_rag.graph.state import WorkflowGraphState


def answer_node(state: WorkflowGraphState) -> dict:
    if state["answer_request"] is None:
        raise ValueError("answer_request must not be None for RAG route")
    if state["current_evidence"] is None:
        raise ValueError("current_evidence must not be None for RAG route")

    answer = generate_answer(
        state["answer_request"],
        state["current_evidence"].context,
        answer_mode=state["answer_mode"],
    )

    return {
        "messages": [HumanMessage(content=state["original_question"]), AIMessage(content=answer)],
        "answer": answer,
        "current_built_context": state["current_evidence"],
    }
