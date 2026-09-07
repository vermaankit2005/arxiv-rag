from __future__ import annotations

# pyright: reportMissingImports=false
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage

from arxiv_rag.answering import generate_answer
from arxiv_rag.retrieval import PaperRetriever

if TYPE_CHECKING:
    from arxiv_rag.graph.workflow_graph import WorkflowGraphState


def rag_node(state: WorkflowGraphState) -> dict:
    if state["answer_request"] is None:
        raise ValueError("answer_request must not be None for RAG route")
    if state["retrieval_query"] is None:
        raise ValueError("retrieval_query must not be None for RAG route")

    built = PaperRetriever().retrieve_context_with_details(state["retrieval_query"])
    answer = generate_answer(state["answer_request"], built.context, answer_mode=state["answer_mode"])

    return {
        "messages": [HumanMessage(content=state["original_question"]), AIMessage(content=answer)],
        "answer": answer,
        "current_built_context": built,
    }
