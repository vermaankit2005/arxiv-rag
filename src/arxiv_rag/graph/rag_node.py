# pyright: reportMissingImports=false
from langchain_core.messages import AIMessage, HumanMessage

from arxiv_rag.answering import generate_answer
from arxiv_rag.graph.state import WorkflowGraphState
from arxiv_rag.retrieval import PaperRetriever


def rag_node(state: WorkflowGraphState) -> dict:
    answer_request = state["answer_request"]
    retrieval_query = state["retrieval_query"]

    if answer_request is None:
        raise ValueError("answer_request must not be None for RAG route")
    if retrieval_query is None:
        raise ValueError("retrieval_query must not be None for RAG route")

    built_context = PaperRetriever().retrieve(retrieval_query)
    answer = generate_answer(
        answer_request,
        built_context.context,
        answer_mode=state["answer_mode"],
    )

    return {
        "messages": [
            HumanMessage(content=state["original_question"]),
            AIMessage(content=answer),
        ],
        "answer": answer,
        "current_context": built_context,
    }
