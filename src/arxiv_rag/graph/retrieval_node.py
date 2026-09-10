# pyright: reportMissingImports=false
from arxiv_rag.graph.state import WorkflowGraphState
from arxiv_rag.retrieval import PaperRetriever


def retrieval_node(state: WorkflowGraphState) -> dict:
    if state["retrieval_query"] is None:
        raise ValueError("retrieval_query must not be None for RAG route")

    built_context = PaperRetriever().retrieve_context_with_details(state["retrieval_query"])
    return {"current_built_context": built_context}
