# pyright: reportMissingImports=false
from langchain_cohere import CohereRerank
from langchain_core.documents import Document

from arxiv_rag.graph.state import WorkflowGraphState
from arxiv_rag.logging import get_logger
from arxiv_rag.retrieval import BuiltContext, RetrievalContext

COHERE_RERANK_MODEL = "rerank-v4.0-pro"
RERANK_TOP_N = 10

log = get_logger(__name__)


def _get_reranker() -> CohereRerank:
    return CohereRerank(model=COHERE_RERANK_MODEL, top_n=RERANK_TOP_N)


def rerank_context(built_context: BuiltContext, query: str) -> BuiltContext:
    documents = [
        Document(page_content=text, metadata={"citation_id": citation_id})
        for citation_id, text in built_context.passages_by_id.items()
    ]

    for citation_id, text in built_context.passages_by_id.items():
        print("\n -- ")
        print(f"Citation_ID: {citation_id} \n")
        print(f"Passage: {text} \n")
        print(" -- \n")

    log.info(
        "reranking %d retrieved passages with model=%s top_n=%d",
        len(documents),
        COHERE_RERANK_MODEL,
        RERANK_TOP_N,
    )

    try:
        reranked_documents = _get_reranker().compress_documents(documents, query)
    except Exception as error:
        log.exception("Cohere passage reranking failed")
        raise RuntimeError("Could not rerank retrieved evidence.") from error

    for document in reranked_documents:
        print("\n -- ")
        print(f"Reranked Citation_ID: {document.metadata.get('citation_id')} \n")
        print(f"Reranked Passage: {document.page_content} \n")
        print(" -- \n")

    passages_by_id = {}
    citations = {}
    context_blocks = []

    for document in reranked_documents:

        citation_id = document.metadata.get("citation_id")
        passage = document.page_content

        citation = built_context.context.citations[citation_id]

        section = citation.label.partition(" — ")[2] or citation.label

        passages_by_id[citation_id] = passage
        citations[citation_id] = citation

        context_blocks.append(
            f"[{citation_id}]\n"
            f"Section: {section}\n"
            f"Text: {passage}"
        )

    log.info(
        "reranked %d retrieved passages down to %d passages",
        len(documents),
        len(reranked_documents),
    )

    return BuiltContext(
        context=RetrievalContext(
            text="\n\n---\n\n".join(context_blocks),
            citations=citations,
        ),
        passages_by_id=passages_by_id,
    )


def rerank_node(state: WorkflowGraphState) -> dict:
    built_context = state["current_built_context"]
    query = state["retrieval_query"]

    if built_context is None:
        raise ValueError("current_built_context must not be None for rerank_node")
    if query is None:
        raise ValueError("retrieval_query must not be None for rerank_node")

    return {"current_reranked_context": rerank_context(built_context, query)}
