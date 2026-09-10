# pyright: reportMissingImports=false
from langchain_cohere import CohereRerank
from langchain_core.documents import Document

from arxiv_rag.logging import get_logger

COHERE_RERANK_MODEL = "rerank-v4.0-pro"
RERANK_TOP_N = 10

log = get_logger(__name__)


def _get_reranker() -> CohereRerank:
    return CohereRerank(model=COHERE_RERANK_MODEL, top_n=RERANK_TOP_N)


def rerank_passages(passages_by_id: dict[str, str], query: str) -> list[str]:
    """Return passage IDs in Cohere relevance order."""
    if not passages_by_id:
        return []

    documents = [
        Document(page_content=text, metadata={"citation_id": citation_id})
        for citation_id, text in passages_by_id.items()
    ]

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

    ranked_ids = []
    for document in reranked_documents:
        citation_id = document.metadata.get("citation_id")
        if not isinstance(citation_id, str) or citation_id not in passages_by_id:
            raise RuntimeError("Reranker returned invalid evidence.")
        ranked_ids.append(citation_id)

    log.info(
        "reranked %d retrieved passages down to %d passages",
        len(documents),
        len(ranked_ids),
    )

    return ranked_ids
