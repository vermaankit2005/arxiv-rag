import httpx

from arxiv_rag.ingestion.documents import convert_loaded_paper_to_documents
from arxiv_rag.ingestion.load_docs import ArxivSampleHTMLLoader
from arxiv_rag.ingestion.vector_db_ingest import VectorStore, get_vector_store
from arxiv_rag.loading.arxiv import load_paper
from arxiv_rag.logging import get_logger

log = get_logger(__name__)


def ingest_documents() -> VectorStore:
    """Rebuild the complete local corpus from the cached source HTML."""
    loader = ArxivSampleHTMLLoader()
    docs_name = loader.get_docs_name()
    log.info("found %d papers", len(docs_name))

    vector_store = get_vector_store(create_if_missing=True)
    vector_store.reset()
    log.info("cleared the existing Chroma collection for a fresh rebuild")

    added = 0
    with httpx.Client(follow_redirects=True) as http_client:
        for index, doc_name in enumerate(docs_name, start=1):
            arxiv_id = doc_name.removesuffix(".html")
            loaded_paper = load_paper(arxiv_id, http_client)
            documents = convert_loaded_paper_to_documents(loaded_paper)

            log.info(
                "[%d/%d] embedding %s (%d documents)",
                index,
                len(docs_name),
                arxiv_id,
                len(documents),
            )
            vector_store.add(documents)
            added += len(documents)
            log.info("[%d/%d] stored %s", index, len(docs_name), arxiv_id)

    log.info("done. stored %d documents from %d papers", added, len(docs_name))
    return vector_store


if __name__ == "__main__":
    ingest_documents()
