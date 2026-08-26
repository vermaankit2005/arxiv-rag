import httpx

from arxiv_rag.ingestion.documents import convert_loaded_paper_to_documents
from arxiv_rag.ingestion.load_docs import ArxivSampleHTMLLoader
from arxiv_rag.ingestion.vector_db_ingest import VectorStore, get_vector_store
from arxiv_rag.loading.arxiv import load_paper
from arxiv_rag.logging import get_logger

log = get_logger(__name__)

CHECK_FOR_EXISTING_DOCUMENTS_IN_DB = False  # Set to True to skip unchanged documents

def ingest_documents() -> VectorStore:
    """Load each paper, embed it, and write it to the store."""
    loader = ArxivSampleHTMLLoader()
    docs_name = loader.get_docs_name()
    log.info("found %d papers", len(docs_name))

    vector_store = get_vector_store(create_if_missing=True)

    added = 0
    with httpx.Client(follow_redirects=True) as http_client:
        for i, doc_name in enumerate(docs_name, start=1):

            arxiv_id = doc_name.replace(".html", "")
            loaded_paper = load_paper(arxiv_id, http_client)
            documents = convert_loaded_paper_to_documents(loaded_paper)

            if CHECK_FOR_EXISTING_DOCUMENTS_IN_DB:
                existing_document_by_id = {
                    document.id: document
                    for document in vector_store.get([document.id for document in documents])
                }
                documents = [
                    document
                    for document in documents if existing_document_by_id.get(document.id) != document
                ]

                if not documents:
                    log.info("[%d/%d] skipped unchanged %s", i, len(docs_name), arxiv_id)
                    continue

            log.info(
                "[%d/%d] embedding %s (%d new or changed passages)",
                i,
                len(docs_name),
                arxiv_id,
                len(documents),
            )
            vector_store.add(documents)
            added += len(documents)
            log.info("[%d/%d] stored %s", i, len(docs_name), arxiv_id)

    log.info("done. stored %d passages from %d papers", added, len(docs_name))
    return vector_store

if __name__ == "__main__":
    ingest_documents()
