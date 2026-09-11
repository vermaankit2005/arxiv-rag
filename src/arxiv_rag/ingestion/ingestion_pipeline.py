from pathlib import Path
from uuid import uuid4

import httpx

from arxiv_rag.ingestion.documents import convert_loaded_paper_to_documents
from arxiv_rag.ingestion.load_docs import getLoader
from arxiv_rag.ingestion.vector_db_ingest import (
    CHROMA_COLLECTION_NAME,
    VectorStore,
    activate_collection,
    get_vector_store,
)
from arxiv_rag.loading.arxiv import NO_HTML_NOTE, load_paper
from arxiv_rag.logging import get_logger

log = get_logger(__name__)

HTML_DIR = Path(__file__).parents[3] / "data" / "raw" / "sampled_html"


def ingest_documents() -> VectorStore:

    """Build a complete corpus separately, then make it active in one step."""
    loader = getLoader()

    docs_name = loader.get_docs_name()

    log.info("found %d papers", len(docs_name))

    if not docs_name:
        raise RuntimeError("No papers were found for ingestion.")

    papers_and_documents = []
    failed_papers = []

    with httpx.Client(follow_redirects=True) as http_client:
        for doc_name in docs_name:
            arxiv_id = doc_name.removesuffix(".html")
            try:
                # Calling the loading component to load the paper.
                loaded_paper = load_paper(arxiv_id, http_client, HTML_DIR)

                if loaded_paper.note == NO_HTML_NOTE:
                    log.error("arXiv %s: %s", arxiv_id, loaded_paper.note)

                documents = convert_loaded_paper_to_documents(loaded_paper)
            except Exception:
                log.exception("failed to prepare arXiv paper %s", arxiv_id)
                failed_papers.append(arxiv_id)
                continue
            papers_and_documents.append((arxiv_id, documents))

    if failed_papers:
        failed = ", ".join(failed_papers)
        raise RuntimeError(f"Could not prepare all papers for ingestion: {failed}.")

    if not any(documents for _, documents in papers_and_documents):
        raise RuntimeError("No documents were prepared for ingestion.")

    collection_name = f"{CHROMA_COLLECTION_NAME}_staging_{uuid4().hex}"
    vector_store = get_vector_store(create_if_missing=True, collection_name=collection_name)
    added = 0

    try:
        for index, (arxiv_id, documents) in enumerate(papers_and_documents, start=1):
            log.info("[%d/%d] embedding %s (%d documents)", index, len(docs_name), arxiv_id, len(documents))

            vector_store.add(documents)
            added += len(documents)

            log.info("[%d/%d] stored %s", index, len(docs_name), arxiv_id)

        activate_collection(collection_name)
    except Exception:
        log.exception(
            "failed to build staging collection %s after adding %d documents",
            collection_name,
            added,
        )
        try:
            vector_store.delete()
        except Exception:
            log.exception("failed to delete incomplete staging collection %s", collection_name)
        raise

    log.info("activated complete Chroma collection %s", collection_name)
    log.info("done. stored %d documents from %d papers", added, len(docs_name))

    return vector_store


def main() -> int:
    try:
        ingest_documents()
    except RuntimeError as error:
        log.error("ingestion stopped: %s", error)
        return 1
    except Exception:
        log.exception("ingestion stopped unexpectedly")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
