import hashlib
import json
import time
from pathlib import Path

import httpx
from langchain_core.documents import Document

from arxiv_rag.ingestion.documents import convert_loaded_paper_to_documents
from arxiv_rag.ingestion.load_docs import ArxivCorpusHtmlLoader, get_loader
from arxiv_rag.ingestion.vector_db_ingest import get_vector_store
from arxiv_rag.ingestion.vector_store import VectorStore, versioned_collection_name
from arxiv_rag.loading.arxiv import NO_HTML_NOTE, load_paper
from arxiv_rag.logging import get_logger

log = get_logger(__name__)

SAMPLE_HTML_DIR = Path(__file__).parents[3] / "data" / "raw" / "sampled_html"
CORPUS_HTML_DIR = Path(__file__).parents[3] / "data" / "raw" / "corpus_html"
INGESTION_CHECKPOINT_FILE = Path(__file__).parents[3] / "data" / "ingestion_checkpoint.json"
MAX_ADD_ATTEMPTS = 5


def _corpus_hash(arxiv_ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(arxiv_ids).encode("utf-8")).hexdigest()


def _write_checkpoint(collection_name: str, corpus_hash: str, next_index: int, document_count: int) -> None:
    checkpoint = {
        "collection_name": collection_name,
        "corpus_hash": corpus_hash,
        "next_index": next_index,
        "document_count": document_count,
    }
    INGESTION_CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = INGESTION_CHECKPOINT_FILE.with_suffix(".tmp")
    temporary_file.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    temporary_file.replace(INGESTION_CHECKPOINT_FILE)


def _load_checkpoint(arxiv_ids: list[str]) -> tuple[str, str, int, int]:
    corpus_hash = _corpus_hash(arxiv_ids)

    if not INGESTION_CHECKPOINT_FILE.exists():
        collection_name = versioned_collection_name("arxiv_papers")
        _write_checkpoint(collection_name, corpus_hash, 0, 0)
        return collection_name, corpus_hash, 0, 0

    try:
        checkpoint = json.loads(INGESTION_CHECKPOINT_FILE.read_text(encoding="utf-8"))
        checkpoint_hash = str(checkpoint["corpus_hash"])
        collection_name = str(checkpoint["collection_name"])
        next_index = int(checkpoint["next_index"])
        document_count = int(checkpoint["document_count"])

    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid ingestion checkpoint: {INGESTION_CHECKPOINT_FILE}") from error

    if checkpoint_hash != corpus_hash:
        raise RuntimeError(f"The ingestion corpus changed. Delete {INGESTION_CHECKPOINT_FILE} to start a new staging build.")
    if next_index < 0 or next_index > len(arxiv_ids):
        raise RuntimeError(f"Invalid ingestion checkpoint index in {INGESTION_CHECKPOINT_FILE}: {next_index}")

    return collection_name, corpus_hash, next_index, document_count


def _retry_after_seconds(error: Exception) -> float:
    payload = getattr(error, "error", None)
    if not isinstance(payload, dict):
        return 0

    try:
        return max(0, float(payload.get("retry_after", 0)))
    except (TypeError, ValueError):
        return 0


def _add_documents_with_retry(vector_store: VectorStore, documents: list[Document]) -> None:
    for attempt in range(1, MAX_ADD_ATTEMPTS + 1):
        try:
            vector_store.add(documents)
            return
        except Exception as error:
            if attempt == MAX_ADD_ATTEMPTS:
                raise

            delay = max(2 ** (attempt - 1), _retry_after_seconds(error))
            log.warning("embedding attempt %d/%d failed; retrying in %.0f seconds: %s", attempt, MAX_ADD_ATTEMPTS, delay, error)
            time.sleep(delay)


def ingest_documents() -> VectorStore:
    """Build or resume a staging corpus one paper at a time, then make it active."""
    loader = get_loader()
    docs_name = loader.get_docs_name()

    log.info("found %d papers", len(docs_name))

    if not docs_name:
        raise RuntimeError("No papers were found for ingestion.")

    collection_name, corpus_hash, next_index, added = _load_checkpoint(docs_name)
    vector_store = get_vector_store(create_if_missing=True, staging=True, collection_name=collection_name)

    if next_index:
        log.info("resuming staging collection %s at paper %d/%d", collection_name, next_index + 1, len(docs_name))

    try:
        with httpx.Client(follow_redirects=True) as http_client:
            for index, arxiv_id in enumerate(docs_name[next_index:], start=next_index + 1):
                dir_path = CORPUS_HTML_DIR if isinstance(loader, ArxivCorpusHtmlLoader) else SAMPLE_HTML_DIR
                loaded_paper = load_paper(arxiv_id, http_client, dir_path)

                if loaded_paper.note == NO_HTML_NOTE:
                    log.error("arXiv %s: %s", arxiv_id, loaded_paper.note)

                documents = convert_loaded_paper_to_documents(loaded_paper)
                log.info("converted arXiv %s into %d documents", arxiv_id, len(documents))

                if documents:
                    log.info("[%d/%d] embedding %s (%d documents)", index, len(docs_name), arxiv_id, len(documents))
                    _add_documents_with_retry(vector_store, documents)
                    added += len(documents)
                    log.info("[%d/%d] stored %s", index, len(docs_name), arxiv_id)

                _write_checkpoint(collection_name, corpus_hash, index, added)

        if added == 0:
            raise RuntimeError("No documents were prepared for ingestion.")

        vector_store.activate()
        INGESTION_CHECKPOINT_FILE.unlink(missing_ok=True)
    except Exception:
        log.exception("staging vector store preserved for resume after adding %d documents", added)
        vector_store.close()
        raise

    log.info("activated complete vector store")
    log.info("done. stored %d documents from %d papers", added, len(docs_name))
    return vector_store


def main() -> int:
    vector_store = None
    try:
        vector_store = ingest_documents()
    except RuntimeError as error:
        log.error("ingestion stopped: %s", error)
        return 1
    except Exception:
        log.exception("ingestion stopped unexpectedly")
        return 1
    finally:
        if vector_store is not None:
            vector_store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
