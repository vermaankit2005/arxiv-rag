from datetime import UTC, datetime

from langchain_core.embeddings import (  # pyright: ignore[reportMissingImports]
    DeterministicFakeEmbedding,
)

from arxiv_rag.ingestion import vector_db_ingest
from arxiv_rag.ingestion.chroma_vector_store import ChromaStore
from arxiv_rag.ingestion.documents import convert_loaded_paper_to_documents
from arxiv_rag.ingestion.vector_store import versioned_collection_name
from arxiv_rag.loading.models import LoadedPaper, Passage


def _documents():
    paper = LoadedPaper(
        arxiv_id="test-paper",
        passages=[
            Passage(
                order=1,
                text="Exact evidence from the source paper.",
                section="Results",
                section_path=["Results"],
                location="#S1.p1",
            )
        ],
    )
    return convert_loaded_paper_to_documents(paper)


def _store(tmp_path):
    return ChromaStore(
        embeddings=DeterministicFakeEmbedding(size=8),
        persist_directory=tmp_path / "chroma",
        collection_name="test_collection",
    )


def test_chroma_round_trip_preserves_document_content_and_metadata(tmp_path):
    store = _store(tmp_path)
    documents = _documents()

    store.add(documents)
    document_id = documents[0].id
    assert document_id is not None
    stored = store.get([document_id])

    assert len(stored) == 1
    assert stored[0].id == documents[0].id
    assert stored[0].page_content == documents[0].page_content
    assert stored[0].metadata == documents[0].metadata


def test_vector_store_factory_hides_the_configured_backend(monkeypatch):
    embeddings = object()
    expected_store = object()
    opened_with = {}

    def open_store(**kwargs):
        opened_with.update(kwargs)
        return expected_store

    monkeypatch.setattr(vector_db_ingest, "get_embeddings", lambda: embeddings)
    monkeypatch.setattr(vector_db_ingest, "application_config", lambda: {"loading": {"active": {"vector_store": "CHROMA"}}})
    monkeypatch.setattr(vector_db_ingest, "get_chroma_store", open_store)

    store = vector_db_ingest.get_vector_store(create_if_missing=True, staging=True)

    assert store is expected_store
    assert opened_with == {
        "embeddings": embeddings,
        "create_if_missing": True,
        "staging": True,
        "collection_name": None,
    }


def test_versioned_collection_name_contains_utc_date_and_time():
    timestamp = datetime(2026, 9, 12, 15, 45, 11, tzinfo=UTC)

    assert versioned_collection_name("arxiv_papers", timestamp) == "arxiv_papers_20260912_154511"


def test_activating_collection_replaces_active_pointer(tmp_path):
    active_collection_file = tmp_path / "active_collection.txt"
    active_collection_file.write_text("old_collection", encoding="utf-8")
    store = ChromaStore(
        embeddings=DeterministicFakeEmbedding(size=8),
        persist_directory=tmp_path / "chroma",
        collection_name="complete_collection",
        active_collection_file=active_collection_file,
    )

    store.activate()

    assert active_collection_file.read_text(encoding="utf-8") == "complete_collection"
    assert list(tmp_path.glob(".active_collection.txt.*")) == []


def test_deleted_staging_collection_can_be_recreated_without_partial_records(tmp_path):
    store = _store(tmp_path)
    documents = _documents()
    store.add([documents[0].model_copy(update={"id": "partial-record"})])

    store.delete()
    replacement = _store(tmp_path)

    assert replacement.get(["partial-record"]) == []
