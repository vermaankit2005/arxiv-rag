from langchain_core.embeddings import (  # pyright: ignore[reportMissingImports]
    DeterministicFakeEmbedding,
)

from arxiv_rag.ingestion import vector_db_ingest
from arxiv_rag.ingestion.documents import convert_loaded_paper_to_documents
from arxiv_rag.ingestion.vector_db_ingest import ChromaStore
from arxiv_rag.loading.models import LoadedPaper, Passage


def _documents():
    paper = LoadedPaper(
        arxiv_id="test-paper",
        sections=[],
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
    stored = store.get([documents[0].id])

    assert len(stored) == 1
    assert stored[0].id == documents[0].id
    assert stored[0].page_content == documents[0].page_content
    assert stored[0].metadata == documents[0].metadata


def test_activating_collection_replaces_active_pointer(monkeypatch, tmp_path):
    active_collection_file = tmp_path / "active_collection.txt"
    active_collection_file.write_text("old_collection", encoding="utf-8")
    monkeypatch.setattr(vector_db_ingest, "CHROMA_DIRECTORY", tmp_path)
    monkeypatch.setattr(vector_db_ingest, "ACTIVE_COLLECTION_FILE", active_collection_file)

    vector_db_ingest.activate_collection("complete_collection")

    assert vector_db_ingest.get_active_collection_name() == "complete_collection"
    assert list(tmp_path.glob(".active_collection.txt.*")) == []


def test_deleted_staging_collection_can_be_recreated_without_partial_records(tmp_path):
    store = _store(tmp_path)
    documents = _documents()
    store.add([documents[0].model_copy(update={"id": "partial-record"})])

    store.delete()
    replacement = _store(tmp_path)

    assert replacement.get(["partial-record"]) == []
