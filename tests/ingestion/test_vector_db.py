from langchain_core.embeddings import (  # pyright: ignore[reportMissingImports]
    DeterministicFakeEmbedding,
)

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


def test_fresh_rebuild_removes_old_records_before_adding_the_corpus(tmp_path):
    store = _store(tmp_path)
    documents = _documents()
    old_document = documents[0].model_copy(update={"id": "old-record"})
    store.add([old_document])

    store.reset()
    store.add(documents)

    assert store.get(["old-record"]) == []
    assert [document.id for document in store.get([documents[0].id])] == [
        documents[0].id
    ]
