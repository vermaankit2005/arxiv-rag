import json
from typing import Any, cast

import pytest
from langchain_core.documents import Document

from arxiv_rag.ingestion import ingestion_pipeline


@pytest.fixture(autouse=True)
def isolated_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(ingestion_pipeline, "INGESTION_CHECKPOINT_FILE", tmp_path / "ingestion_checkpoint.json")
    monkeypatch.setattr(ingestion_pipeline.time, "sleep", lambda seconds: None)


class FakeHttpClient:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class FakeLoader:
    def get_docs_name(self):
        return ["paper-1", "paper-2"]


class FakeLoadedPaper:
    def __init__(self, arxiv_id):
        self.arxiv_id = arxiv_id
        self.note = None


class RecordingStore:
    def __init__(self, fail_on_add=None, fail_on_delete=False):
        self.fail_on_add = fail_on_add
        self.fail_on_delete = fail_on_delete
        self.added = []
        self.activated = False
        self.deleted = False
        self.closed = False

    def add(self, documents):
        if len(self.added) + 1 == self.fail_on_add:
            raise RuntimeError("embedding failed")
        self.added.append(documents)
        return []

    def activate(self):
        self.activated = True

    def delete(self):
        self.deleted = True
        if self.fail_on_delete:
            raise RuntimeError("cleanup failed")

    def close(self):
        self.closed = True


def _patch_loading(monkeypatch):
    monkeypatch.setattr(ingestion_pipeline, "get_loader", lambda: FakeLoader())
    monkeypatch.setattr(ingestion_pipeline.httpx, "Client", FakeHttpClient)
    monkeypatch.setattr(
        ingestion_pipeline,
        "load_paper",
        lambda arxiv_id, client, html_dir: FakeLoadedPaper(arxiv_id),
    )
    monkeypatch.setattr(
        ingestion_pipeline,
        "convert_loaded_paper_to_documents",
        lambda loaded_paper: [f"document-for-{loaded_paper.arxiv_id}"],
    )


def test_parse_failure_deletes_staging_and_stops_before_later_papers(monkeypatch):
    _patch_loading(monkeypatch)
    store = RecordingStore()
    loaded = []

    def load(arxiv_id, client, html_dir):
        assert html_dir == ingestion_pipeline.SAMPLE_HTML_DIR
        loaded.append(arxiv_id)
        raise RuntimeError("parse failed")

    monkeypatch.setattr(ingestion_pipeline, "load_paper", load)
    monkeypatch.setattr(ingestion_pipeline, "get_vector_store", lambda **kwargs: store)

    try:
        ingestion_pipeline.ingest_documents()
    except RuntimeError as error:
        assert str(error) == "parse failed"
    else:
        raise AssertionError("Expected parsing to fail")

    assert loaded == ["paper-1"]
    assert store.added == []
    assert store.activated is False
    assert store.deleted is False
    assert store.closed is True


def test_embedding_failure_preserves_staging_and_leaves_active_collection_untouched(monkeypatch):
    _patch_loading(monkeypatch)
    store = RecordingStore(fail_on_add=2)
    monkeypatch.setattr(ingestion_pipeline, "get_vector_store", lambda **kwargs: store)

    try:
        ingestion_pipeline.ingest_documents()
    except RuntimeError as error:
        assert str(error) == "embedding failed"
    else:
        raise AssertionError("Expected embedding to fail")

    assert store.added == [["document-for-paper-1"]]
    assert store.activated is False
    assert store.deleted is False
    assert store.closed is True
    checkpoint = json.loads(ingestion_pipeline.INGESTION_CHECKPOINT_FILE.read_text(encoding="utf-8"))
    assert checkpoint["next_index"] == 1


def test_empty_input_stops_before_creating_staging_collection(monkeypatch):
    class EmptyLoader:
        def get_docs_name(self):
            return []

    monkeypatch.setattr(ingestion_pipeline, "get_loader", lambda: EmptyLoader())
    monkeypatch.setattr(
        ingestion_pipeline,
        "get_vector_store",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not create staging collection")),
    )

    try:
        ingestion_pipeline.ingest_documents()
    except RuntimeError as error:
        assert str(error) == "No papers were found for ingestion."
    else:
        raise AssertionError("Expected empty ingestion to fail")


def test_zero_prepared_documents_deletes_staging_collection(monkeypatch):
    _patch_loading(monkeypatch)
    store = RecordingStore()
    monkeypatch.setattr(ingestion_pipeline, "convert_loaded_paper_to_documents", lambda paper: [])
    monkeypatch.setattr(ingestion_pipeline, "get_vector_store", lambda **kwargs: store)

    try:
        ingestion_pipeline.ingest_documents()
    except RuntimeError as error:
        assert str(error) == "No documents were prepared for ingestion."
    else:
        raise AssertionError("Expected document-free ingestion to fail")

    assert store.added == []
    assert store.activated is False
    assert store.deleted is False
    assert store.closed is True


def test_add_retries_five_times_and_respects_retry_after(monkeypatch):
    class RetryableError(RuntimeError):
        error = {"retry_after": 60}

    class RetryStore(RecordingStore):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def add(self, documents):
            self.attempts += 1
            if self.attempts < 5:
                raise RetryableError("temporary failure")
            return super().add(documents)

    delays = []
    store = RetryStore()
    monkeypatch.setattr(ingestion_pipeline.time, "sleep", delays.append)

    ingestion_pipeline._add_documents_with_retry(cast(Any, store), [Document(page_content="document")])

    assert store.attempts == 5
    assert delays == [60, 60, 60, 60]


def test_resume_uses_same_collection_and_starts_at_first_unfinished_paper(monkeypatch):
    _patch_loading(monkeypatch)
    first_store = RecordingStore(fail_on_add=2)
    second_store = RecordingStore()
    stores = iter([first_store, second_store])
    requested_collections = []

    def get_store(**kwargs):
        requested_collections.append(kwargs["collection_name"])
        return next(stores)

    monkeypatch.setattr(ingestion_pipeline, "get_vector_store", get_store)

    with pytest.raises(RuntimeError, match="embedding failed"):
        ingestion_pipeline.ingest_documents()

    returned_store = ingestion_pipeline.ingest_documents()

    assert returned_store is second_store
    assert requested_collections[0] == requested_collections[1]
    assert second_store.added == [["document-for-paper-2"]]
    assert second_store.activated is True
    assert not ingestion_pipeline.INGESTION_CHECKPOINT_FILE.exists()


def test_ingestion_main_returns_failure_status(monkeypatch):
    monkeypatch.setattr(
        ingestion_pipeline,
        "ingest_documents",
        lambda: (_ for _ in ()).throw(RuntimeError("failed")),
    )

    assert ingestion_pipeline.main() == 1


def test_ingestion_main_closes_the_store(monkeypatch):
    store = RecordingStore()
    monkeypatch.setattr(ingestion_pipeline, "ingest_documents", lambda: store)

    assert ingestion_pipeline.main() == 0
    assert store.closed is True


def test_complete_staging_collection_is_activated_after_all_writes(monkeypatch):
    _patch_loading(monkeypatch)
    store = RecordingStore()
    factory_calls = []

    def get_store(**kwargs):
        factory_calls.append(kwargs)
        return store

    monkeypatch.setattr(ingestion_pipeline, "get_vector_store", get_store)

    returned_store = ingestion_pipeline.ingest_documents()

    assert returned_store is store
    assert store.added == [["document-for-paper-1"], ["document-for-paper-2"]]
    assert store.activated is True
    assert len(factory_calls) == 1
    assert factory_calls[0]["create_if_missing"] is True
    assert factory_calls[0]["staging"] is True
    assert factory_calls[0]["collection_name"].startswith("arxiv_papers_")
    assert not ingestion_pipeline.INGESTION_CHECKPOINT_FILE.exists()
