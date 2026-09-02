from arxiv_rag.ingestion import ingestion_pipeline


class FakeHttpClient:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class FakeLoader:
    def get_docs_name(self):
        return ["paper-1.html", "paper-2.html"]


class RecordingStore:
    def __init__(self, fail_on_add=None, fail_on_delete=False):
        self.fail_on_add = fail_on_add
        self.fail_on_delete = fail_on_delete
        self.added = []
        self.deleted = False

    def add(self, documents):
        if len(self.added) + 1 == self.fail_on_add:
            raise RuntimeError("embedding failed")
        self.added.append(documents)
        return []

    def delete(self):
        self.deleted = True
        if self.fail_on_delete:
            raise RuntimeError("cleanup failed")


def _patch_loading(monkeypatch):
    monkeypatch.setattr(ingestion_pipeline, "ArxivSampleHTMLLoader", FakeLoader)
    monkeypatch.setattr(ingestion_pipeline.httpx, "Client", FakeHttpClient)
    monkeypatch.setattr(ingestion_pipeline, "load_paper", lambda arxiv_id, client: arxiv_id)
    monkeypatch.setattr(
        ingestion_pipeline,
        "convert_loaded_paper_to_documents",
        lambda loaded_paper: [f"document-for-{loaded_paper}"],
    )


def test_parse_failure_processes_later_papers_but_does_not_stage_partial_corpus(monkeypatch):
    _patch_loading(monkeypatch)
    loaded = []

    def load(arxiv_id, client):
        loaded.append(arxiv_id)
        if arxiv_id == "paper-1":
            raise RuntimeError("parse failed")
        return arxiv_id

    monkeypatch.setattr(ingestion_pipeline, "load_paper", load)
    monkeypatch.setattr(
        ingestion_pipeline,
        "get_vector_store",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not create staging collection")),
    )
    monkeypatch.setattr(
        ingestion_pipeline,
        "activate_collection",
        lambda name: (_ for _ in ()).throw(AssertionError("must not activate collection")),
    )

    try:
        ingestion_pipeline.ingest_documents()
    except RuntimeError as error:
        assert "paper-1" in str(error)
    else:
        raise AssertionError("Expected parsing to fail")

    assert loaded == ["paper-1", "paper-2"]


def test_embedding_failure_deletes_staging_and_leaves_active_collection_untouched(monkeypatch):
    _patch_loading(monkeypatch)
    store = RecordingStore(fail_on_add=2)
    monkeypatch.setattr(ingestion_pipeline, "get_vector_store", lambda **kwargs: store)
    monkeypatch.setattr(
        ingestion_pipeline,
        "activate_collection",
        lambda name: (_ for _ in ()).throw(AssertionError("must not activate collection")),
    )

    try:
        ingestion_pipeline.ingest_documents()
    except RuntimeError as error:
        assert str(error) == "embedding failed"
    else:
        raise AssertionError("Expected embedding to fail")

    assert store.added == [["document-for-paper-1"]]
    assert store.deleted is True


def test_empty_input_stops_before_creating_staging_collection(monkeypatch):
    class EmptyLoader:
        def get_docs_name(self):
            return []

    monkeypatch.setattr(ingestion_pipeline, "ArxivSampleHTMLLoader", EmptyLoader)
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


def test_zero_prepared_documents_stops_before_creating_staging_collection(monkeypatch):
    _patch_loading(monkeypatch)
    monkeypatch.setattr(ingestion_pipeline, "convert_loaded_paper_to_documents", lambda paper: [])
    monkeypatch.setattr(
        ingestion_pipeline,
        "get_vector_store",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not create staging collection")),
    )

    try:
        ingestion_pipeline.ingest_documents()
    except RuntimeError as error:
        assert str(error) == "No documents were prepared for ingestion."
    else:
        raise AssertionError("Expected document-free ingestion to fail")


def test_cleanup_failure_preserves_the_embedding_failure(monkeypatch):
    _patch_loading(monkeypatch)
    store = RecordingStore(fail_on_add=1, fail_on_delete=True)
    monkeypatch.setattr(ingestion_pipeline, "get_vector_store", lambda **kwargs: store)

    try:
        ingestion_pipeline.ingest_documents()
    except RuntimeError as error:
        assert str(error) == "embedding failed"
    else:
        raise AssertionError("Expected embedding to fail")

    assert store.deleted is True


def test_ingestion_main_returns_failure_status(monkeypatch):
    monkeypatch.setattr(
        ingestion_pipeline,
        "ingest_documents",
        lambda: (_ for _ in ()).throw(RuntimeError("failed")),
    )

    assert ingestion_pipeline.main() == 1


def test_complete_staging_collection_is_activated_after_all_writes(monkeypatch):
    _patch_loading(monkeypatch)
    store = RecordingStore()
    requested_collections = []
    activated_collections = []

    def get_store(**kwargs):
        requested_collections.append(kwargs["collection_name"])
        return store

    monkeypatch.setattr(ingestion_pipeline, "get_vector_store", get_store)
    monkeypatch.setattr(ingestion_pipeline, "activate_collection", activated_collections.append)

    returned_store = ingestion_pipeline.ingest_documents()

    assert returned_store is store
    assert store.added == [["document-for-paper-1"], ["document-for-paper-2"]]
    assert activated_collections == requested_collections
    assert requested_collections[0].startswith("arxiv_papers_staging_")
