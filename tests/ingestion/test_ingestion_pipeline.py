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
    def __init__(self, fail_on_add=None):
        self.fail_on_add = fail_on_add
        self.added = []
        self.deleted = False

    def add(self, documents):
        if len(self.added) + 1 == self.fail_on_add:
            raise RuntimeError("embedding failed")
        self.added.append(documents)
        return []

    def delete(self):
        self.deleted = True


def _patch_loading(monkeypatch):
    monkeypatch.setattr(ingestion_pipeline, "ArxivSampleHTMLLoader", FakeLoader)
    monkeypatch.setattr(ingestion_pipeline.httpx, "Client", FakeHttpClient)
    monkeypatch.setattr(ingestion_pipeline, "load_paper", lambda arxiv_id, client: arxiv_id)
    monkeypatch.setattr(
        ingestion_pipeline,
        "convert_loaded_paper_to_documents",
        lambda loaded_paper: [f"document-for-{loaded_paper}"],
    )


def test_parse_failure_leaves_active_collection_untouched(monkeypatch):
    _patch_loading(monkeypatch)
    monkeypatch.setattr(
        ingestion_pipeline,
        "load_paper",
        lambda arxiv_id, client: (_ for _ in ()).throw(RuntimeError("parse failed")),
    )
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
        assert str(error) == "parse failed"
    else:
        raise AssertionError("Expected parsing to fail")


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
