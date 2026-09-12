from typing import Any, cast

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from arxiv_rag.ingestion import weaviate_vector_store
from arxiv_rag.ingestion.weaviate_vector_store import WeaviateStore


class RecordingCollections:
    def __init__(self):
        self.deleted_collection = None

    def delete(self, collection_name):
        self.deleted_collection = collection_name


class RecordingClient:
    def __init__(self):
        self.collections = RecordingCollections()
        self.closed = False

    def close(self):
        self.closed = True


class RecordingEmbeddingClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class RecordingEmbeddings:
    def __init__(self):
        self._client = RecordingEmbeddingClient()


class RecordingDatabase:
    def __init__(self):
        self.added = None
        self.added_ids = None
        self.requested_ids = None
        self.search = None
        self.deleted_ids = None

    def add_documents(self, documents, ids=None):
        self.added = documents
        self.added_ids = ids
        return ids

    def get_by_ids(self, ids):
        self.requested_ids = ids
        return ["stored-document"]

    def similarity_search_with_score(self, query, k, alpha):
        self.search = (query, k, alpha)
        return [("matching-document", 0.25)]

    def delete(self, ids):
        self.deleted_ids = ids


def _store(database=None, collection_name="test_collection", active_collection_file=None, client=None, embeddings=None):
    store = cast(Any, object.__new__(WeaviateStore))
    store._collection_name = collection_name
    store._active_collection_file = active_collection_file
    store._embeddings = embeddings or DeterministicFakeEmbedding(size=8)
    store._db = database or RecordingDatabase()
    store.client = client or RecordingClient()
    return store


def _replace_constructor(monkeypatch, opened_with):
    def initialize(self, embeddings, collection_name=weaviate_vector_store.WEAVIATE_COLLECTION_NAME, active_collection_file=weaviate_vector_store.ACTIVE_COLLECTION_FILE):
        opened_with.update({"embeddings": embeddings, "collection_name": collection_name})
        self._collection_name = collection_name
        self._active_collection_file = active_collection_file

    monkeypatch.setattr(WeaviateStore, "__init__", initialize)


def test_constructor_connects_to_cloud_and_builds_vector_store(monkeypatch, tmp_path):
    credentials = object()
    client = object()
    connected_with = {}
    vector_store_with = {}

    monkeypatch.setenv("WEAVIATE_URL", "https://example.weaviate.network")
    monkeypatch.setenv("WEAVIATE_API_KEY", "secret-key")
    monkeypatch.setattr(weaviate_vector_store.Auth, "api_key", lambda value: credentials if value == "secret-key" else None)

    def connect(**kwargs):
        connected_with.update(kwargs)
        return client

    class FakeVectorStore:
        def __init__(self, **kwargs):
            vector_store_with.update(kwargs)

    monkeypatch.setattr(weaviate_vector_store.weaviate, "connect_to_weaviate_cloud", connect)
    monkeypatch.setattr(weaviate_vector_store, "WeaviateVectorStore", FakeVectorStore)

    active_file = tmp_path / "active_collection.txt"
    embeddings = DeterministicFakeEmbedding(size=8)
    store = WeaviateStore(embeddings=embeddings, collection_name="papers", active_collection_file=active_file)

    assert connected_with == {"cluster_url": "https://example.weaviate.network", "auth_credentials": credentials}
    assert vector_store_with == {"client": client, "index_name": "papers", "text_key": "text", "embedding": embeddings}
    assert store.client is client
    assert store._active_collection_file == active_file


def test_open_creates_a_timestamped_collection(monkeypatch):
    opened_with = {}
    _replace_constructor(monkeypatch, opened_with)
    monkeypatch.setattr(weaviate_vector_store.vector_store_contract, "versioned_collection_name", lambda base_name: f"{base_name}_20260912_154511")

    embeddings = DeterministicFakeEmbedding(size=8)
    WeaviateStore.open(embeddings=embeddings, create_if_missing=True, staging=True)

    assert opened_with["embeddings"] is embeddings
    assert opened_with["collection_name"] == "arxiv_papers_20260912_154511"


def test_open_rejects_staging_collection_that_cannot_be_created():
    try:
        WeaviateStore.open(embeddings=DeterministicFakeEmbedding(size=8), create_if_missing=False, staging=True)
    except ValueError as error:
        assert str(error) == "A staging vector store must be created before it can be used."
    else:
        raise AssertionError("Expected staging open to fail without create_if_missing")


def test_open_uses_the_active_collection(monkeypatch, tmp_path):
    active_file = tmp_path / "active_collection.txt"
    active_file.write_text("complete_collection", encoding="utf-8")
    opened_with = {}
    monkeypatch.setattr(weaviate_vector_store, "ACTIVE_COLLECTION_FILE", active_file)
    monkeypatch.setattr(weaviate_vector_store, "WEAVIATE_DIRECTORY", tmp_path)
    _replace_constructor(monkeypatch, opened_with)

    WeaviateStore.open(embeddings=DeterministicFakeEmbedding(size=8))

    assert opened_with["collection_name"] == "complete_collection"


def test_empty_active_collection_file_is_rejected(monkeypatch, tmp_path):
    active_file = tmp_path / "active_collection.txt"
    active_file.write_text("  ", encoding="utf-8")
    monkeypatch.setattr(weaviate_vector_store, "ACTIVE_COLLECTION_FILE", active_file)

    try:
        WeaviateStore._get_active_collection_name()
    except RuntimeError as error:
        assert str(active_file) in str(error)
    else:
        raise AssertionError("Expected an empty active collection file to fail")


def test_add_returns_ids_from_weaviate():
    database = RecordingDatabase()
    store = _store(database)
    documents = [Document(id="deterministic-id", page_content="document")]

    ids = store.add(documents)

    assert database.added == documents
    assert database.added_ids == ["deterministic-id"]
    assert ids == ["deterministic-id"]


def test_get_delegates_to_weaviate():
    database = RecordingDatabase()
    store = _store(database)

    documents = store.get(["document-id"])

    assert database.requested_ids == ["document-id"]
    assert documents == ["stored-document"]


def test_similarity_search_delegates_query_and_limit():
    database = RecordingDatabase()
    store = _store(database)

    matches = store.similarity_search_with_score("attention", k=7)

    assert database.search == ("attention", 7, 0.5)
    assert matches == [("matching-document", 0.25)]


def test_delete_removes_the_weaviate_collection():
    client = RecordingClient()
    store = _store(collection_name="incomplete_collection", client=client)

    store.delete()

    assert client.collections.deleted_collection == "incomplete_collection"


def test_close_closes_weaviate_and_embedding_clients():
    client = RecordingClient()
    embeddings = RecordingEmbeddings()
    store = _store(client=client, embeddings=embeddings)

    store.close()

    assert client.closed is True
    assert embeddings._client.closed is True


def test_activate_atomically_replaces_active_pointer(tmp_path):
    active_file = tmp_path / "state" / "active_collection.txt"
    store = _store(collection_name="complete_collection", active_collection_file=active_file)

    store.activate()

    assert active_file.read_text(encoding="utf-8") == "complete_collection"
    assert list(active_file.parent.glob(".active_collection.txt.*")) == []
