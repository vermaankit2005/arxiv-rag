import os
from pathlib import Path
from uuid import uuid4

import weaviate
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_weaviate import WeaviateVectorStore
from weaviate.classes.init import Auth

from arxiv_rag.ingestion import vector_store as vector_store_contract
from arxiv_rag.logging import get_logger

WEAVIATE_DIRECTORY = Path(__file__).parents[3] / "weaviate_db"
WEAVIATE_COLLECTION_NAME = "arxiv_papers"
ACTIVE_COLLECTION_FILE = WEAVIATE_DIRECTORY / "active_collection.txt"

log = get_logger(__name__)

class WeaviateStore(vector_store_contract.VectorStore):
    def __init__(self, embeddings: Embeddings, collection_name: str = WEAVIATE_COLLECTION_NAME,
                 active_collection_file: Path = ACTIVE_COLLECTION_FILE) -> None:

        self._collection_name = collection_name
        self._active_collection_file = active_collection_file
        self._embeddings = embeddings
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.environ["WEAVIATE_URL"],
            auth_credentials=Auth.api_key(os.environ["WEAVIATE_API_KEY"]),
        )
        try:
            self._db = WeaviateVectorStore(
                client=self.client,
                index_name=collection_name,
                text_key="text",
                embedding=embeddings,
            )
        except Exception:
            try:
                self.client.close()
            finally:
                vector_store_contract.close_embedding_client(self._embeddings)
            raise

    @classmethod
    def open(cls, embeddings: Embeddings, create_if_missing: bool = False, staging: bool = False, collection_name: str | None = None) -> "WeaviateStore":

        if staging and not create_if_missing:
            raise ValueError("A staging vector store must be created before it can be used.")

        if not create_if_missing and not WEAVIATE_DIRECTORY.exists():
            raise FileNotFoundError(f"No Weaviate database found at {WEAVIATE_DIRECTORY}. Run ingestion first.")

        if collection_name is None:
            collection_name = vector_store_contract.versioned_collection_name(WEAVIATE_COLLECTION_NAME) if staging else cls._get_active_collection_name()
        log.info("opening Weaviate collection %s", collection_name)
        return WeaviateStore(embeddings=embeddings, collection_name=collection_name)

    @staticmethod
    def _get_active_collection_name() -> str:
        if not ACTIVE_COLLECTION_FILE.exists():
            return WEAVIATE_COLLECTION_NAME

        collection_name = ACTIVE_COLLECTION_FILE.read_text(encoding="utf-8").strip()

        if not collection_name:
            raise RuntimeError(f"Active Weaviate collection file is empty: {ACTIVE_COLLECTION_FILE}")
        return collection_name

    def add(self, documents: list[Document]) -> list[str]:
        ids = []
        for document in documents:
            if document.id is None:
                raise ValueError("Every document must have a deterministic ID before ingestion.")
            ids.append(document.id)
        return self._db.add_documents(documents, ids=ids)

    def get(self, ids: list[str]) -> list[Document]:
        # Implement the logic to retrieve documents by IDs from Weaviate
        return self._db.get_by_ids(ids)

    def similarity_search_with_score(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        return self._db.similarity_search_with_score(query, k=k, alpha=0.5)

    def activate(self) -> None:
        """Atomically point new retrievers at this completely built collection."""
        self._active_collection_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = self._active_collection_file.with_name(
            f".{self._active_collection_file.name}.{uuid4().hex}"
        )
        temporary_file.write_text(self._collection_name, encoding="utf-8")
        temporary_file.replace(self._active_collection_file)

    def delete(self) -> None:
        self.client.collections.delete(self._collection_name)

    def close(self) -> None:
        try:
            self.client.close()
        finally:
            vector_store_contract.close_embedding_client(self._embeddings)


def get_weaviate_store(embeddings: Embeddings, create_if_missing: bool = False, staging: bool = False, collection_name: str | None = None) -> vector_store_contract.VectorStore:
    return WeaviateStore.open(
        embeddings=embeddings,
        create_if_missing=create_if_missing,
        staging=staging,
        collection_name=collection_name,
    )
