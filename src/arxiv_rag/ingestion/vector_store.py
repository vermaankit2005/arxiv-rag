from abc import ABC, abstractmethod
from datetime import UTC, datetime

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def versioned_collection_name(base_name: str, timestamp: datetime | None = None) -> str:
    """Return a readable collection name using a UTC creation timestamp."""
    timestamp = timestamp or datetime.now(UTC)
    return f"{base_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"


def close_embedding_client(embeddings: Embeddings) -> None:
    """Close the sync client created internally by OllamaEmbeddings."""
    client = getattr(embeddings, "_client", None)
    if client is not None:
        client.close()


class VectorStore(ABC):
    """Storage contract used by ingestion and retrieval."""

    @abstractmethod
    def add(self, documents: list[Document]) -> list[str]:
        pass

    @abstractmethod
    def get(self, ids: list[str]) -> list[Document]:
        pass

    @abstractmethod
    def similarity_search_with_score(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        pass

    @abstractmethod
    def activate(self) -> None:
        """Make this store's completed corpus available to retrieval."""

    @abstractmethod
    def delete(self) -> None:
        """Delete this store's corpus."""

    @abstractmethod
    def close(self) -> None:
        """Release resources held by this store, if any."""
