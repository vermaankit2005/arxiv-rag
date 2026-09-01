from abc import ABC, abstractmethod
from pathlib import Path
from uuid import uuid4

from langchain_chroma import Chroma  # pyright: ignore[reportMissingImports]
from langchain_core.documents import Document  # pyright: ignore[reportMissingImports]
from langchain_ollama import OllamaEmbeddings  # pyright: ignore[reportMissingImports]

from arxiv_rag.ollama_config import get_ollama_connection

CHROMA_DIRECTORY = Path(__file__).parents[3] / "chroma_db"
CHROMA_DATABASE_FILE = CHROMA_DIRECTORY / "chroma.sqlite3"
CHROMA_COLLECTION_NAME = "arxiv_papers"
ACTIVE_COLLECTION_FILE = CHROMA_DIRECTORY / "active_collection.txt"


class VectorStore(ABC):
    @abstractmethod
    def add(self, documents: list[Document]) -> list[str]:
        ...

    @abstractmethod
    def get(self, ids: list[str]) -> list[Document]:
        ...

    @abstractmethod
    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> list[tuple[Document, float]]:
        ...

    @abstractmethod
    def delete(self) -> None:
        """Delete this collection."""
        ...


class ChromaStore(VectorStore):
    def __init__(
        self,
        embeddings,
        persist_directory: Path = CHROMA_DIRECTORY,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ) -> None:
        self._db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_directory),
        )

    def add(self, documents: list[Document]) -> list[str]:
        return self._db.add_documents(documents)

    def get(self, ids: list[str]) -> list[Document]:
        return self._db.get_by_ids(ids)

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> list[tuple[Document, float]]:
        return self._db.similarity_search_with_score(query, k=k)

    def delete(self) -> None:
        self._db.delete_collection()


def get_active_collection_name() -> str:
    if not ACTIVE_COLLECTION_FILE.exists():
        return CHROMA_COLLECTION_NAME

    collection_name = ACTIVE_COLLECTION_FILE.read_text(encoding="utf-8").strip()
    if not collection_name:
        raise RuntimeError(f"Active Chroma collection file is empty: {ACTIVE_COLLECTION_FILE}")
    return collection_name


def activate_collection(collection_name: str) -> None:
    """Atomically point new retrievers at a completely built collection."""
    CHROMA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    temporary_file = ACTIVE_COLLECTION_FILE.with_name(f".{ACTIVE_COLLECTION_FILE.name}.{uuid4().hex}")
    temporary_file.write_text(collection_name, encoding="utf-8")
    temporary_file.replace(ACTIVE_COLLECTION_FILE)


def get_vector_store(create_if_missing: bool = False, collection_name: str | None = None) -> VectorStore:
    if not create_if_missing and not CHROMA_DATABASE_FILE.exists():
        raise FileNotFoundError(
            f"No Chroma database found at {CHROMA_DIRECTORY}. Run ingestion first."
        )

    base_url, headers = get_ollama_connection()
    embeddings = OllamaEmbeddings(
        model="qwen3-embedding:4b",
        base_url=base_url,
        client_kwargs={"headers": headers},
    )
    return ChromaStore(embeddings, collection_name=collection_name or get_active_collection_name())
