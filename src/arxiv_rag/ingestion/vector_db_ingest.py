import os
from abc import ABC, abstractmethod
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma  # pyright: ignore[reportMissingImports]
from langchain_core.documents import Document  # pyright: ignore[reportMissingImports]
from langchain_ollama import OllamaEmbeddings  # pyright: ignore[reportMissingImports]

CHROMA_DIRECTORY = Path(__file__).parents[3] / "chroma_db"
CHROMA_DATABASE_FILE = CHROMA_DIRECTORY / "chroma.sqlite3"
CHROMA_COLLECTION_NAME = "arxiv_papers"


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
    def reset(self) -> None:
        """Remove every document so the full corpus can be rebuilt safely."""
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

    def reset(self) -> None:
        self._db.reset_collection()


def get_vector_store(create_if_missing: bool = False) -> VectorStore:
    if not create_if_missing and not CHROMA_DATABASE_FILE.exists():
        raise FileNotFoundError(
            f"No Chroma database found at {CHROMA_DIRECTORY}. Run ingestion first."
        )

    load_dotenv()
    base_url = os.environ.get("OLLAMA_BASE_URL")
    if not base_url:
        raise RuntimeError("OLLAMA_BASE_URL is not set. Add it to your .env file.")

    embeddings = OllamaEmbeddings(
        model="qwen3-embedding:4b",
        base_url=base_url,
    )
    return ChromaStore(embeddings)
