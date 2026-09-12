from arxiv_rag.ingestion.chroma_vector_store import get_chroma_store
from arxiv_rag.ingestion.vector_store import VectorStore
from arxiv_rag.ingestion.weaviate_vector_store import get_weaviate_store
from arxiv_rag.model_provider import get_embeddings
from arxiv_rag.util import application_config


def get_vector_store(create_if_missing: bool = False, staging: bool = False, collection_name: str | None = None) -> VectorStore:
    """Return the configured store without exposing its backend to callers."""
    if application_config()["loading"]["active"]["vector_store"] != "WEAVIATE":
        return get_chroma_store(
            embeddings=get_embeddings(),
            create_if_missing=create_if_missing,
            staging=staging,
            collection_name=collection_name,
        )

    return get_weaviate_store(
        embeddings=get_embeddings(),
        create_if_missing=create_if_missing,
        staging=staging,
        collection_name=collection_name,
    )
