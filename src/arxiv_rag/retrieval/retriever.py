import json
from dataclasses import dataclass

from langchain_core.documents import Document  # pyright: ignore[reportMissingImports]

from arxiv_rag.ingestion.vector_db_ingest import VectorStore, get_vector_store

DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class SourcePassage:
    text: str
    location: str
    section_path: list[str]


@dataclass(frozen=True)
class Citation:
    label: str
    url: str


@dataclass(frozen=True)
class RetrievalContext:
    text: str
    citations: dict[str, Citation]


def get_source_passage_for_a_document(document: Document) -> list[SourcePassage]:
    """Read the original source passages stored during ingestion."""
    value = document.metadata.get("source_passages")

    try:
        items = json.loads(value)
        return [
            SourcePassage(
                text=item["text"],
                location=item["location"],
                section_path=item["section_path"],
            )
            for item in items
        ]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError("Retrieved source-passage metadata is invalid.") from error


def _build_section_breadcrumbs(section_path: list[str]) -> str:
    if not section_path:
        return "Unsectioned"
    return " > ".join(section_path)


def build_context(results: list[tuple[Document, float]]) -> RetrievalContext:
    """Expand ranked Documents into deduplicated, exactly citable passages."""
    context_passages = []
    citations = {}
    seen_passages = set()

    for document, _score in results:
        arxiv_id = document.metadata.get("arxiv_id")

        source_passage_doc = get_source_passage_for_a_document(document)

        for passage in source_passage_doc:

            source_key = (arxiv_id, passage.location, passage.text)
            if source_key in seen_passages:
                continue
            # this is check is required to avoid duplicate passages in the context, especially when multiple documents have overlapping passages
            seen_passages.add(source_key)

            section_bread_crumbs = _build_section_breadcrumbs(passage.section_path)

            url = f"https://arxiv.org/html/{arxiv_id}{passage.location}"

            citation_id = f"P{len(citations) + 1}"
            citations[citation_id] = Citation(label=f"{arxiv_id} — {section_bread_crumbs}", url=url)

            context_passages.append(
                f"[{citation_id}]\n"
                f"Section: {section_bread_crumbs}\n"
                f"Text: {passage.text}"
            )

    return RetrievalContext(
        text="\n\n---\n\n".join(context_passages),
        citations=citations,
    )


class PaperRetriever:

    def __init__(self, vector_store: VectorStore | None = None, top_k: int = DEFAULT_TOP_K, ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self._vector_store = vector_store or get_vector_store()
        self._top_k = top_k

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        """Return ranked retrieval Documents for a non-empty question."""
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        return self._vector_store.similarity_search_with_score(
            question,
            k=self._top_k,
        )

    def retrieve_context(self, question: str) -> RetrievalContext:
        """Retrieve and expand a question into exact source-passage context."""
        retrieved_docs_with_rank = self.retrieve(question)
        return build_context(retrieved_docs_with_rank)


if __name__ == "__main__":
    retriever = PaperRetriever()
    question = "Explain what is decoder?"
    context = retriever.retrieve_context(question)
    print(f"Context text:\n{context.text}\n")
    print("Citations:")
    for citation_id, citation in context.citations.items():
        print(f"{citation_id}: {citation.label} -> {citation.url}")
