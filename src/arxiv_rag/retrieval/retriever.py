import json
from dataclasses import dataclass

from langchain_core.documents import Document  # pyright: ignore[reportMissingImports]
from langsmith import traceable

from arxiv_rag.ingestion.vector_db_ingest import VectorStore, get_vector_store
from arxiv_rag.logging import get_logger

DEFAULT_TOP_K = 8

log = get_logger(__name__)


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


@dataclass(frozen=True)
class BuiltContext:
    context: RetrievalContext
    passages_by_id: dict[str, str]


def get_source_passage_for_a_document(document: Document) -> list[SourcePassage]:
    """Read the original source passages stored during ingestion."""
    value = document.metadata.get("source_passages")

    try:
        if not isinstance(value, str):
            raise TypeError("source_passages must be JSON text")
        items = json.loads(value)
        if not isinstance(items, list):
            raise TypeError("source_passages must contain a list")

        passages = []
        for item in items:
            if not isinstance(item, dict):
                raise TypeError("each source passage must be an object")
            text = item["text"]
            location = item["location"]
            section_path = item["section_path"]

            if not isinstance(text, str) or not text.strip():
                raise TypeError("passage text must be non-empty text")
            if not isinstance(location, str) or not location.strip():
                raise TypeError("passage location must be non-empty text")
            if not isinstance(section_path, list) or not all(isinstance(section, str) for section in section_path):
                raise TypeError("section_path must be a list of text values")

            passages.append(SourcePassage(text=text, location=location, section_path=section_path))

        return passages

    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError("Retrieved source-passage metadata is invalid.") from error


def _build_section_breadcrumbs(section_path: list[str]) -> str:
    if not section_path:
        return "Unsectioned"
    return " > ".join(section_path)


@traceable(
    name="build_context",
    process_inputs=lambda inputs: {},
    process_outputs=lambda outputs: {
        "passages": outputs.context.text,
        "citation_ids": list(outputs.context.citations),
    },
)
def build_context_with_details(results: list[tuple[Document, float]]) -> BuiltContext:
    """Expand ranked Documents into deduplicated, exactly citable passages."""

    context_passages = []
    citations = {}
    passages_by_id = {}
    seen_passages = set()
    valid_documents = 0

    for document, _score in results:
        try:
            arxiv_id = document.metadata.get("arxiv_id")
            if not isinstance(arxiv_id, str) or not arxiv_id.strip():
                raise RuntimeError("Retrieved arXiv metadata is invalid.")
            source_passage_doc = get_source_passage_for_a_document(document)
        except RuntimeError:
            log.exception("skipping malformed retrieved document %s", document.id)
            continue

        valid_documents += 1
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

            passages_by_id[citation_id] = passage.text

            context_passages.append(
                f"[{citation_id}]\n"
                f"Section: {section_bread_crumbs}\n"
                f"Text: {passage.text}"
            )

    if results and valid_documents == 0:
        raise RuntimeError("Retrieved evidence is invalid.")

    retrieval_context = RetrievalContext(
        text="\n\n---\n\n".join(context_passages),
        citations=citations,
    )

    if not citations:
        log.warning("no citable passages found in %d retrieved documents", len(results))
    else:
        log.info("built context: %d passages from %d documents", len(citations), len(results))

    return BuiltContext(
        context=retrieval_context,
        passages_by_id=passages_by_id,
    )


def build_context(results: list[tuple[Document, float]]) -> RetrievalContext:
    """Expand ranked Documents into citable passages, without the passage text."""
    return build_context_with_details(results).context


class PaperRetriever:

    def __init__(self, vector_store: VectorStore | None = None, top_k: int = DEFAULT_TOP_K, ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self._vector_store = vector_store or get_vector_store()
        self._top_k = top_k

    @traceable(
        name="retrieve",
        process_outputs=lambda outputs: {"num_documents": len(outputs)},
    )
    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        """Return ranked retrieval Documents for a non-empty question."""
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        try:
            results = self._vector_store.similarity_search_with_score(question, k=self._top_k)
        except Exception as error:
            log.exception("evidence retrieval failed (top_k=%d)", self._top_k)
            raise RuntimeError("Could not retrieve evidence.") from error

        log.info("retrieved %d documents (top_k=%d)", len(results), self._top_k)

        return results

    def retrieve_context_with_details(self, question: str) -> BuiltContext:
        """Retrieve context and keep the passage text callers need to show evidence."""
        retrieved_docs_with_rank = self.retrieve(question)
        return build_context_with_details(retrieved_docs_with_rank)

    def retrieve_context(self, question: str) -> RetrievalContext:
        """Retrieve and expand a question into exact source-passage context."""
        return self.retrieve_context_with_details(question).context


if __name__ == "__main__":
    retriever = PaperRetriever()
    question = "Explain what is decoder?"
    context = retriever.retrieve_context(question)
    print(f"Context text:\n{context.text}\n")
    print("Citations:")
    for citation_id, citation in context.citations.items():
        print(f"{citation_id}: {citation.label} -> {citation.url}")
