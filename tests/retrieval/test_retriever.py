import json

from langchain_core.documents import Document  # pyright: ignore[reportMissingImports]

from arxiv_rag import retrieval
from arxiv_rag.ingestion.vector_db_ingest import VectorStore


class RecordingStore(VectorStore):
    def __init__(self, results=None):
        self.results = results or []
        self.query = None
        self.k = None

    def add(self, documents):
        return [str(document.id) for document in documents]

    def get(self, ids):
        return []

    def similarity_search_with_score(self, query, k=4):
        self.query = query
        self.k = k
        return self.results

    def delete(self):
        return None


def _document(source_passages, arxiv_id="1706.03762v7"):
    return Document(
        page_content="Combined retrieval text",
        metadata={
            "arxiv_id": arxiv_id,
            "source_passages": json.dumps(source_passages),
        },
    )


def _passage(text, location="#S6.T2", section_path=None):
    return {
        "text": text,
        "location": location,
        "section_path": section_path or ["Results", "Machine Translation"],
        "kind": "prose",
    }


def test_retriever_uses_the_configured_top_k():
    store = RecordingStore()
    paper_retriever = retrieval.PaperRetriever(store, top_k=5)

    paper_retriever.retrieve_context("  How does attention work?  ")

    assert store.query == "How does attention work?"
    assert store.k == 5


def test_retrieve_context_with_details_keeps_the_passage_text():
    document = _document([_passage("The model achieved 28.4 BLEU.")])
    store = RecordingStore([(document, 0.5)])

    built = retrieval.PaperRetriever(store).retrieve_context_with_details("How good is it?")

    assert "[P1]" in built.context.text
    assert built.passages_by_id == {"P1": "The model achieved 28.4 BLEU."}


def test_build_context_pairs_each_passage_with_its_exact_anchor():
    document = _document(
        [
            _passage("The model achieved 28.4 BLEU."),
            _passage("Training was substantially faster.", "#S6.p3"),
        ]
    )

    context = retrieval.build_context([(document, 0.5)])

    assert "[P1]" in context.text
    assert "Text: The model achieved 28.4 BLEU." in context.text
    assert "URL:" not in context.text
    assert "[P2]" in context.text
    assert context.citations["P1"].url.endswith("#S6.T2")
    assert context.citations["P2"].url.endswith("#S6.p3")


def test_build_context_details_keeps_passage_ids_aligned_with_context():
    document = _document(
        [
            _passage("The model achieved 28.4 BLEU."),
            _passage("Training was substantially faster.", "#S6.p3"),
        ]
    )

    built = retrieval.build_context_with_details([(document, 0.5)])

    assert built.passages_by_id == {
        "P1": "The model achieved 28.4 BLEU.",
        "P2": "Training was substantially faster.",
    }
    assert "[P1]" in built.context.text
    assert "[P2]" in built.context.text


def test_build_context_deduplicates_exact_overlap():
    passage = _passage("Shared overlap passage.", "#S3.p2", ["Architecture"])

    context = retrieval.build_context(
        [(_document([passage]), 0.4), (_document([passage]), 0.5)]
    )

    assert context.text.count("Text: Shared overlap passage.") == 1
    assert list(context.citations) == ["P1"]


def test_build_context_keeps_different_parts_from_the_same_anchor():
    first = _document([_passage("First part of a large table.", "#A5.T9.2")])
    second = _document([_passage("Second part of the same table.", "#A5.T9.2")])

    context = retrieval.build_context([(first, 0.4), (second, 0.5)])

    assert "Text: First part of a large table." in context.text
    assert "Text: Second part of the same table." in context.text
    assert list(context.citations) == ["P1", "P2"]
    assert context.citations["P1"].url == context.citations["P2"].url


def test_build_context_skips_a_malformed_document_and_keeps_valid_evidence():
    malformed = Document(
        page_content="Legacy combined text",
        metadata={"arxiv_id": "1706.03762v7"},
    )
    valid = _document([_passage("The model achieved 28.4 BLEU.")])

    context = retrieval.build_context([(malformed, 0.4), (valid, 0.5)])

    assert "The model achieved 28.4 BLEU." in context.text
    assert list(context.citations) == ["P1"]


def test_build_context_stops_when_all_documents_are_malformed():
    missing_arxiv_id = _document([_passage("Passage")], arxiv_id="")
    empty_passage = _document([_passage("", location="")])

    try:
        retrieval.build_context([(missing_arxiv_id, 0.4), (empty_passage, 0.5)])
    except RuntimeError as error:
        assert str(error) == "Retrieved evidence is invalid."
    else:
        raise AssertionError("Expected corrupt retrieval evidence to fail")


def test_retrieval_failure_is_logged_and_chained_to_a_runtime_error(caplog):
    class FailingStore(RecordingStore):
        def similarity_search_with_score(self, query, k=4):
            raise OSError("database unavailable")

    paper_retriever = retrieval.PaperRetriever(FailingStore(), top_k=3)
    question = "How does attention work?"

    with caplog.at_level("ERROR", logger="arxiv_rag"):
        try:
            paper_retriever.retrieve(question)
        except RuntimeError as error:
            assert str(error) == "Could not retrieve evidence."
            assert isinstance(error.__cause__, OSError)
        else:
            raise AssertionError("Expected retrieval to fail")

    messages = [record.getMessage() for record in caplog.records]
    assert "evidence retrieval failed (top_k=3)" in messages
    assert not any(question in message for message in messages)


def test_traceable_retrieve_still_returns_ranked_documents():
    document = _document([_passage("The model achieved 28.4 BLEU.")])
    store = RecordingStore(results=[(document, 0.2)])
    results = retrieval.PaperRetriever(store).retrieve("How does attention work?")

    assert results == [(document, 0.2)]
    assert not isinstance(results, dict)


def test_traceable_build_context_still_returns_retrieval_context():
    document = _document([_passage("The model achieved 28.4 BLEU.")])
    context = retrieval.build_context([(document, 0.5)])

    assert context.text.startswith("[P1]")
    assert list(context.citations) == ["P1"]
    assert not hasattr(context, "context")
