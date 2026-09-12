from contextlib import nullcontext

from langchain_core.documents import Document  # pyright: ignore[reportMissingImports]

import arxiv_rag.retrieval.reranker as reranker_module


def test_rerank_passages_returns_passage_ids_in_relevance_order(monkeypatch):
    client_closed = []

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            client_closed.append(True)

    class FakeReranker:
        client = FakeClient()

        def compress_documents(self, documents, query):
            assert query == "attention"
            assert [document.page_content for document in documents] == [
                "First passage",
                "Second passage",
            ]
            return [
                Document(page_content="Second passage", metadata={"citation_id": "P2"})
            ]

    monkeypatch.setattr(reranker_module, "_get_reranker", FakeReranker)

    ranked_ids = reranker_module.rerank_passages(
        {"P1": "First passage", "P2": "Second passage"},
        "attention",
    )

    assert ranked_ids == ["P2"]
    assert client_closed == [True]


def test_rerank_passages_wraps_cohere_failures(monkeypatch):
    class FailingReranker:
        client = nullcontext()

        def compress_documents(self, documents, query):
            raise OSError("Cohere unavailable")

    monkeypatch.setattr(reranker_module, "_get_reranker", FailingReranker)

    try:
        reranker_module.rerank_passages({"P1": "First passage"}, "attention")
    except RuntimeError as error:
        assert str(error) == "Could not rerank retrieved evidence."
        assert isinstance(error.__cause__, OSError)
    else:
        raise AssertionError("Expected reranking to fail")


def test_rerank_passages_skips_cohere_when_there_are_no_passages(monkeypatch):
    def fail_if_called():
        raise AssertionError("Cohere should not be called for empty evidence")

    monkeypatch.setattr(reranker_module, "_get_reranker", fail_if_called)

    assert reranker_module.rerank_passages({}, "attention") == []
