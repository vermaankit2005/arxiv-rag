from langchain_core.documents import Document  # pyright: ignore[reportMissingImports]

import arxiv_rag.graph.rerank_node as rerank_node_module
from arxiv_rag.retrieval import BuiltContext, Citation, RetrievalContext


def test_rerank_context_preserves_original_and_builds_ranked_context(monkeypatch):
    original = BuiltContext(
        context=RetrievalContext(
            text="original context",
            citations={
                "P1": Citation(label="paper — Introduction", url="https://example.com/1"),
                "P2": Citation(label="paper — Results", url="https://example.com/2"),
            },
        ),
        passages_by_id={"P1": "First passage", "P2": "Second passage"},
    )

    class FakeReranker:
        def compress_documents(self, documents, query):
            assert query == "attention"
            assert [document.page_content for document in documents] == [
                "First passage",
                "Second passage",
            ]
            return [Document(page_content="Second passage", metadata={"citation_id": "P2"})]

    monkeypatch.setattr(rerank_node_module, "_get_reranker", FakeReranker)

    reranked = rerank_node_module.rerank_context(original, "attention")

    assert original.passages_by_id == {"P1": "First passage", "P2": "Second passage"}
    assert reranked.passages_by_id == {"P2": "Second passage"}
    assert reranked.context.citations == {"P2": original.context.citations["P2"]}
    assert reranked.context.text == "[P2]\nSection: Results\nText: Second passage"


def test_rerank_node_keeps_original_context_in_state(monkeypatch):
    original = BuiltContext(
        context=RetrievalContext(text="", citations={}),
        passages_by_id={},
    )
    state = {
        "current_built_context": original,
        "retrieval_query": "attention",
    }
    monkeypatch.setattr(rerank_node_module, "rerank_context", lambda context, query: context)

    result = rerank_node_module.rerank_node(state)  # pyright: ignore[reportArgumentType]

    assert result == {"current_reranked_context": original}
    assert state["current_built_context"] is original
