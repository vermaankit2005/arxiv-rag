from arxiv_rag.retrieval import PaperRetriever


def fetch_reranked_passages(inputs: dict) -> dict:
    """Run shipping retrieval and expose the final reranked passages with provenance."""
    retriever = PaperRetriever()
    try:
        built_context = retriever.retrieve(inputs["question"])
    finally:
        retriever.close()

    if not built_context.passages_by_id:
        return {"passages": []}

    passages = []
    for passage_id, text in built_context.passages_by_id.items():

        citation = built_context.context.citations[passage_id]
        arxiv_id = citation.label.partition(" — ")[0]
        url_prefix = f"https://arxiv.org/html/{arxiv_id}"

        if not citation.url.startswith(url_prefix):
            raise RuntimeError("Reranked passage citation metadata is invalid.")

        passages.append({
            "arxiv_id": arxiv_id,
            "text": text,
            "location": citation.url.removeprefix(url_prefix),
        })

    return {"passages": passages}
