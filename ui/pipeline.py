"""The only place the UI reaches into the RAG pipeline."""

import streamlit as st

from arxiv_rag.retrieval import PaperRetriever, build_context_with_details


@st.cache_resource(show_spinner=False)
def get_retriever(top_k: int) -> PaperRetriever:
    """One retriever per top_k, opened once and reused across reruns."""
    return PaperRetriever(top_k=top_k)


def retrieve_evidence(question: str, top_k: int):
    """Retrieve passages, keeping both the model context and the passage text.

    PaperRetriever.retrieve_context drops the passage text, and the UI needs it
    to show what each citation actually says.
    """
    results = get_retriever(top_k).retrieve(question)
    return build_context_with_details(results)
