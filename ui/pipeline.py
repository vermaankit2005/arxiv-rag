"""The only place the UI reaches into the RAG pipeline."""

import streamlit as st  # pyright: ignore[reportMissingImports]

from arxiv_rag.answering import AnswerMode
from arxiv_rag.answering.__main__ import AnsweredQuestion, answer_question
from arxiv_rag.retrieval import PaperRetriever


@st.cache_resource(show_spinner=False)
def get_retriever(top_k: int) -> PaperRetriever:
    """One retriever per top_k, opened once and reused across reruns."""
    return PaperRetriever(top_k=top_k)


def answer_in_conversation(
    question: str,
    top_k: int,
    thread_id: str,
    answer_mode: AnswerMode = "standard",
) -> AnsweredQuestion:
    """Run the same backend entry point the CLI uses, inside this chat's thread."""
    return answer_question(
        question,
        thread_id=thread_id,
        retriever=get_retriever(top_k),
        answer_mode=answer_mode,
    )
