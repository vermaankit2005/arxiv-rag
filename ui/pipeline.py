"""The only place the UI reaches into the RAG pipeline."""

from arxiv_rag.answering import AnswerMode
from arxiv_rag.answering.__main__ import AnsweredQuestion, answer_question


def answer_in_conversation(question: str, thread_id: str, answer_mode: AnswerMode = "standard") -> AnsweredQuestion:
    """Run the same backend entry point the CLI uses, inside this chat's thread."""
    return answer_question(question, thread_id=thread_id, answer_mode=answer_mode)
