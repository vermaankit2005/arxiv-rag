"""The only place the UI reaches into the RAG pipeline."""

# pyright: reportMissingImports=false
from collections.abc import Iterator

from arxiv_rag.answering import AnswerMode
from arxiv_rag.answering.service_stream import AnswerChunk, AnswerComplete, answer_question_stream

# Keep the non-streaming path nearby while streaming is under development. To
# switch back, restore this import and function, then restore the commented UI
# block in streamlit_app.py.
# from arxiv_rag.answering.service import AnsweredQuestion, answer_question
#
# def answer_in_conversation(question: str, thread_id: str,
#                            answer_mode: AnswerMode = "standard") -> AnsweredQuestion:
#     """Run the non-streaming backend entry point inside this chat's thread."""
#     return answer_question(question, thread_id=thread_id, answer_mode=answer_mode)


def stream_answer_in_conversation(question: str, thread_id: str,
                                  answer_mode: AnswerMode = "standard") -> Iterator[AnswerChunk | AnswerComplete]:
    """Stream the backend answer inside this chat's existing thread."""
    return answer_question_stream(question, thread_id=thread_id, answer_mode=answer_mode)
