import uuid
from collections.abc import Iterator

from arxiv_rag.answering import AnswerMode
from arxiv_rag.answering.models import AnswerChunk, AnswerComplete, AnsweredQuestion
from arxiv_rag.graph.workflow_graph import stream_workflow_graph
from arxiv_rag.retrieval import RetrievalContext

AnswerEvent = AnswerChunk | AnswerComplete


def answer_question_stream(question: str, thread_id: str | None = None,
                           answer_mode: AnswerMode = "standard") -> Iterator[AnswerEvent]:
    """Yield display chunks followed by one authoritative final response."""
    thread_id = thread_id or str(uuid.uuid4())
    final_state = None

    for event in stream_workflow_graph(question, thread_id, answer_mode=answer_mode):
        if isinstance(event, str):
            yield AnswerChunk(text=event)
        else:
            final_state = event

    if final_state is None:
        raise RuntimeError("The workflow did not complete successfully.")

    built_context = final_state.get("current_context")
    answer_type = final_state["route"]
    if answer_type is None:
        raise RuntimeError("The workflow completed without selecting an answer route.")

    yield AnswerComplete(result=AnsweredQuestion(
        thread_id=thread_id,
        answer=final_state["answer"],
        context=built_context.context if built_context else RetrievalContext(text="", citations={}),
        passages_by_id=built_context.passages_by_id if built_context else {},
        answer_type=answer_type,
        answer_mode=final_state["answer_mode"],
    ))
