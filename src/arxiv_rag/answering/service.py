import uuid

from langsmith import traceable

from arxiv_rag.answering import AnswerMode
from arxiv_rag.answering.models import AnsweredQuestion
from arxiv_rag.answering.service_stream import AnswerComplete, answer_question_stream


def answer_question(question: str, thread_id: str | None = None,
                    answer_mode: AnswerMode = "standard") -> AnsweredQuestion:
    """Answer one question and return the answer with its supporting evidence."""
    thread_id = thread_id or str(uuid.uuid4())

    return _answer_question(
        question,
        thread_id,
        answer_mode,
        langsmith_extra={"metadata": {"thread_id": thread_id, "answer_mode": answer_mode}},
    )

@traceable(
    name="answer_question",
    process_inputs=lambda inputs: {"question": inputs["question"], "answer_mode": inputs["answer_mode"]},
    process_outputs=lambda outputs: {
        "answer": outputs.answer,
        "passages": outputs.context.text,
    },
)
def _answer_question(question: str, thread_id: str, answer_mode: AnswerMode) -> AnsweredQuestion:
    for event in answer_question_stream(question, thread_id=thread_id, answer_mode=answer_mode):
        if isinstance(event, AnswerComplete):
            return event.result

    raise RuntimeError("The answer stream completed without a final result.")
