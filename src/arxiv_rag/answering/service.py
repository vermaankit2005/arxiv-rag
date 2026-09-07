import uuid
from dataclasses import dataclass
from typing import Literal

from langsmith import traceable

from arxiv_rag.answering import AnswerMode
from arxiv_rag.graph.workflow_graph import invoke_workflow_graph
from arxiv_rag.retrieval import RetrievalContext


@dataclass(frozen=True)
class AnsweredQuestion:
    thread_id: str
    answer: str
    context: RetrievalContext
    passages_by_id: dict[str, str]
    answer_type: Literal["chat", "rag"]
    answer_mode: AnswerMode


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

    # Calling the workflow graph to process the question and generate an answer
    workflow_result = invoke_workflow_graph(question, thread_id, answer_mode=answer_mode)

    built_context = workflow_result.get("current_built_context")

    answer_type = workflow_result["route"]
    if answer_type is None:
        raise RuntimeError("The workflow completed without selecting an answer route.")

    return AnsweredQuestion(
        thread_id=thread_id,
        answer=workflow_result["answer"],
        context=built_context.context if built_context else RetrievalContext(text="", citations={}),
        passages_by_id=built_context.passages_by_id if built_context else {},
        answer_type=answer_type,
        answer_mode=workflow_result["answer_mode"],
    )
