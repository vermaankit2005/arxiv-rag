import sys
import uuid
from dataclasses import dataclass
from typing import Literal

from langsmith import traceable

from arxiv_rag.answering import AnswerMode, render_answer
from arxiv_rag.graph.workflow_graph import invoke_workflow_graph
from arxiv_rag.logging import get_logger
from arxiv_rag.retrieval import RetrievalContext

log = get_logger(__name__)


@dataclass(frozen=True)
class AnsweredQuestion:
    thread_id: str
    answer: str
    context: RetrievalContext
    passages_by_id: dict[str, str]
    answer_type: Literal["chat", "rag"]
    answer_mode: AnswerMode


def main() -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure:
        reconfigure(encoding="utf-8")

    try:
        question = input("Question: ").strip()
        if not question:
            return 0

        result = answer_question(question)
        print("\n ---- Answer ----\n")
        print(render_answer(result.answer, result.context.citations, clickable=sys.stdout.isatty()))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        log.error("could not answer question: %s", error)
        return 1
    except Exception:
        log.exception("could not answer question because of an unexpected failure")
        return 1
    return 0


def answer_question(question: str, thread_id: str | None = None,
                    answer_mode: AnswerMode = "standard") -> AnsweredQuestion:
    """Answer one question and return the evidence behind it.

    Callers that keep several questions in one conversation pass the same
    thread_id every time, so the traces group together. Callers that leave it
    out, such as the CLI and eval runs, get a fresh thread per question.

    """
    thread_id = thread_id or str(uuid.uuid4())
    return _answer_question(
        question,
        thread_id,
        answer_mode,
        langsmith_extra={"metadata": {"thread_id": thread_id, "answer_mode": answer_mode}},
    )


# @traceable(
#     name="answer_question",
#     process_inputs=lambda inputs: {"question": inputs["question"], "answer_mode": inputs["answer_mode"]},
#     process_outputs=lambda outputs: {
#         "answer": outputs.answer,
#         "passages": outputs.context.text,
#     },
# )
# def _answer_question(question: str, thread_id: str, retriever: PaperRetriever | None, answer_mode: AnswerMode) -> AnsweredQuestion:
#
#     built: BuiltContext = (retriever or PaperRetriever()).retrieve_context_with_details(question)
#     answer = generate_answer(question, built.context, answer_mode=answer_mode)
#
#     return AnsweredQuestion(
#         thread_id=thread_id,
#         answer=answer,
#         context=built.context,
#         passages_by_id=built.passages_by_id,
#     )


#   ------------------- Changing the flow to call the graph instead of RAG pipeline directly -------------------
#   ------------------- RAG pipeline is now part of the graph, so we can call the graph directly to get the answer -------------------
@traceable(
    name="answer_question",
    process_inputs=lambda inputs: {"question": inputs["question"], "answer_mode": inputs["answer_mode"]},
    process_outputs=lambda outputs: {
        "answer": outputs.answer,
        "passages": outputs.context.text,
    },
)
def _answer_question(question: str, thread_id: str, answer_mode: AnswerMode) -> AnsweredQuestion:

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


if __name__ == "__main__":
    raise SystemExit(main())
