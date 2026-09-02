import sys
import uuid
from dataclasses import dataclass

from langsmith import traceable

from arxiv_rag.answering import generate_answer, render_answer
from arxiv_rag.retrieval import BuiltContext, PaperRetriever, RetrievalContext


@dataclass(frozen=True)
class AnsweredQuestion:
    thread_id: str
    answer: str
    context: RetrievalContext
    passages_by_id: dict[str, str]


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure:
        reconfigure(encoding="utf-8")

    question = input("Question: ").strip()
    if not question:
        return

    result = answer_question(question)

    print("\n ---- Answer ----\n")
    print(render_answer(result.answer, result.context.citations, clickable=sys.stdout.isatty()))


def answer_question(question: str, thread_id: str | None = None, retriever: PaperRetriever | None = None) -> AnsweredQuestion:
    """Answer one question and return the evidence behind it.

    Callers that keep several questions in one conversation pass the same
    thread_id every time, so the traces group together. Callers that leave it
    out, such as the CLI and eval runs, get a fresh thread per question.
    """
    thread_id = thread_id or str(uuid.uuid4())
    return _answer_question(question, thread_id, retriever, langsmith_extra={"metadata": {"thread_id": thread_id}})


@traceable(
    name="answer_question",
    process_inputs=lambda inputs: {"question": inputs["question"]},
    process_outputs=lambda outputs: {
        "answer": outputs.answer,
        "passages": outputs.context.text,
    },
)
def _answer_question(question: str, thread_id: str, retriever: PaperRetriever | None) -> AnsweredQuestion:
    built: BuiltContext = (retriever or PaperRetriever()).retrieve_context_with_details(question)
    answer = generate_answer(question, built.context)

    return AnsweredQuestion(
        thread_id=thread_id,
        answer=answer,
        context=built.context,
        passages_by_id=built.passages_by_id,
    )


if __name__ == "__main__":
    main()
