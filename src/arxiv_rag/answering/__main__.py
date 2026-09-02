import sys

from arxiv_rag.answering import generate_answer, render_answer
from arxiv_rag.retrieval import PaperRetriever


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure:
        reconfigure(encoding="utf-8")

    question = input("Question: ").strip()
    if not question:
        return

    context = PaperRetriever().retrieve_context(question)
    answer = generate_answer(question, context)
    print("\n ---- Answer ----\n")
    print(render_answer(answer, context.citations, clickable=sys.stdout.isatty()))


def answer_question(question: str) -> dict:
    context = PaperRetriever().retrieve_context(question)
    return generate_answer(question, context)


if __name__ == "__main__":
    main()
