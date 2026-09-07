# pyright: reportMissingImports=false
import sys

from arxiv_rag.answering import render_answer
from arxiv_rag.answering.service import answer_question
from arxiv_rag.logging import get_logger

log = get_logger(__name__)


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


if __name__ == "__main__":
    raise SystemExit(main())
