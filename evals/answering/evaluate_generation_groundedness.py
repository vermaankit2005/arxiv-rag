import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama  # pyright: ignore[reportMissingImports]
from langsmith import Client
from openevals.llm import create_llm_as_judge  # pyright: ignore[reportMissingImports]
from openevals.prompts import RAG_GROUNDEDNESS_PROMPT  # pyright: ignore[reportMissingImports]

from arxiv_rag import answering
from arxiv_rag.retrieval import Citation, RetrievalContext

load_dotenv()

LANGSMITH_DATASET_NAME = "generation_quality_dataset"
EXPERIMENT_PREFIX = "generation_groundedness"
JUDGE_MODEL_NAME = "qwen3.8:27b"

EXPERIMENT_METADATA = {
    "metric": "groundedness",
    "dataset": LANGSMITH_DATASET_NAME,
    "embedding_model": "qwen3-embedding:4b",
    "corpus": "12-papers-384-documents",
    "vector_db": "chroma",
    "judge_model": JUDGE_MODEL_NAME,
}

judge_model = ChatOllama(
    model=JUDGE_MODEL_NAME,
    base_url=os.environ["OLLAMA_BASE_URL"],
    temperature=0,
)

groundedness_judge = create_llm_as_judge(
    prompt=RAG_GROUNDEDNESS_PROMPT,
    feedback_key="groundedness",
    judge=judge_model,
    choices=[0, 0.25, 0.5, 0.75, 1],
)

def _build_section_breadcrumbs(section_path: list[str]) -> str:
    if not section_path:
        return "Unsectioned"
    return " > ".join(section_path)

def _build_passages(context_passages: list[dict]) -> RetrievalContext:
    """Build a list of passages with text and citation information."""

    citations = {}
    context_passages_list = []

    for passage in context_passages:
        section_bread_crumbs = _build_section_breadcrumbs(passage["section_path"])

        url = f"https://arxiv.org/html/{passage['arxiv_id']}{passage['location']}"

        citation_id = f"P{len(citations) + 1}"
        citations[citation_id] = Citation(label=f"{passage['arxiv_id']} — {section_bread_crumbs}", url=url)

        context_passages_list.append(
            f"[{citation_id}]\n"
            f"Section: {section_bread_crumbs}\n"
            f"Text: {passage['text']}"
        )

    return RetrievalContext(
        text="\n\n---\n\n".join(context_passages_list),
        citations=citations,
    )

def generate_answer_for_evaluation(inputs: dict) -> dict:
    """Generate an answer for a given question and log to LangSmith."""
    question = inputs.get("question", "")
    context_passages = inputs.get("context_passages", [])
    retrieval_context = _build_passages(context_passages)
    answer = answering.generate_answer(question, retrieval_context)
    return {
        "answer": answer,
    }

def evaluate_groundedness(inputs: dict, outputs: dict) -> dict:
    context = {
        "documents": [
            passage["text"]
            for passage in inputs.get("context_passages", [])
        ]
    }

    return groundedness_judge(
        context=context,
        outputs={"answer": outputs.get("answer", "")},
    )


def run_groundedness() -> None:
    """Evaluate the retriever by fetching documents for a given question and log to LangSmith."""
    client = Client()
    client.evaluate(
        generate_answer_for_evaluation,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[evaluate_groundedness],
        metadata=EXPERIMENT_METADATA,
        experiment_prefix=EXPERIMENT_PREFIX,
        description=(
            "Evaluate the groundedness of generated answers by comparing them to reference evidence units."
        ),
    )

if __name__ == "__main__":
    run_groundedness()
