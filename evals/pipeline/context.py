from arxiv_rag.answering import generate_answer
from arxiv_rag.retrieval import PaperRetriever


def generate_pipeline_answer_for_evaluation(inputs: dict) -> dict:
    """Run live production retrieval and generation for one evaluation question."""
    question = inputs.get("question", "")
    context = PaperRetriever().retrieve_context(question)
    answer = generate_answer(question, context)
    return {"answer": answer}
