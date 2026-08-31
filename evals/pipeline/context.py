from arxiv_rag.answering import generate_answer
from arxiv_rag.retrieval import PaperRetriever, build_context_details


def generate_pipeline_answer_for_evaluation(inputs: dict) -> dict:
    """Run live production retrieval and generation for one evaluation question."""
    question = inputs.get("question", "")
    context = PaperRetriever().retrieve_context(question)
    answer = generate_answer(question, context)
    return {"answer": answer}


def generate_pipeline_answer_and_passages_for_evaluation(inputs: dict) -> dict:
    """Run the pipeline and expose its retrieved evidence to evaluators."""
    question = inputs.get("question", "")
    results = PaperRetriever().retrieve(question)
    built = build_context_details(results)
    answer = generate_answer(question, built.context)
    return {
        "answer": answer,
        "retrieved_context": built.context.text,
        "retrieved_passages": built.passages_by_id,
    }
