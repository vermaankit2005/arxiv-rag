from langsmith import Client

from arxiv_rag import retrieval

LANGSMITH_DATASET_NAME = "retrieval_evidence_dataset"
EXPERIMENT_PREFIX = "retriever-mrr-at-5"
EXPERIMENT_METADATA = {
    "metric": "mean_reciprocal_rank_at_5",
    "dataset": LANGSMITH_DATASET_NAME,
    "embedding_model": "qwen3-embedding:4b",
    "top_k": retrieval.DEFAULT_TOP_K,
    "corpus": "12-papers-384-documents",
    "vector_db": "chroma",
}
doc_ranking = {
    "rank_1": 1.0,
    "rank_2": 0.5,
    "rank_3": 0.3333,
    "rank_4": 0.25,
    "rank_5": 0.1
}

retriever = retrieval.PaperRetriever()


def fetch_docs_for_evaluation(inputs: dict) -> dict | None:
    retrieved_doc_list = retriever.retrieve(inputs["question"])

    if not retrieved_doc_list:
        return None

    return {
        "doc_list": [doc for doc, _ in retrieved_doc_list],
    }


def evaluate_mrr(outputs: dict, reference_outputs: dict) -> dict:
    score = 0.0

    for evidence_unit in reference_outputs.get("evidence_units", []):
        unit_matched = False

        for accepted_evidence in evidence_unit.get("accepted_evidence", []):

            for rank, doc in enumerate(outputs.get("doc_list", []), start=1):
                if accepted_evidence.get("quote") in doc.page_content:
                    score +=(1 / rank)
                    unit_matched = True
                    break
            if unit_matched:
                break
        if unit_matched:
            break

    return {
        "key": "mrr_at_5",
        "score": score,
    }

def run_mrr() -> None:
    """Evaluate the retriever by fetching documents for a given question and log to LangSmith."""
    client = Client()
    client.evaluate(
        fetch_docs_for_evaluation,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[evaluate_mrr],
        metadata=EXPERIMENT_METADATA,
        experiment_prefix=EXPERIMENT_PREFIX,
        description=(
            "This checks how early the first useful Document appears in the top 5. "
            "For each question, the formula is: 1 / rank of the first Document "
            "containing required evidence. The score is 0 when no evidence appears "
            "in the top 5, and the final score is the average across all questions."
        ),
    )

if __name__ == "__main__":
    run_mrr()
