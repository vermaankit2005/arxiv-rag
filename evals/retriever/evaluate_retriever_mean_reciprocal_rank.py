"""This checks how early the first useful Document appears in the top 5. For each question, the formula is: 1 / rank of the first Document containing required evidence. The score is 0 when no evidence appears in the top 5, and the final score is the average across all questions."""

from dotenv import load_dotenv
from langsmith import Client

from arxiv_rag import retrieval

DESCRIPTION = __doc__
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


def fetch_docs_for_evaluation(inputs: dict) -> dict | None:

    retriever = retrieval.PaperRetriever()
    retrieved_doc_list = retriever.retrieve(inputs["question"])

    if not retrieved_doc_list:
        return None

    documents = []
    for doc, _ in retrieved_doc_list:
        source_passages = []
        for source_passage in retrieval.get_source_passages_for_a_document(doc):
            source_passages.append({
                "arxiv_id": doc.metadata.get("arxiv_id"),
                "text": source_passage.text,
                "location": source_passage.location,
            })

        documents.append({"source_passages": source_passages})

    return {"documents": documents}


def _passage_matches_evidence(source_passage: dict, reference_outputs: dict) -> bool:

    for evidence_unit in reference_outputs.get("evidence_units", []):

        for accepted_evidence in evidence_unit.get("accepted_evidence", []):

            if (
                accepted_evidence.get("arxiv_id") == source_passage.get("arxiv_id")
                and accepted_evidence.get("location") == source_passage.get("location")
                and accepted_evidence.get("quote") in source_passage.get("text", "")
            ):
                return True

    return False


def evaluate_mrr(outputs: dict, reference_outputs: dict) -> dict:
    for rank, document in enumerate(outputs.get("documents", []), start=1):
        if any(
            _passage_matches_evidence(source_passage, reference_outputs)
            for source_passage in document.get("source_passages", [])
        ):
            return {"key": "mrr_at_5", "score": 1 / rank}

    return {"key": "mrr_at_5", "score": 0.0}


def run_mrr() -> None:
    """Evaluate the retriever by fetching documents for a given question and log to LangSmith."""
    load_dotenv()
    client = Client()
    client.evaluate(
        fetch_docs_for_evaluation,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[evaluate_mrr],
        metadata=EXPERIMENT_METADATA,
        experiment_prefix=EXPERIMENT_PREFIX,
        description=DESCRIPTION,
    )


if __name__ == "__main__":
    run_mrr()
