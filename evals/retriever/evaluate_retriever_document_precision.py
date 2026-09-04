"""This checks how many of the top 5 retrieved Documents contain required evidence. The formula is: relevant Documents / 5. We use Documents as the unit because the passage-level metric treated useful neighbouring passages as noise when they were not part of the minimal answer key."""

from langsmith import Client

from arxiv_rag import retrieval

DESCRIPTION = __doc__
LANGSMITH_DATASET_NAME = "retrieval_evidence_dataset"
EXPERIMENT_PREFIX = "retriever-document-precision-at-5"
EXPERIMENT_METADATA = {
    "metric": "document_precision_at_5",
    "evaluation_unit": "document",
    "dataset": LANGSMITH_DATASET_NAME,
    "embedding_model": "qwen3-embedding:4b",
    "top_k": retrieval.DEFAULT_TOP_K,
    "corpus": "12-papers-384-documents",
    "vector_db": "chroma",
}

retriever = retrieval.PaperRetriever()


def fetch_docs_for_evaluation(inputs: dict) -> dict | None:
    retrieved_doc_list = retriever.retrieve(inputs["question"])

    if not retrieved_doc_list:
        return None

    documents = []

    for doc, _ in retrieved_doc_list:
        source_passages = [
            {
                "arxiv_id": doc.metadata.get("arxiv_id"),
                "text": source_passage.text,
                "location": source_passage.location,
            }
            for source_passage in retrieval.get_source_passages_for_a_document(doc)
        ]
        documents.append({"source_passages": source_passages})

    return {"documents": documents}


def _passage_matches_evidence(source_passage: dict, reference_outputs: dict) -> bool:
    for evidence_unit in reference_outputs.get("evidence_units", []):
        for accepted_evidence in evidence_unit.get("accepted_evidence", []):
            if (
                accepted_evidence.get("arxiv_id") == source_passage.get("arxiv_id")
                and accepted_evidence.get("location")
                == source_passage.get("location")
                and accepted_evidence.get("quote") in source_passage.get("text")
            ):
                return True

    return False


def evaluate_document_precision(outputs: dict, reference_outputs: dict) -> dict:
    relevant_documents = 0

    for document in outputs.get("documents", []):
        if any(
            _passage_matches_evidence(source_passage, reference_outputs)
            for source_passage in document.get("source_passages", [])
        ):
            relevant_documents += 1

    score = relevant_documents / retrieval.DEFAULT_TOP_K

    return {
        "key": "document_precision_at_5",
        "score": score,
        "comment": (
            f"Found required evidence in {relevant_documents}/"
            f"{retrieval.DEFAULT_TOP_K} retrieved Documents."
        ),
    }


def run_document_precision() -> None:
    client = Client()
    client.evaluate(
        fetch_docs_for_evaluation,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[evaluate_document_precision],
        metadata=EXPERIMENT_METADATA,
        experiment_prefix=EXPERIMENT_PREFIX,
        description=DESCRIPTION,
    )


if __name__ == "__main__":
    run_document_precision()
