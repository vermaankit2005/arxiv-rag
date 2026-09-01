"""This checks how much of the source-passage context from the top 5 Documents directly matches the required evidence. The formula is: relevant unique source passages / all unique source passages. Higher means less unrelated context was retrieved."""

from langsmith import Client

from arxiv_rag import retrieval

DESCRIPTION = __doc__
LANGSMITH_DATASET_NAME = "retrieval_evidence_dataset"
EXPERIMENT_PREFIX = "retriever-context-precision-at-5"
EXPERIMENT_METADATA = {
    "metric": "context_precision_at_5",
    "evaluation_unit": "source_passage",
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

    source_passage_set = set()
    source_passage_list = []

    for doc, _ in retrieved_doc_list:
        for source_passage in retrieval.get_source_passage_for_a_document(doc):
            source_key = (
                doc.metadata.get("arxiv_id"),
                source_passage.location,
                source_passage.text,
            )

            if source_key in source_passage_set:
                continue
            source_passage_set.add(source_key)

            source_passage_list.append(
                {
                    "arxiv_id": doc.metadata.get("arxiv_id"),
                    "text": source_passage.text,
                    "location": source_passage.location,
                }
            )

    return {"source_passages": source_passage_list}


def evaluate_context_precision(outputs: dict, reference_outputs: dict) -> dict:
    matched_sources = 0
    source_passages = outputs.get("source_passages", [])
    total_sources = len(source_passages)

    for source_passage in source_passages:
        passage_matched = False

        for evidence_unit in reference_outputs.get("evidence_units", []):
            for accepted_evidence in evidence_unit.get("accepted_evidence", []):
                if (
                    accepted_evidence.get("arxiv_id")
                    == source_passage.get("arxiv_id")
                    and accepted_evidence.get("location")
                    == source_passage.get("location")
                    and accepted_evidence.get("quote")
                    in source_passage.get("text")
                ):
                    passage_matched = True
                    break

            if passage_matched:
                break

        if passage_matched:
            matched_sources += 1

    score = matched_sources / total_sources if total_sources > 0 else 0.0

    return {
        "key": "context_precision",
        "score": score,
    }


def run_context_precision() -> None:
    client = Client()
    client.evaluate(
        fetch_docs_for_evaluation,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[evaluate_context_precision],
        metadata=EXPERIMENT_METADATA,
        experiment_prefix=EXPERIMENT_PREFIX,
        description=DESCRIPTION,
    )


if __name__ == "__main__":
    run_context_precision()
