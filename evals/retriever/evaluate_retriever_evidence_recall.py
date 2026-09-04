"""This checks how much of the evidence needed for each question appears in the top 5 Documents. For each question, the formula is: covered evidence units / total evidence units. The final score is the average across all questions."""

from langsmith import Client

from arxiv_rag import retrieval

DESCRIPTION = __doc__
LANGSMITH_DATASET_NAME = "retrieval_evidence_dataset"
EXPERIMENT_PREFIX = "retriever-evidence-recall-at-5"
EXPERIMENT_METADATA = {
    "metric": "evidence_recall_at_5",
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

        for source_passage in retrieval.get_source_passages_for_a_document(doc):

            source_key = (doc.metadata.get("arxiv_id"), source_passage.location, source_passage.text)

            if source_key in source_passage_set:
                continue
            source_passage_set.add(source_key)

            source_passage_list.append({
                "arxiv_id": doc.metadata.get("arxiv_id"),
                "text": source_passage.text,
                "location": source_passage.location,
            })

    return {
        "source_passages": source_passage_list
    }


def evaluate_evidence_recall(outputs: dict, reference_outputs: dict) -> dict:
    len_covered_evidence_unit = 0
    len_all_evidence_unit = len(reference_outputs.get("evidence_units", []))

    for evidence_unit in reference_outputs.get("evidence_units", []):
        unit_matched = False

        for accepted_evidence in evidence_unit.get("accepted_evidence", []):

            for source_passage in outputs.get("source_passages", []):
                if (
                        accepted_evidence.get("arxiv_id") == source_passage.get("arxiv_id")
                        and accepted_evidence.get("location") == source_passage.get("location")
                        and accepted_evidence.get("quote") in source_passage.get("text")
                ):
                    unit_matched = True
                    break  # leave source_passage loop

            if unit_matched:
                break  # leave accepted_evidence loop → next evidence_unit
        if unit_matched:
            len_covered_evidence_unit += 1

    score = len_covered_evidence_unit / len_all_evidence_unit if len_all_evidence_unit > 0 else 0.0

    return {
        "key": "evidence_recall",
        "score": score,
    }


def run_evidence_recall() -> None:
    """Evaluate the retriever by fetching documents for a given question and log to LangSmith."""
    client = Client()
    client.evaluate(
        fetch_docs_for_evaluation,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[evaluate_evidence_recall],
        metadata=EXPERIMENT_METADATA,
        experiment_prefix=EXPERIMENT_PREFIX,
        description=DESCRIPTION,
    )


if __name__ == "__main__":
    run_evidence_recall()
