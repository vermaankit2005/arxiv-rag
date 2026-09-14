"""This checks how much required evidence appears in the final 15 reranked passages. For each question, the formula is: covered evidence units / total evidence units. The final score is the average across all questions."""

from dotenv import load_dotenv
from langsmith import Client

from arxiv_rag import retrieval
from arxiv_rag.retrieval.reranker import COHERE_RERANK_MODEL, RERANK_TOP_N
from evals.retriever.reranker_context import fetch_reranked_passages
from evals.utils import eval_upload_enabled, print_local_score

DESCRIPTION = __doc__
LANGSMITH_DATASET_NAME = "retrieval_evidence_dataset"
EXPERIMENT_PREFIX = "reranker-passage-evidence-recall-at-15"
EXPERIMENT_METADATA = {
    "metric": "passage_evidence_recall_at_15",
    "evaluation_unit": "passage",
    "dataset": LANGSMITH_DATASET_NAME,
    "embedding_model": "qwen3-embedding:4b",
    "retriever_top_k": retrieval.DEFAULT_TOP_K,
    "reranker_model": COHERE_RERANK_MODEL,
    "reranker_top_n": RERANK_TOP_N,
    "corpus": "12-papers-384-documents",
    "vector_db": "chroma",
}


def fetch_passages_for_evaluation(inputs: dict) -> dict:
    return fetch_reranked_passages(inputs)


def evaluate_passage_evidence_recall(outputs: dict, reference_outputs: dict) -> dict:
    covered_evidence_units = 0
    evidence_units = reference_outputs.get("evidence_units", [])

    for evidence_unit in evidence_units:
        unit_matched = False
        for accepted_evidence in evidence_unit.get("accepted_evidence", []):

            for passage in outputs.get("passages", []):
                if (
                    accepted_evidence.get("arxiv_id") == passage.get("arxiv_id")
                    and accepted_evidence.get("location") == passage.get("location")
                    and accepted_evidence.get("quote") in passage.get("text", "")
                ):
                    unit_matched = True
                    break
            if unit_matched:
                break
        if unit_matched:
            covered_evidence_units += 1

    score = covered_evidence_units / len(evidence_units) if evidence_units else 0.0
    return {"key": "passage_evidence_recall_at_15", "score": score}


def run_passage_evidence_recall() -> None:
    load_dotenv()
    client = Client()
    upload_results = eval_upload_enabled()
    results = client.evaluate(
        fetch_passages_for_evaluation,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[evaluate_passage_evidence_recall],
        metadata=EXPERIMENT_METADATA,
        experiment_prefix=EXPERIMENT_PREFIX,
        description=DESCRIPTION,
        max_concurrency=1,
        blocking=True,
        upload_results=upload_results,
    )

    if not upload_results:
        completed_results = list(results)
        print_local_score(completed_results, "passage_evidence_recall_at_15")


if __name__ == "__main__":
    run_passage_evidence_recall()
