from evals.retriever.evaluate_retriever_mean_reciprocal_rank import evaluate_mrr


def _passage(arxiv_id: str, location: str, text: str) -> dict:
    return {"arxiv_id": arxiv_id, "location": location, "text": text}


def _document(*passages: dict) -> dict:
    return {"source_passages": list(passages)}


def _references() -> dict:
    return {
        "evidence_units": [
            {
                "id": "first-listed-unit",
                "accepted_evidence": [
                    {"arxiv_id": "paper-a", "location": "#a", "quote": "alpha evidence"}
                ],
            },
            {
                "id": "second-listed-unit",
                "accepted_evidence": [
                    {"arxiv_id": "paper-b", "location": "#b", "quote": "beta evidence"}
                ],
            },
        ]
    }


def test_mrr_uses_earliest_relevant_document_across_all_evidence_units():
    outputs = {
        "documents": [
            _document(_passage("paper-b", "#b", "contains beta evidence")),
            _document(_passage("paper-a", "#a", "contains alpha evidence")),
        ]
    }

    result = evaluate_mrr(outputs, _references())

    assert result == {"key": "mrr_at_5", "score": 1.0}


def test_mrr_requires_arxiv_id_location_and_quote_to_match():
    outputs = {
        "documents": [
            _document(_passage("wrong-paper", "#a", "contains alpha evidence")),
            _document(_passage("paper-a", "#wrong", "contains alpha evidence")),
            _document(_passage("paper-a", "#a", "contains alpha evidence")),
        ]
    }

    result = evaluate_mrr(outputs, _references())

    assert result == {"key": "mrr_at_5", "score": 1 / 3}


def test_mrr_is_zero_when_no_document_contains_required_evidence():
    outputs = {
        "documents": [
            _document(_passage("paper-a", "#a", "unrelated text")),
            _document(_passage("paper-b", "#b", "still unrelated")),
        ]
    }

    result = evaluate_mrr(outputs, _references())

    assert result == {"key": "mrr_at_5", "score": 0.0}
