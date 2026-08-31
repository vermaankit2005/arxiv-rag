from evals.answering import evaluate_generation_completeness as completeness


def _passage(citation_id: str, text: str) -> dict:
    return {
        "id": citation_id,
        "arxiv_id": "paper-v1",
        "location": f"#{citation_id}",
        "section_path": ["Results"],
        "text": text,
    }


def _fact(fact_id: str, fact: str, passage_ids: list[str]) -> dict:
    return {
        "id": fact_id,
        "fact": fact,
        "supporting_passage_ids": passage_ids,
    }


class RecordingJudge:
    def __init__(self, scores: list[bool]):
        self.scores = iter(scores)
        self.calls = []

    def __call__(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return {
            "key": "required_fact_coverage",
            "score": next(self.scores),
            "comment": "The required fact was checked.",
        }


def test_scores_covered_required_facts_over_all_required_facts():
    judge = RecordingJudge([True, False, True])
    inputs = {
        "question": "What happened?",
        "context_passages": [
            _passage("P1", "First fact evidence."),
            _passage("P2", "Second and third fact evidence."),
        ],
    }
    reference_outputs = {
        "required_facts": [
            _fact("F1", "First fact.", ["P1"]),
            _fact("F2", "Second fact.", ["P2"]),
            _fact("F3", "Third fact.", ["P2"]),
        ]
    }

    result = completeness.evaluate_completeness(
        inputs,
        {"answer": "The answer covers the first and third facts."},
        reference_outputs,
        judge,
    )

    assert result == {
        "key": "completeness",
        "score": 2 / 3,
        "comment": "2/3 required facts were covered.",
    }
    assert [call["inputs"]["required_fact"]["id"] for call in judge.calls] == ["F1", "F2", "F3"]


def test_rejects_an_invalid_binary_coverage_score():
    class InvalidJudge:
        def __call__(self, **kwargs) -> dict:
            return {"score": 0.5}

    try:
        completeness.evaluate_completeness(
            {
                "question": "Question",
                "context_passages": [_passage("P1", "Evidence")],
            },
            {"answer": "Answer"},
            {"required_facts": [_fact("F1", "Fact", ["P1"])]},
            InvalidJudge(),
        )
    except ValueError as error:
        assert str(error) == "Completeness judge returned an invalid score: 0.5"
    else:
        raise AssertionError("Expected a non-binary completeness score to fail")


def test_requires_at_least_one_required_fact():
    try:
        completeness.evaluate_completeness(
            {"question": "Question", "context_passages": []},
            {"answer": "Answer"},
            {"required_facts": []},
            RecordingJudge([]),
        )
    except ValueError as error:
        assert str(error) == "Completeness evaluation requires at least one required fact"
    else:
        raise AssertionError("Expected an empty required-fact list to fail")
