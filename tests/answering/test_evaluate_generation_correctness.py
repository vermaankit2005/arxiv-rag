from evals.answering import evaluate_generation_correctness as correctness


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
    def __init__(self, score):
        self.score = score
        self.calls = []

    def __call__(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return {
            "key": "correctness",
            "score": self.score,
            "comment": "The answer was checked against the reference.",
        }


def test_builds_required_facts_with_only_their_supporting_passages():
    references = correctness._build_fact_references(
        [_passage("P1", "Neighbour"), _passage("P2", "The score was 92.7.")],
        [_fact("F1", "The score was 92.7.", ["P2"])],
    )

    assert references == [{
        "id": "F1",
        "fact": "The score was 92.7.",
        "supporting_passages": [{"id": "P2", "text": "The score was 92.7."}],
    }]


def test_returns_the_restricted_correctness_score_and_comment():
    judge = RecordingJudge(0.75)
    result = correctness.evaluate_correctness(
        {
            "question": "What score was reported?",
            "context_passages": [_passage("P2", "The score was 92.7.")],
        },
        {"answer": "The score was 92.7 [P2]."},
        {"required_facts": [_fact("F1", "The score was 92.7.", ["P2"])]},
        judge=judge,
    )

    assert result == {
        "key": "correctness",
        "score": 0.75,
        "comment": "The answer was checked against the reference.",
    }
    assert judge.calls[0]["inputs"]["required_facts"][0]["id"] == "F1"


def test_rejects_unknown_supporting_passage_ids_before_judging():
    judge = RecordingJudge(1)

    try:
        correctness.evaluate_correctness(
            {"question": "Question", "context_passages": []},
            {"answer": "Answer"},
            {"required_facts": [_fact("F1", "Fact", ["P9"])]},
            judge=judge,
        )
    except ValueError as error:
        assert str(error) == "Required fact F1 used unknown passage IDs: P9"
    else:
        raise AssertionError("Expected an unknown passage ID to fail")

    assert judge.calls == []


def test_rejects_a_score_outside_the_five_allowed_values():
    try:
        correctness.evaluate_correctness(
            {
                "question": "Question",
                "context_passages": [_passage("P1", "Evidence")],
            },
            {"answer": "Answer"},
            {"required_facts": [_fact("F1", "Fact", ["P1"])]},
            judge=RecordingJudge(0.6),
        )
    except ValueError as error:
        assert str(error) == "Correctness judge returned an invalid score: 0.6"
    else:
        raise AssertionError("Expected an invalid correctness score to fail")
