from evals.answering import evaluate_generation_naturalness as naturalness


class RecordingJudge:
    def __init__(self, score, comment: str | None = None):
        self.score = score
        self.comment = comment
        self.calls = []

    def __call__(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        result = {"key": "naturalness", "score": self.score}
        if self.comment is not None:
            result["comment"] = self.comment
        return result


def test_returns_the_restricted_naturalness_score_and_comment():
    judge = RecordingJudge(0.5, "Readable, but noticeably mechanical.")

    result = naturalness.evaluate_naturalness(
        {"question": "How does the method work?"},
        {"answer": "First, the method retrieves evidence [P1]."},
        judge,
    )

    assert result == {
        "key": "naturalness",
        "score": 0.5,
        "comment": "Readable, but noticeably mechanical.",
    }
    assert judge.calls == [{
        "inputs": {"question": "How does the method work?"},
        "outputs": {"answer": "First, the method retrieves evidence [P1]."},
    }]


def test_rejects_a_score_outside_the_five_allowed_values():
    try:
        naturalness.evaluate_naturalness(
            {"question": "Question"},
            {"answer": "Answer"},
            RecordingJudge(0.6),
        )
    except ValueError as error:
        assert str(error) == "Naturalness judge returned an invalid score: 0.6"
    else:
        raise AssertionError("Expected an invalid naturalness score to fail")


def test_rubric_keeps_naturalness_separate_from_answer_quality():
    assert "Do not score factual correctness" in naturalness.NATURALNESS_PROMPT
    assert "Ignore citation markers" in naturalness.NATURALNESS_PROMPT
    assert "Do not reward\nchattiness" in naturalness.NATURALNESS_PROMPT
