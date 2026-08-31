from evals.answering import evaluate_generation_citation_support as citation_support


def _passage(citation_id: str, text: str) -> dict:
    return {
        "id": citation_id,
        "arxiv_id": "paper-v1",
        "location": f"#{citation_id}",
        "section_path": ["Results"],
        "text": text,
    }


class RecordingJudge:
    def __init__(self, scores: list[bool]):
        self.scores = iter(scores)
        self.calls = []

    def __call__(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        score = next(self.scores)
        return {
            "key": "citation_support_pair",
            "score": score,
            "comment": "The passage was checked.",
        }


def test_extracts_each_statement_and_its_attached_citation():
    answer = (
        "The decoder masks future positions [P2]. "
        "It also offsets output embeddings [P2] [P3]."
    )

    pairs = citation_support._extract_statement_citation_pairs(answer)

    assert pairs == [
        ("The decoder masks future positions", "P2"),
        ("It also offsets output embeddings", "P2"),
        ("It also offsets output embeddings", "P3"),
    ]


def test_extracts_citation_after_sentence_punctuation_and_markdown_bullet():
    answer = "## Answer\n- The model scored 91.3 F1. [P1]"

    pairs = citation_support._extract_statement_citation_pairs(answer)

    assert pairs == [("The model scored 91.3 F1", "P1")]


def test_scores_supported_pairs_over_all_cited_pairs():
    inputs = {
        "question": "What happened?",
        "context_passages": [
            _passage("P1", "The model scored 91.3 F1."),
            _passage("P2", "The model used four layers."),
        ],
    }
    outputs = {"answer": "The model scored 91.3 F1 [P1] [P2]."}
    judge = RecordingJudge([True, False])

    result = citation_support.evaluate_citation_support(inputs, outputs, judge)

    assert result["key"] == "citation_support"
    assert result["score"] == 0.5
    assert result["comment"].startswith("1/2 statement-citation pairs were supported.")
    assert judge.calls[0]["outputs"] == {"passage": "The model scored 91.3 F1."}
    assert judge.calls[1]["outputs"] == {"passage": "The model used four layers."}


def test_returns_zero_when_answer_has_no_statement_citation_pairs():
    result = citation_support.evaluate_citation_support(
        {"question": "What happened?", "context_passages": []},
        {"answer": "I do not know."},
        RecordingJudge([]),
    )

    assert result == {
        "key": "citation_support",
        "score": 0.0,
        "comment": "The answer contained no statement-citation pairs.",
    }


def test_rejects_unknown_citation_ids_before_calling_judge():
    judge = RecordingJudge([])

    try:
        citation_support.evaluate_citation_support(
            {"question": "What happened?", "context_passages": [_passage("P1", "Evidence")]},
            {"answer": "A claim [P9]."},
            judge,
        )
    except ValueError as error:
        assert str(error) == "Answer used unknown citation IDs: P9"
    else:
        raise AssertionError("Expected an unknown citation ID to fail")

    assert judge.calls == []


def test_generation_context_preserves_frozen_passage_ids():
    context = citation_support._build_retrieval_context([_passage("P7", "Frozen evidence")])

    assert "[P7]" in context.text
    assert list(context.citations) == ["P7"]
