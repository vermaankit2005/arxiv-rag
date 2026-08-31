from dataclasses import dataclass

from arxiv_rag.answering import __main__ as answering_cli
from arxiv_rag.answering import generator, renderer
from arxiv_rag.retrieval import Citation, RetrievalContext


@dataclass
class FakeResponse(generator.ChatResponse):
    content: str


class RecordingModel(generator.ChatModel):
    def __init__(self, answer: str):
        self.answer = answer
        self.prompt = None

    def invoke(self, prompt: str) -> FakeResponse:
        self.prompt = prompt
        return FakeResponse(self.answer)


def _context() -> RetrievalContext:
    return RetrievalContext(
        text=(
            "[P1]\nSection: Introduction\nText: Transformers use attention.\n\n"
            "---\n\n[P2]\nSection: Results\nText: The model reached 28.4 BLEU."
        ),
        citations={
            "P1": Citation(
                label="paper — Introduction",
                url="https://arxiv.org/html/paper#S1",
            ),
            "P2": Citation(
                label="paper — Results",
                url="https://arxiv.org/html/paper#S2",
            ),
        },
    )


def test_generator_uses_gemma_model_name():
    assert generator.MODEL_NAME == "gemma4:26b"


def test_generate_answer_returns_normal_text_with_valid_inline_citations():
    model = RecordingModel(
        "Transformers use attention [P1]. The model reached 28.4 BLEU [P2]."
    )

    answer = generator.generate_answer("How does it work?", _context(), model)

    assert answer == model.answer
    assert model.prompt is not None
    assert "Question:\nHow does it work?" in model.prompt
    assert "[P1]" in model.prompt
    assert "Answer directly and use clear Markdown" in model.prompt
    assert "Use only passage IDs" in model.prompt


def test_generate_answer_rejects_an_unknown_citation_id():
    model = RecordingModel("The model reached 28.4 BLEU [P9].")

    try:
        generator.generate_answer("What score did it reach?", _context(), model)
    except RuntimeError as error:
        assert "unknown citation IDs: P9" in str(error)
    else:
        raise AssertionError("Expected an unknown citation ID to fail")


def test_generate_answer_rejects_model_written_urls_case_insensitively():
    for url in ["https://example.com", "HTTPS://example.com", "hTtP://example.com"]:
        model = RecordingModel(f"See {url} for the result [P1].")

        try:
            generator.generate_answer("What happened?", _context(), model)
        except RuntimeError as error:
            assert "must not contain model-written URLs" in str(error)
        else:
            raise AssertionError(f"Expected model-written URL to fail: {url}")


def test_generate_answer_rejects_an_uncited_answer():
    model = RecordingModel("Transformers use attention.")

    try:
        generator.generate_answer("How do Transformers work?", _context(), model)
    except RuntimeError as error:
        assert "must contain at least one citation" in str(error)
    else:
        raise AssertionError("Expected an uncited answer to fail")


def test_generate_answer_returns_unknown_without_calling_model_for_empty_context():
    model = RecordingModel("This must not be used.")
    context = RetrievalContext(text="", citations={})

    answer = generator.generate_answer("What happened?", context, model)

    assert answer == generator.INSUFFICIENT_EVIDENCE_ANSWER
    assert model.prompt is None


def test_generate_answer_rejects_an_empty_question():
    try:
        generator.generate_answer("   ", _context(), RecordingModel("Unused"))
    except ValueError as error:
        assert str(error) == "question must not be empty"
    else:
        raise AssertionError("Expected an empty question to fail")


def test_terminal_renderer_creates_a_hidden_clickable_citation():
    rendered = renderer.render_answer("Answer [P1].", _context().citations, clickable=True)

    assert rendered == "Answer \033]8;;https://arxiv.org/html/paper#S1\033\\[1]\033]8;;\033\\."
    assert "Sources:" not in rendered


def test_command_line_entry_asks_a_question_and_prints_the_answer(monkeypatch, capsys):
    context = _context()

    class FakeRetriever:
        def retrieve_context(self, question: str) -> RetrievalContext:
            assert question == "How does it work?"
            return context

    monkeypatch.setattr("builtins.input", lambda _: "How does it work?")
    monkeypatch.setattr(answering_cli, "PaperRetriever", FakeRetriever)
    monkeypatch.setattr(answering_cli, "generate_answer", lambda question, supplied_context: "Answer [P1].")

    answering_cli.main()

    output = capsys.readouterr().out
    assert "Answer [1]." in output
    assert "Sources:\n[1] paper — Introduction\nhttps://arxiv.org/html/paper#S1" in output
