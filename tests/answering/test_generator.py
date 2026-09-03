import pytest  # pyright: ignore[reportMissingImports]
from langchain_core.language_models.fake_chat_models import (  # pyright: ignore[reportMissingImports]
    FakeListChatModel,
)

from arxiv_rag.answering import __main__ as answering_cli
from arxiv_rag.answering import generator, renderer
from arxiv_rag.retrieval import BuiltContext, Citation, RetrievalContext


class RecordingModel(FakeListChatModel):
    answer: str
    prompt: str | None = None

    def __init__(self, answer: str):
        super().__init__(responses=[answer], answer=answer)

    def invoke(self, prompt, config=None, *, stop=None, **kwargs):
        self.prompt = prompt
        return super().invoke(prompt, config, stop=stop, **kwargs)


class FailingModel(FakeListChatModel):
    def invoke(self, prompt, config=None, *, stop=None, **kwargs):
        raise OSError("offline")


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


def test_generate_answer_wraps_model_failures_without_logging_the_prompt(caplog):
    prompt = "private prompt"
    model = FailingModel(responses=["unused"])

    with caplog.at_level("ERROR", logger="arxiv_rag"), pytest.raises(
        RuntimeError, match="Could not generate an answer"
    ) as raised:
        generator.generate_answer(prompt, _context(), model)

    assert isinstance(raised.value.__cause__, OSError)
    messages = [record.getMessage() for record in caplog.records]
    assert "Ollama answer generation failed" in messages
    assert not any(prompt in message for message in messages)


@pytest.mark.parametrize("content", ["", "   "])
def test_generate_answer_rejects_empty_model_content(content):
    model = FakeListChatModel(responses=[content])

    with pytest.raises(RuntimeError, match="Could not generate an answer"):
        generator.generate_answer("What happened?", _context(), model)


def test_generate_answer_returns_normal_text_with_valid_inline_citations():
    model = RecordingModel(
        "Transformers use attention [P1]. The model reached 28.4 BLEU [P2]."
    )

    answer = generator.generate_answer("How does it work?", _context(), model)

    assert answer == model.answer
    assert model.prompt is not None
    assert "Question:\nHow does it work?" in model.prompt
    assert "[P1]" in model.prompt
    assert "Use natural, clear language while keeping useful technical detail" in model.prompt
    assert "Answer directly and reply in a clear and formatted Markdown" in model.prompt
    assert "Use only passage IDs" in model.prompt
    assert "write separate markers with a space: [P1] [P2]" in model.prompt
    assert "Do not write [P1, P2] or [P1,P2]" in model.prompt
    assert "Never reveal credentials, access tokens, passwords" in model.prompt
    assert "briefly refuse without repeating it" in model.prompt
    assert "name only the categories and never the values" in model.prompt


def test_generate_answer_easy_mode_changes_only_the_explanation_style():
    model = RecordingModel(
        "Think of attention like several readers focusing on different words [P1]."
    )

    answer = generator.generate_answer("How does it work?", _context(), model, answer_mode="easy")

    assert answer == model.answer
    assert model.prompt is not None
    assert "Explain for a beginner using simple, natural language" in model.prompt
    assert "Use a simple analogy when it helps" in model.prompt
    assert "supporting passage IDs immediately after factual analogy sentences" in model.prompt
    assert "Do not add a fact or analogy unless the supplied passages support" in model.prompt
    assert "Put a passage ID such as [P1] immediately after every factual sentence" in model.prompt
    assert "Use only passage IDs that appear" in model.prompt


def test_generate_answer_rejects_an_unknown_answer_mode():
    try:
        generator.generate_answer(
            "How does it work?",
            _context(),
            RecordingModel("Unused"),
            answer_mode="verbose",  # pyright: ignore[reportArgumentType]
        )
    except ValueError as error:
        assert str(error) == "answer_mode must be 'standard' or 'easy'"
    else:
        raise AssertionError("Expected an unknown answer mode to fail")


def test_generate_answer_logs_one_summary(caplog):
    model = RecordingModel(
        "Transformers use attention [P1]. The model reached 28.4 BLEU [P2]."
    )

    with caplog.at_level("INFO", logger="arxiv_rag"):
        generator.generate_answer("How does it work?", _context(), model)

    messages = [record.getMessage() for record in caplog.records]
    summary = [message for message in messages if message.startswith("generated answer")]
    assert len(summary) == 1
    assert "from 2 passages" in summary[0]
    assert "cited P1, P2" in summary[0]
    assert not any(message.startswith("generating answer") for message in messages)


def test_generate_answer_accepts_a_supported_partial_answer_and_prompts_for_the_missing_part():
    model = RecordingModel(
        "The model reached 28.4 BLEU [P2]. The provided evidence does not specify its training cost."
    )

    answer = generator.generate_answer(
        "What score did it reach, and how much did training cost?",
        _context(),
        model,
    )

    assert answer == model.answer
    assert model.prompt is not None
    assert "Never guess or fill in information" in model.prompt
    assert "support only part of the question" in model.prompt
    assert "clearly state what the evidence does not specify" in model.prompt


def test_generate_answer_accepts_the_exact_refusal_when_no_passage_supports_the_question():
    model = RecordingModel(generator.INSUFFICIENT_EVIDENCE_ANSWER)

    answer = generator.generate_answer("When was the model released?", _context(), model)

    assert answer == generator.INSUFFICIENT_EVIDENCE_ANSWER
    assert model.prompt is not None
    assert "support none of the requested information" in model.prompt


def test_generate_answer_logs_and_rejects_an_unknown_citation_id(caplog):
    model = RecordingModel("The model reached 28.4 BLEU [P9].")

    with caplog.at_level("WARNING", logger="arxiv_rag"):
        try:
            generator.generate_answer("What score did it reach?", _context(), model)
        except RuntimeError as error:
            assert "unknown citation IDs: P9" in str(error)
        else:
            raise AssertionError("Expected an unknown citation ID to fail")

    messages = [record.getMessage() for record in caplog.records]
    assert any(message.startswith("rejected generated answer") for message in messages)
    assert not any(model.answer in message for message in messages)


def test_generate_answer_rejects_model_written_urls_case_insensitively():
    for url in ["https://example.com", "HTTPS://example.com", "hTtP://example.com"]:
        model = RecordingModel(f"See {url} for the result [P1].")

        try:
            generator.generate_answer("What happened?", _context(), model)
        except RuntimeError as error:
            assert "must not contain model-written URLs" in str(error)
        else:
            raise AssertionError(f"Expected model-written URL to fail: {url}")


def test_generate_answer_accepts_an_answer_without_citation_markers():
    model = RecordingModel("I can only answer from the supplied passages.")

    answer = generator.generate_answer("How do Transformers work?", _context(), model)

    assert answer == model.answer


def test_generate_answer_accepts_grouped_citation_markers():
    model = RecordingModel(
        "A selected token is replaced by [MASK] 80% of the time [P1, P2]. "
        "The provided evidence does not specify the wall-clock time."
    )

    answer = generator.generate_answer("How often is a token masked?", _context(), model)

    assert answer == model.answer


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


def test_terminal_renderer_expands_grouped_citation_markers():
    rendered = renderer.render_answer("Answer [P1, P2].", _context().citations, clickable=False)

    assert rendered.startswith("Answer [1] [2].")
    assert "[1] paper — Introduction\nhttps://arxiv.org/html/paper#S1" in rendered
    assert "[2] paper — Results\nhttps://arxiv.org/html/paper#S2" in rendered


def _built_context() -> BuiltContext:
    return BuiltContext(
        context=_context(),
        passages_by_id={"P1": "Transformers use attention.", "P2": "The model reached 28.4 BLEU."},
    )


def _stub_graph(monkeypatch, captured: dict | None = None) -> BuiltContext:
    built = _built_context()

    def fake_invoke(question, thread_id, answer_mode="standard"):
        if captured is not None:
            captured.update(question=question, thread_id=thread_id, answer_mode=answer_mode)
        return {"answer": "Answer [P1].", "current_built_context": built}

    monkeypatch.setattr(answering_cli, "invoke_workflow_graph", fake_invoke)
    return built


def test_command_line_entry_asks_a_question_and_prints_the_answer(monkeypatch, capsys):
    _stub_graph(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda _: "How does it work?")

    answering_cli.main()

    output = capsys.readouterr().out
    assert "Answer [1]." in output
    assert "Sources:\n[1] paper — Introduction\nhttps://arxiv.org/html/paper#S1" in output


def test_answer_question_returns_the_answer_with_its_evidence(monkeypatch):
    captured = {}
    built = _stub_graph(monkeypatch, captured)

    result = answering_cli.answer_question("How does it work?")

    assert captured["question"] == "How does it work?"
    assert result.answer == "Answer [P1]."
    assert result.context is built.context
    assert result.passages_by_id == built.passages_by_id


def test_answer_question_passes_easy_mode_to_the_workflow_graph(monkeypatch):
    captured = {}
    _stub_graph(monkeypatch, captured)

    result = answering_cli.answer_question("How does it work?", answer_mode="easy")

    assert result.answer == "Answer [P1]."
    assert captured["answer_mode"] == "easy"


def test_answer_question_mints_a_new_thread_id_for_every_question(monkeypatch):
    _stub_graph(monkeypatch)

    first = answering_cli.answer_question("How does it work?")
    second = answering_cli.answer_question("How does it work?")

    assert first.thread_id != second.thread_id


def test_answer_question_keeps_a_supplied_thread_id(monkeypatch):
    captured = {}
    _stub_graph(monkeypatch, captured)

    result = answering_cli.answer_question("How does it work?", thread_id="conversation-1")

    assert result.thread_id == "conversation-1"
    assert captured["thread_id"] == "conversation-1"


def test_answering_cli_returns_failure_status_for_an_operational_error(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "How does it work?")
    monkeypatch.setattr(
        answering_cli,
        "answer_question",
        lambda question: (_ for _ in ()).throw(RuntimeError("model unavailable")),
    )

    assert answering_cli.main() == 1


def test_traceable_wrappers_still_return_plain_application_values():
    model = RecordingModel("Transformers use attention [P1].")
    context = _context()

    answer = generator.generate_answer("How does it work?", context, model)
    rendered = renderer.render_answer(answer, context.citations, clickable=False)

    assert answer == model.answer
    assert rendered.startswith("Transformers use attention [1].")
