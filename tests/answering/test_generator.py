from arxiv_rag.answering import __main__ as answering_cli, chat_model
from arxiv_rag.answering import generator, renderer
from arxiv_rag.retrieval import BuiltContext, Citation, RetrievalContext


class RecordingModel(chat_model.ChatModel):
    def __init__(self, answer: str):
        self.answer = answer
        self.prompt = None

    def invoke(self, prompt: str) -> chat_model.ChatResponse:
        self.prompt = prompt
        return chat_model.ChatResponse(content=self.answer)


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


def test_chat_model_uses_env_model_and_cloudflare_headers(monkeypatch):
    captured_options = {}
    headers = {
        "CF-Access-Client-Id": "client-id",
        "CF-Access-Client-Secret": "client-secret",
    }

    monkeypatch.setenv("GENERATOR_MODEL", "generator-from-env")
    monkeypatch.setattr(chat_model, "get_ollama_connection", lambda: ("https://ollama.test", headers))
    monkeypatch.setattr(chat_model, "ChatOllama", lambda **options: captured_options.update(options) or object())

    chat_model.OllamaChatModel._get_chat_model()

    assert captured_options["model"] == "generator-from-env"
    assert captured_options["base_url"] == "https://ollama.test"
    assert captured_options["client_kwargs"] == {"headers": headers}


def test_generate_answer_returns_normal_text_with_valid_inline_citations():
    model = RecordingModel(
        "Transformers use attention [P1]. The model reached 28.4 BLEU [P2]."
    )

    answer = generator.generate_answer("How does it work?", _context(), model)

    assert answer == model.answer
    assert model.prompt is not None
    assert "Question:\nHow does it work?" in model.prompt
    assert "[P1]" in model.prompt
    assert "Answer directly and reply in a clear and formatted Markdown" in model.prompt
    assert "Use only passage IDs" in model.prompt
    assert "write separate markers with a space: [P1] [P2]" in model.prompt
    assert "Do not write [P1, P2] or [P1,P2]" in model.prompt


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


class FakeRetriever:
    def __init__(self, built: BuiltContext):
        self.built = built
        self.questions = []

    def retrieve_context_with_details(self, question: str) -> BuiltContext:
        self.questions.append(question)
        return self.built


def _built_context() -> BuiltContext:
    return BuiltContext(
        context=_context(),
        passages_by_id={"P1": "Transformers use attention.", "P2": "The model reached 28.4 BLEU."},
    )


def _fake_pipeline(monkeypatch) -> FakeRetriever:
    retriever = FakeRetriever(_built_context())
    monkeypatch.setattr(answering_cli, "PaperRetriever", lambda: retriever)
    monkeypatch.setattr(answering_cli, "generate_answer", lambda question, supplied_context: "Answer [P1].")
    return retriever


def test_command_line_entry_asks_a_question_and_prints_the_answer(monkeypatch, capsys):
    _fake_pipeline(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda _: "How does it work?")

    answering_cli.main()

    output = capsys.readouterr().out
    assert "Answer [1]." in output
    assert "Sources:\n[1] paper — Introduction\nhttps://arxiv.org/html/paper#S1" in output


def test_answer_question_returns_the_answer_with_its_evidence(monkeypatch):
    retriever = _fake_pipeline(monkeypatch)

    result = answering_cli.answer_question("How does it work?")

    assert retriever.questions == ["How does it work?"]
    assert result.answer == "Answer [P1]."
    assert result.context is retriever.built.context
    assert result.passages_by_id == retriever.built.passages_by_id


def test_answer_question_mints_a_new_thread_id_for_every_question(monkeypatch):
    _fake_pipeline(monkeypatch)

    first = answering_cli.answer_question("How does it work?")
    second = answering_cli.answer_question("How does it work?")

    assert first.thread_id != second.thread_id


def test_answer_question_keeps_a_supplied_thread_id(monkeypatch):
    _fake_pipeline(monkeypatch)

    result = answering_cli.answer_question("How does it work?", thread_id="conversation-1")

    assert result.thread_id == "conversation-1"


def test_answer_question_uses_a_supplied_retriever(monkeypatch):
    _fake_pipeline(monkeypatch)
    supplied = FakeRetriever(_built_context())

    result = answering_cli.answer_question("How does it work?", retriever=supplied)

    assert supplied.questions == ["How does it work?"]
    assert result.context is supplied.built.context


def test_traceable_wrappers_still_return_plain_application_values():
    model = RecordingModel("Transformers use attention [P1].")
    context = _context()

    answer = generator.generate_answer("How does it work?", context, model)
    rendered = renderer.render_answer(answer, context.citations, clickable=False)

    assert answer == model.answer
    assert rendered.startswith("Transformers use attention [1].")
