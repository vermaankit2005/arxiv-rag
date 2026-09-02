from types import SimpleNamespace

import pytest  # pyright: ignore[reportMissingImports]

from arxiv_rag.answering.chat_model import OllamaChatModel


class FakeOllama:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error

    def invoke(self, prompt):
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.content)


def _use_fake_ollama(monkeypatch, fake):
    monkeypatch.setattr(OllamaChatModel, "_get_chat_model", staticmethod(lambda: fake))


def test_ollama_failure_is_logged_without_the_prompt(monkeypatch, caplog):
    _use_fake_ollama(monkeypatch, FakeOllama(error=OSError("offline")))
    prompt = "private prompt"

    with caplog.at_level("ERROR", logger="arxiv_rag"), pytest.raises(
        RuntimeError, match="Could not generate an answer"
    ) as raised:
        OllamaChatModel().invoke(prompt)

    assert isinstance(raised.value.__cause__, OSError)
    messages = [record.getMessage() for record in caplog.records]
    assert "Ollama answer generation failed" in messages
    assert not any(prompt in message for message in messages)


@pytest.mark.parametrize("content", [None, "", "   "])
def test_empty_or_invalid_ollama_content_is_rejected(monkeypatch, content):
    _use_fake_ollama(monkeypatch, FakeOllama(content=content))

    with pytest.raises(RuntimeError, match="Could not generate an answer"):
        OllamaChatModel().invoke("private prompt")
