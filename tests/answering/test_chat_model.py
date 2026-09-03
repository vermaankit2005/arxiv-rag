from arxiv_rag.answering import chat_model


def test_get_chat_model_uses_env_model_and_cloudflare_headers(monkeypatch):
    captured_options = {}
    headers = {
        "CF-Access-Client-Id": "client-id",
        "CF-Access-Client-Secret": "client-secret",
    }
    sentinel = object()

    monkeypatch.setenv("GENERATOR_MODEL", "generator-from-env")
    monkeypatch.setattr(chat_model, "get_ollama_connection", lambda: ("https://ollama.test", headers))
    monkeypatch.setattr(
        chat_model,
        "ChatOllama",
        lambda **options: captured_options.update(options) or sentinel,
    )

    result = chat_model.get_chat_model()

    assert result is sentinel
    assert captured_options["model"] == "generator-from-env"
    assert captured_options["base_url"] == "https://ollama.test"
    assert captured_options["temperature"] == 0
    assert captured_options["client_kwargs"] == {"headers": headers}
    assert captured_options["reasoning"] is False
    assert captured_options["num_ctx"] == 8192
