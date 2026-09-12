from arxiv_rag import model_provider, util


def _write_config(path):
    path.write_text(
        """---
chat:
  base_url: https://vllm.test/v1
  model: configured-model
  api_key: unused
  temperature: 0
embeddings:
  base_url: https://ollama.test
  model: qwen3-embedding:4b
""",
        encoding="utf-8",
    )


def test_get_model_provider_uses_yaml_configuration_and_cloudflare_headers(monkeypatch, tmp_path):
    captured_options = {}
    sentinel = object()
    application_file = tmp_path / "application.yaml"
    _write_config(application_file)
    monkeypatch.setattr(util, "APPLICATION_FILE", application_file)
    monkeypatch.setenv("CF-ACCESS-CLIENT-ID", "client-id")
    monkeypatch.setenv("CF-ACCESS-CLIENT-SECRET", "client-secret")
    monkeypatch.setattr(
        model_provider,
        "ChatOpenAI",
        lambda **options: captured_options.update(options) or sentinel,
    )

    result = model_provider.get_chat_model()

    assert result is sentinel
    assert captured_options == {
        "model": "configured-model",
        "base_url": "https://vllm.test/v1",
        "api_key": "unused",
        "temperature": 0,
        "default_headers": {
            "CF-Access-Client-Id": "client-id",
            "CF-Access-Client-Secret": "client-secret",
        },
    }


def test_get_embeddings_uses_remote_ollama_configuration_and_cloudflare_headers(
    monkeypatch, tmp_path
):
    captured_options = {}
    sentinel = object()
    application_file = tmp_path / "application.yaml"
    _write_config(application_file)
    monkeypatch.setattr(util, "APPLICATION_FILE", application_file)
    monkeypatch.setenv("OLLAMA-CF-ACCESS-CLIENT-ID", "ollama-client-id")
    monkeypatch.setenv("OLLAMA-CF-ACCESS-CLIENT-SECRET", "ollama-client-secret")
    monkeypatch.setattr(
        model_provider,
        "OllamaEmbeddings",
        lambda **options: captured_options.update(options) or sentinel,
    )

    result = model_provider.get_embeddings()

    assert result is sentinel
    assert captured_options == {
        "model": "qwen3-embedding:4b",
        "base_url": "https://ollama.test",
        "client_kwargs": {
            "headers": {
                "CF-Access-Client-Id": "ollama-client-id",
                "CF-Access-Client-Secret": "ollama-client-secret",
            }
        },
    }
