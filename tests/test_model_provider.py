from arxiv_rag import model_provider


def _write_config(path):
    path.write_text(
        """---
chat:
  provider: openai
  base_url: https://vllm.test/v1
  model: configured-model
  api_key: unused
  temperature: 0
  default_headers:
    CF-Access-Client-Id: ${CF-ACCESS-CLIENT-ID}
    CF-Access-Client-Secret: ${CF-ACCESS-CLIENT-SECRET}
embeddings:
  provider: ollama
  base_url: http://localhost:11434
  model: qwen3-embedding:4b
""",
        encoding="utf-8",
    )


def test_get_model_provider_uses_yaml_configuration_and_cloudflare_headers(monkeypatch, tmp_path):
    captured_options = {}
    sentinel = object()
    config_file = tmp_path / "config.yaml"
    _write_config(config_file)
    monkeypatch.setattr(model_provider, "CONFIG_FILE", config_file)
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


def test_get_embeddings_uses_local_ollama_yaml_configuration(monkeypatch, tmp_path):
    captured_options = {}
    sentinel = object()
    config_file = tmp_path / "config.yaml"
    _write_config(config_file)
    monkeypatch.setattr(model_provider, "CONFIG_FILE", config_file)
    monkeypatch.setattr(
        model_provider,
        "OllamaEmbeddings",
        lambda **options: captured_options.update(options) or sentinel,
    )

    result = model_provider.get_embeddings()

    assert result is sentinel
    assert captured_options == {
        "model": "qwen3-embedding:4b",
        "base_url": "http://localhost:11434",
    }
