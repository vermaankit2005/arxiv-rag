from evals import judges


def test_build_judge_model_uses_default_model_and_cloudflare_headers(monkeypatch):
    captured_options = {}
    expected_model = object()
    headers = {
        "CF-Access-Client-Id": "client-id",
        "CF-Access-Client-Secret": "client-secret",
    }

    monkeypatch.setattr(judges, "get_ollama_connection", lambda: ("https://ollama.test", headers))
    monkeypatch.setattr(
        judges,
        "ChatOllama",
        lambda **options: captured_options.update(options) or expected_model,
    )

    model = judges.build_judge_model()

    assert model is expected_model
    assert captured_options == {
        "model": judges.DEFAULT_JUDGE_MODEL,
        "base_url": "https://ollama.test",
        "temperature": 0,
        "client_kwargs": {"headers": headers},
        "reasoning": False,
        "num_ctx": 8192,
    }


def test_build_judge_model_uses_caller_model(monkeypatch):
    captured_options = {}

    monkeypatch.setattr(judges, "get_ollama_connection", lambda: ("https://ollama.test", {}))
    monkeypatch.setattr(
        judges,
        "ChatOllama",
        lambda **options: captured_options.update(options) or object(),
    )

    judges.build_judge_model("custom-model:latest")

    assert captured_options["model"] == "custom-model:latest"
