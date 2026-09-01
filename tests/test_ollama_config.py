from arxiv_rag.ollama_config import get_ollama_connection


def test_get_ollama_connection_builds_cloudflare_access_headers(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.example.com")
    monkeypatch.setenv("CF-ACCESS-CLIENT-ID", "client-id")
    monkeypatch.setenv("CF-ACCESS-CLIENT-SECRET", "client-secret")

    base_url, headers = get_ollama_connection()

    assert base_url == "https://ollama.example.com"
    assert headers == {
        "CF-Access-Client-Id": "client-id",
        "CF-Access-Client-Secret": "client-secret",
    }


def test_get_ollama_connection_reports_missing_configuration(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "")
    monkeypatch.setenv("CF-ACCESS-CLIENT-ID", "")
    monkeypatch.setenv("CF-ACCESS-CLIENT-SECRET", "")

    try:
        get_ollama_connection()
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError("Expected missing Ollama configuration to fail")

    assert "OLLAMA_BASE_URL" in message
    assert "CF-ACCESS-CLIENT-ID" in message
    assert "CF-ACCESS-CLIENT-SECRET" in message
