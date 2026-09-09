import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings  # pyright: ignore[reportMissingImports]
from langchain_core.language_models import BaseChatModel  # pyright: ignore[reportMissingImports]
from langchain_ollama import OllamaEmbeddings  # pyright: ignore[reportMissingImports]
from langchain_openai import ChatOpenAI  # pyright: ignore[reportMissingImports]

APPLICATION_FILE = Path(__file__).parents[2] / "application.yaml"


def _application_config() -> dict[str, Any]:
    load_dotenv()
    with APPLICATION_FILE.open(encoding="utf-8") as application_file:
        return yaml.safe_load(application_file)


def _cloudflare_access_headers() -> dict[str, str]:
    return {
        "CF-Access-Client-Id": os.environ["CF-ACCESS-CLIENT-ID"],
        "CF-Access-Client-Secret": os.environ["CF-ACCESS-CLIENT-SECRET"],
    }


def get_generator_model_name() -> str:
    """Return the configured response-model name."""
    return _application_config()["chat"]["model"]


def get_judge_model_name() -> str:
    """Return the configured evaluator-model name."""
    return _application_config()["chat"]["model"]


def get_chat_model(model_name: str | None = None) -> BaseChatModel:
    """Create the OpenAI-compatible vLLM chat model."""
    config = _application_config()["chat"]
    return ChatOpenAI(
        model=model_name or config["model"],
        base_url=config["base_url"],
        api_key=config["api_key"],
        temperature=config["temperature"],
        default_headers=_cloudflare_access_headers(),
    )


def get_embeddings() -> Embeddings:
    """Create the locally hosted Ollama embedding model."""
    config = _application_config()["embeddings"]
    return OllamaEmbeddings(model=config["model"], base_url=config["base_url"])
