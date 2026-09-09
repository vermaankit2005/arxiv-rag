import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings  # pyright: ignore[reportMissingImports]
from langchain_core.language_models import BaseChatModel  # pyright: ignore[reportMissingImports]
from langchain_ollama import OllamaEmbeddings  # pyright: ignore[reportMissingImports]
from langchain_openai import ChatOpenAI  # pyright: ignore[reportMissingImports]

CONFIG_FILE = Path(__file__).parents[2] / "config.yaml"
ENVIRONMENT_VALUE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_-]*)\}$")


@dataclass(frozen=True)
class ChatModelConfig:
    base_url: str
    model: str
    api_key: str
    temperature: float
    headers: dict[str, str]


@dataclass(frozen=True)
class EmbeddingModelConfig:
    base_url: str
    model: str


def _required_string(section: dict[str, Any], name: str) -> str:
    value = section.get(name)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Missing {name} in {CONFIG_FILE}.")
    return value.strip()


def _expand_environment_value(value: str) -> str:
    match = ENVIRONMENT_VALUE.fullmatch(value)
    if not match:
        return value

    name = match.group(1)
    environment_value = os.environ.get(name, "").strip()
    if not environment_value:
        raise RuntimeError(f"Missing environment variable referenced by {CONFIG_FILE}: {name}.")
    return environment_value


def _load_config() -> dict[str, Any]:
    load_dotenv()
    if not CONFIG_FILE.exists():
        raise RuntimeError(f"Missing {CONFIG_FILE}. Copy config.example.yaml to config.yaml.")

    with CONFIG_FILE.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise RuntimeError(f"{CONFIG_FILE} must contain a YAML mapping.")
    return config


def get_chat_model_config() -> ChatModelConfig:
    config = _load_config()
    chat = config.get("chat")
    if not isinstance(chat, dict):
        raise RuntimeError(f"Missing chat configuration in {CONFIG_FILE}.")
    if _required_string(chat, "provider") != "openai":
        raise RuntimeError("Only the openai chat provider is supported.")

    headers = chat.get("default_headers", {})
    if not isinstance(headers, dict) or not all(
        isinstance(name, str) and isinstance(value, str) for name, value in headers.items()
    ):
        raise RuntimeError(f"chat.default_headers in {CONFIG_FILE} must map strings to strings.")

    temperature = chat.get("temperature", 0)
    if not isinstance(temperature, (int, float)):
        raise RuntimeError(f"chat.temperature in {CONFIG_FILE} must be a number.")
    try:
        parsed_temperature = float(temperature)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"chat.temperature in {CONFIG_FILE} must be a number.") from error

    return ChatModelConfig(
        base_url=_required_string(chat, "base_url"),
        model=_required_string(chat, "model"),
        api_key=_expand_environment_value(_required_string(chat, "api_key")),
        temperature=parsed_temperature,
        headers={name: _expand_environment_value(value) for name, value in headers.items()},
    )


def get_embedding_model_config() -> EmbeddingModelConfig:
    config = _load_config()
    embeddings = config.get("embeddings")
    if not isinstance(embeddings, dict):
        raise RuntimeError(f"Missing embeddings configuration in {CONFIG_FILE}.")
    if _required_string(embeddings, "provider") != "ollama":
        raise RuntimeError("Only the ollama embeddings provider is supported.")

    return EmbeddingModelConfig(
        base_url=_required_string(embeddings, "base_url"),
        model=_required_string(embeddings, "model"),
    )


def get_generator_model_name() -> str:
    """Return the configured response-model name."""
    return get_chat_model_config().model


def get_judge_model_name() -> str:
    """Return the configured evaluator-model name."""
    return get_chat_model_config().model


def get_chat_model(model_name: str | None = None) -> BaseChatModel:
    """Create the OpenAI-compatible vLLM chat model."""
    config = get_chat_model_config()
    return ChatOpenAI(
        model=model_name or config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        temperature=config.temperature,
        default_headers=config.headers,
    )


def get_embeddings() -> Embeddings:
    """Create the locally hosted Ollama embedding model."""
    config = get_embedding_model_config()
    return OllamaEmbeddings(model=config.model, base_url=config.base_url)
