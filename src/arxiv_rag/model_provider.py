import os

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI

from arxiv_rag.util import application_config


def _cloudflare_access_headers(environment_prefix: str = "CF-ACCESS") -> dict[str, str]:
    return {
        "CF-Access-Client-Id": os.environ[f"{environment_prefix}-CLIENT-ID"],
        "CF-Access-Client-Secret": os.environ[f"{environment_prefix}-CLIENT-SECRET"],
    }


def get_generator_model_name() -> str:
    """Return the configured response-model name."""
    return application_config()["chat"]["model"]


def get_judge_model_name() -> str:
    """Return the configured evaluator-model name."""
    return application_config()["chat"]["model"]


def get_chat_model(model_name: str | None = None) -> BaseChatModel:
    """Create the OpenAI-compatible vLLM chat model."""
    config = application_config()["chat"]
    return ChatOpenAI(
        model=model_name or config["model"],
        base_url=config["base_url"],
        api_key=config["api_key"],
        temperature=config["temperature"],
        default_headers=_cloudflare_access_headers(),
    )


def get_judge_model(model_name: str | None = None) -> BaseChatModel:
    """Create the OpenAI-compatible vLLM evaluator model."""
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0)


def get_embeddings() -> Embeddings:
    """Create the remote Ollama embedding model."""
    config = application_config()["embeddings"]
    return OllamaEmbeddings(
        model=config["model"],
        base_url=config["base_url"],
        client_kwargs={"headers": _cloudflare_access_headers("OLLAMA-CF-ACCESS")},
    )
