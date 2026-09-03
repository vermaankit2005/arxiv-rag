import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_core.language_models import (  # pyright: ignore[reportMissingImports]
    BaseChatModel,
)
from langchain_groq import ChatGroq  # pyright: ignore[reportMissingImports]
from langchain_ollama import ChatOllama  # pyright: ignore[reportMissingImports]

from arxiv_rag.ollama_config import get_generator_model, get_ollama_connection


def get_chat_model() -> BaseChatModel:
    base_url, headers = get_ollama_connection()
    return ChatOllama(
        model=get_generator_model(),
        base_url=base_url,
        temperature=0,
        client_kwargs={"headers": headers},
        reasoning=False,
        num_ctx=8192,
    )

#  ---------------- Throwaway Groq model configuration for testing ----------------

@dataclass(frozen=True)
class GroqChatModelConfig:
    model: str
    api_key: str = field(repr=False)


def get_groq_chat_model_config() -> GroqChatModelConfig:
    """Load the Groq chat model configuration from the environment."""
    load_dotenv()
    env_names = ("GROQ_MODEL", "GROQ_API_KEY")
    values = {name: os.environ.get(name, "").strip() for name in env_names}
    missing_names = [name for name, value in values.items() if not value]
    if missing_names:
        missing = ", ".join(missing_names)
        raise RuntimeError(f"Missing Groq configuration in .env: {missing}.")

    return GroqChatModelConfig(
        model=values["GROQ_MODEL"],
        api_key=values["GROQ_API_KEY"],
    )


def get_groq_chat_model() -> BaseChatModel:
    """Create the separately configured Groq chat model."""
    config = get_groq_chat_model_config()
    return ChatGroq(
        model=config.model,
        api_key=config.api_key,
        temperature=0,
    )
