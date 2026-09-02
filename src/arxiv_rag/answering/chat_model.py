from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_ollama import ChatOllama  # pyright: ignore[reportMissingImports]

from arxiv_rag.logging import get_logger
from arxiv_rag.ollama_config import get_ollama_connection, get_generator_model

log = get_logger(__name__)


@dataclass(frozen=True)
class ChatResponse:
    content: str


class ChatModel(ABC):
    @abstractmethod
    def invoke(self, prompt: str) -> ChatResponse:
        ...


class OllamaChatModel(ChatModel):
    @staticmethod
    def _get_chat_model() -> ChatOllama:
        base_url, headers = get_ollama_connection()

        return ChatOllama(
            model=get_generator_model(),
            base_url=base_url,
            temperature=0,
            client_kwargs={"headers": headers},
            reasoning=False,
            num_ctx=8192,
        )

    def invoke(self, prompt: str) -> ChatResponse:
        chat_model = self._get_chat_model()
        try:
            response = chat_model.invoke(prompt)
        except Exception as error:
            log.exception("Ollama answer generation failed")
            raise RuntimeError("Could not generate an answer.") from error

        if not isinstance(response.content, str) or not response.content.strip():
            log.error("Ollama returned an empty or invalid answer")
            raise RuntimeError("Could not generate an answer.")
        return ChatResponse(content=response.content)


def get_chat_model() -> ChatModel:
    return OllamaChatModel()
