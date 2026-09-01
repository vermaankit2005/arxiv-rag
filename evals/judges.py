from langchain_ollama import ChatOllama  # pyright: ignore[reportMissingImports]

from arxiv_rag.ollama_config import get_judge_model, get_ollama_connection


def build_judge_model(model_name: str | None = None) -> ChatOllama:
    base_url, headers = get_ollama_connection()
    return ChatOllama(
        model=model_name or get_judge_model(),
        base_url=base_url,
        temperature=0,
        client_kwargs={"headers": headers},
        reasoning=False,
        num_ctx=8192,
    )
