from langchain_ollama import ChatOllama  # pyright: ignore[reportMissingImports]

from arxiv_rag.ollama_config import get_ollama_connection

DEFAULT_JUDGE_MODEL = "gemma4:26b"


def build_judge_model(model_name: str = DEFAULT_JUDGE_MODEL) -> ChatOllama:
    base_url, headers = get_ollama_connection()
    return ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=0,
        client_kwargs={"headers": headers},
        reasoning=False,
        num_ctx=8192,
    )
