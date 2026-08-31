import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama  # pyright: ignore[reportMissingImports]

load_dotenv()

def build_judge_model(model_name: str) -> ChatOllama:
    return ChatOllama(
        model=model_name,
        base_url=os.environ["OLLAMA_BASE_URL"],
        temperature=0,
    )
