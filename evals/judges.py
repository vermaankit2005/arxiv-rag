from arxiv_rag.answering.chat_model import get_chat_model


def build_judge_model(model_name: str | None = None):
    """Create the configured OpenAI-compatible evaluator model."""
    return get_chat_model(model_name)
