from arxiv_rag.model_provider import get_chat_model, get_judge_model


def build_judge_model(model_name: str | None = None):
    """Create the configured OpenAI-compatible evaluator model."""
    return get_chat_model(model_name)
    # return get_judge_model(model_name)