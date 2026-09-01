import os

from dotenv import load_dotenv


def _required_env(name: str) -> str:
    load_dotenv()
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing Ollama configuration in .env: {name}.")
    return value


def get_ollama_connection() -> tuple[str, dict[str, str]]:
    """Load the Ollama URL and Cloudflare Access headers from the environment."""
    load_dotenv()

    required_names = (
        "OLLAMA_BASE_URL",
        "CF-ACCESS-CLIENT-ID",
        "CF-ACCESS-CLIENT-SECRET",
    )
    missing_names = [name for name in required_names if not os.environ.get(name, "").strip()]
    if missing_names:
        missing = ", ".join(missing_names)
        raise RuntimeError(f"Missing Ollama configuration in .env: {missing}.")

    return os.environ["OLLAMA_BASE_URL"], {
        "CF-Access-Client-Id": os.environ["CF-ACCESS-CLIENT-ID"],
        "CF-Access-Client-Secret": os.environ["CF-ACCESS-CLIENT-SECRET"],
    }


def get_generator_model() -> str:
    """Load the generator chat model name from the environment."""
    return _required_env("GENERATOR_MODEL")


def get_judge_model() -> str:
    """Load the evaluation-judge chat model name from the environment."""
    return _required_env("JUDGE_MODEL")
