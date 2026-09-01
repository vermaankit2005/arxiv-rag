import os

from dotenv import load_dotenv


def get_ollama_connection() -> tuple[str, dict[str, str]]:
    """Load the Ollama URL and Cloudflare Access headers from the environment."""
    load_dotenv()

    required_names = (
        "OLLAMA_BASE_URL",
        "CF-ACCESS-CLIENT-ID",
        "CF-ACCESS-CLIENT-SECRET",
    )
    missing_names = [name for name in required_names if not os.environ.get(name)]
    if missing_names:
        missing = ", ".join(missing_names)
        raise RuntimeError(f"Missing Ollama configuration in .env: {missing}.")

    return os.environ["OLLAMA_BASE_URL"], {
        "CF-Access-Client-Id": os.environ["CF-ACCESS-CLIENT-ID"],
        "CF-Access-Client-Secret": os.environ["CF-ACCESS-CLIENT-SECRET"],
    }
