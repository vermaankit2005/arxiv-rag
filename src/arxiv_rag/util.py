import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

APPLICATION_FILE = Path(__file__).parents[2] / "application.yaml"
RUNTIME_ENV_NAME = "ARXIV_RAG_RUNTIME"
EVAL_RUNTIME_NAME = "eval"
EVAL_HTML_CORPUS = "SAMPLE"
EVAL_VECTOR_STORE = "CHROMA"


def _apply_eval_runtime(config: dict[str, Any]) -> None:
    try:
        runtime = config["evals"]["runtime"]
        html_corpus = runtime["html_corpus"]
        vector_store = runtime["vector_store"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("Eval runtime configuration is missing from application.yaml.") from error

    if html_corpus != EVAL_HTML_CORPUS or vector_store != EVAL_VECTOR_STORE:
        raise RuntimeError("Eval runtime must use html_corpus=SAMPLE and vector_store=CHROMA.")

    try:
        config["loading"]["active"] = dict(runtime)
    except (KeyError, TypeError) as error:
        raise RuntimeError("Loading configuration is missing from application.yaml.") from error


def application_config() -> dict[str, Any]:
    load_dotenv()
    with APPLICATION_FILE.open(encoding="utf-8") as application_file:
        config = yaml.safe_load(application_file)

    if not isinstance(config, dict):
        raise RuntimeError("application.yaml must contain a mapping.")

    runtime_name = os.environ.get(RUNTIME_ENV_NAME, "").strip()
    if runtime_name:
        if runtime_name != EVAL_RUNTIME_NAME:
            raise RuntimeError(f"Unknown {RUNTIME_ENV_NAME} value: {runtime_name}")
        _apply_eval_runtime(config)

    return config


def activate_eval_runtime() -> dict[str, Any]:
    """Select and validate the fixed SAMPLE/CHROMA runtime for eval processes."""
    os.environ[RUNTIME_ENV_NAME] = EVAL_RUNTIME_NAME
    return application_config()
