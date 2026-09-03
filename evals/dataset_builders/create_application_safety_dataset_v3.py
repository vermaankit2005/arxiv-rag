"""Create the expanded application-safety v3 datasets in LangSmith."""

import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client  # pyright: ignore[reportMissingImports]

from evals.application.safety import LANGSMITH_DATASET_NAMES
from evals.dataset_builders.create_application_safety_dataset import (
    build_langsmith_examples,
    validate_examples,
)

ROOT = Path(__file__).parents[2]
DATASET_PATH = ROOT / "evals" / "dataset" / "application_safety_dataset_v3.json"
EXPECTED_COUNTS = dict.fromkeys(LANGSMITH_DATASET_NAMES, 10)


def load_examples() -> list[dict]:
    """Read and validate the expanded local v3 safety cases."""
    try:
        examples = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read {DATASET_PATH}") from error

    validate_examples(examples, EXPECTED_COUNTS)
    return examples


def create_application_safety_datasets(client: Client | None = None) -> None:
    """Publish four immutable v3 metric datasets with ten cases each."""
    load_dotenv()
    langsmith_client = client or Client()

    existing_names = [
        name
        for name in LANGSMITH_DATASET_NAMES.values()
        if langsmith_client.has_dataset(dataset_name=name)
    ]
    if existing_names:
        raise RuntimeError(
            f"LangSmith dataset(s) already exist: {', '.join(existing_names)}. "
            "Delete the incomplete v3 publication before retrying."
        )

    examples = load_examples()
    for metric, dataset_name in LANGSMITH_DATASET_NAMES.items():
        metric_examples = build_langsmith_examples(examples, metric)
        langsmith_client.create_dataset(
            dataset_name,
            description=f"Expanded safety policy v2 cases for {metric}, dataset v3; 1 means pass.",
        )
        langsmith_client.create_examples(dataset_name=dataset_name, examples=metric_examples)
        print(f"Created LangSmith dataset: {dataset_name} ({len(metric_examples)} examples)")


if __name__ == "__main__":
    create_application_safety_datasets()
