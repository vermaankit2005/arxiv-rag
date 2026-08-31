"""Create the answer evidence-behavior dataset in LangSmith."""

import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).parents[2]
DATASET_PATH = ROOT / "evals" / "dataset" / "generation_evidence_behavior_dataset.json"
LANGSMITH_DATASET_NAME = "generation_evidence_behavior_dataset"


def load_examples() -> list[dict]:
    """Read the curated local examples."""
    try:
        examples = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read {DATASET_PATH}") from error

    if not isinstance(examples, list) or not examples:
        raise RuntimeError("Evidence-behavior dataset must be a non-empty list")
    return examples


def create_generation_evidence_behavior_dataset() -> None:
    """Publish the examples without replacing an existing LangSmith dataset."""
    load_dotenv()
    client = Client()

    if client.has_dataset(dataset_name=LANGSMITH_DATASET_NAME):
        raise RuntimeError(
            f"LangSmith dataset already exists: {LANGSMITH_DATASET_NAME}. "
            "Delete it explicitly before publishing a changed dataset."
        )

    examples = []
    for example in load_examples():
        examples.append({
            "inputs": {
                "question": example["question"],
                "context_passages": example["context_passages"],
            },
            "outputs": {
                "expected_behavior": example["expected_behavior"],
                "supported_facts": example["supported_facts"],
                "unsupported_parts": example["unsupported_parts"],
            },
            "metadata": {
                "example_id": example["id"],
                **example.get("metadata", {}),
            },
            "split": "test",
        })

    client.create_dataset(
        LANGSMITH_DATASET_NAME,
        description=(
            "Nine curated examples for fully supported, partially supported, and "
            "unsupported answer-generation behavior."
        ),
    )
    client.create_examples(dataset_name=LANGSMITH_DATASET_NAME, examples=examples)
    print(f"Created LangSmith dataset: {LANGSMITH_DATASET_NAME} ({len(examples)} examples)")


if __name__ == "__main__":
    create_generation_evidence_behavior_dataset()
