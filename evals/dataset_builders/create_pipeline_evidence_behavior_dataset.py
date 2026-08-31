"""Create the end-to-end pipeline evidence-behavior dataset in LangSmith."""

import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).parents[2]
DATASET_PATH = ROOT / "evals" / "dataset" / "pipeline_evidence_behavior_dataset.json"
LANGSMITH_DATASET_NAME = "pipeline_evidence_behavior_dataset"


def load_examples() -> list[dict]:
    """Read the curated pipeline behavior questions."""
    try:
        examples = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read {DATASET_PATH}") from error

    if not isinstance(examples, list) or not examples:
        raise RuntimeError("Pipeline evidence-behavior dataset must be a non-empty list")
    return examples


def create_pipeline_evidence_behavior_dataset() -> None:
    """Publish the questions without replacing an existing LangSmith dataset."""
    load_dotenv()
    client = Client()

    if client.has_dataset(dataset_name=LANGSMITH_DATASET_NAME):
        raise RuntimeError(
            f"LangSmith dataset already exists: {LANGSMITH_DATASET_NAME}. "
            "Delete it explicitly before publishing a changed dataset."
        )

    examples = [
        {
            "inputs": {"question": example["question"]},
            "metadata": {
                "example_id": example["id"],
                **example.get("metadata", {}),
            },
            "split": "test",
        }
        for example in load_examples()
    ]

    client.create_dataset(
        LANGSMITH_DATASET_NAME,
        description=(
            "Nine questions for judging answer, partial-answer, and refusal "
            "behavior against evidence returned by live production retrieval."
        ),
    )
    client.create_examples(dataset_name=LANGSMITH_DATASET_NAME, examples=examples)
    print(f"Created LangSmith dataset: {LANGSMITH_DATASET_NAME} ({len(examples)} examples)")


if __name__ == "__main__":
    create_pipeline_evidence_behavior_dataset()
