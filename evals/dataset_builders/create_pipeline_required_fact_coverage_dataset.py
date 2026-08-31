"""Create the end-to-end pipeline required-fact dataset in LangSmith."""

import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).parents[2]
SOURCE_DATASET_PATH = ROOT / "evals" / "dataset" / "generation_quality_dataset.json"
LANGSMITH_DATASET_NAME = "pipeline_required_fact_coverage_dataset"


def load_source_examples() -> list[dict]:
    """Read the frozen generation labels used to derive the pipeline dataset."""
    try:
        examples = json.loads(SOURCE_DATASET_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read {SOURCE_DATASET_PATH}") from error

    if not isinstance(examples, list) or not examples:
        raise RuntimeError("Pipeline source dataset must be a non-empty list")
    return examples


def build_pipeline_examples() -> list[dict]:
    """Keep required facts hidden from the live retrieval and generation target."""
    pipeline_examples = []

    for example in load_source_examples():
        required_facts = [
            required_fact["fact"]
            for required_fact in example["required_facts"]
        ]

        pipeline_examples.append({
            "inputs": {"question": example["question"]},
            "outputs": {"required_facts": required_facts},
            "metadata": {
                "example_id": example["id"],
                "source_dataset": "generation_quality_dataset",
                **example.get("metadata", {}),
            },
            "split": "test",
        })

    return pipeline_examples


def create_pipeline_required_fact_coverage_dataset() -> None:
    """Publish a separate pipeline dataset without duplicating local labels."""
    load_dotenv()
    client = Client()

    if client.has_dataset(dataset_name=LANGSMITH_DATASET_NAME):
        raise RuntimeError(
            f"LangSmith dataset already exists: {LANGSMITH_DATASET_NAME}. "
            "Delete it explicitly before publishing a changed dataset."
        )

    examples = build_pipeline_examples()
    client.create_dataset(
        LANGSMITH_DATASET_NAME,
        description=(
            "Questions and hidden required-fact references for measuring final-answer "
            "coverage through live production retrieval and generation."
        ),
    )
    client.create_examples(dataset_name=LANGSMITH_DATASET_NAME, examples=examples)
    print(f"Created LangSmith dataset: {LANGSMITH_DATASET_NAME} ({len(examples)} examples)")


if __name__ == "__main__":
    create_pipeline_required_fact_coverage_dataset()
