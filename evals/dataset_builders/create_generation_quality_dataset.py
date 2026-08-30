"""Create the frozen answer-quality dataset in LangSmith."""

import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).parents[2]
DATASET_PATH = ROOT / "evals" / "dataset" / "generation_quality_dataset.json"
LANGSMITH_DATASET_NAME = "generation_quality_dataset"
DATASET_DESCRIPTION = (
    "Frozen questions, source passages, and required facts for evaluating "
    "grounded answer generation, citation support, correctness, and completeness."
)


class GenerationQualityDatasetBuilder:
    """Load the frozen answer-quality examples and publish them to LangSmith."""

    def __init__(self, client: Client | None = None) -> None:
        load_dotenv()
        self.client = client or Client()

    def load_examples(self) -> list[dict]:
        """Read and validate the frozen local dataset."""
        try:
            examples = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Could not read the generation-quality dataset from {DATASET_PATH}"
            ) from error

        if not isinstance(examples, list) or not examples:
            raise RuntimeError("Generation-quality dataset must be a non-empty list")
        return examples

    def build_langsmith_examples(self) -> list[dict]:
        """Map local examples to LangSmith inputs, reference outputs, and metadata."""
        langsmith_examples = []
        for example in self.load_examples():
            langsmith_examples.append(
                {
                    "inputs": {
                        "question": example["question"],
                        "context_passages": example["context_passages"],
                    },
                    "outputs": {"required_facts": example["required_facts"]},
                    "metadata": {
                        "example_id": example["id"],
                        **example.get("metadata", {}),
                    },
                    "split": "test",
                }
            )
        return langsmith_examples

    def create(self) -> None:
        """Create the dataset and its examples without replacing existing data."""
        if self.client.has_dataset(dataset_name=LANGSMITH_DATASET_NAME):
            raise RuntimeError(
                f"LangSmith dataset already exists: {LANGSMITH_DATASET_NAME}. "
                "Delete it explicitly before publishing a changed frozen dataset."
            )

        examples = self.build_langsmith_examples()
        self.client.create_dataset(
            LANGSMITH_DATASET_NAME,
            description=DATASET_DESCRIPTION,
        )
        self.client.create_examples(
            dataset_name=LANGSMITH_DATASET_NAME,
            examples=examples,
        )
        print(
            f"Created LangSmith dataset: {LANGSMITH_DATASET_NAME} "
            f"({len(examples)} examples)"
        )


if __name__ == "__main__":
    GenerationQualityDatasetBuilder().create()
