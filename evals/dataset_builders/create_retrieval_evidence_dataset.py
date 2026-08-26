import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).parents[2]
DATASET_PATH = ROOT / "evals" / "dataset" / "retrieval_evidence_dataset.json"
LANGSMITH_DATASET_NAME = "retrieval_evidence_dataset"


def load_local_dataset() -> list[dict]:
    """Read the curated retrieval examples."""
    try:
        examples = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Could not read the retrieval evidence dataset") from error

    if not isinstance(examples, list) or not examples:
        raise RuntimeError("Retrieval evidence dataset must be a non-empty list")
    return examples


def create_retrieval_evidence_dataset() -> None:
    """Upload the curated answer key without replacing an existing dataset."""
    examples = load_local_dataset()
    load_dotenv()
    langsmith_client = Client()

    if langsmith_client.has_dataset(dataset_name=LANGSMITH_DATASET_NAME):
        local_by_question = {
            example["inputs"]["question"]: example for example in examples
        }
        existing = {}
        for existing_example in langsmith_client.list_examples(
            dataset_name=LANGSMITH_DATASET_NAME
        ):
            if existing_example.inputs is None or existing_example.outputs is None:
                raise RuntimeError("LangSmith retrieval example is missing data")
            question = existing_example.inputs["question"]
            local_example = local_by_question.get(question, {})
            expected_metadata = local_example.get("metadata", {})
            actual_metadata = existing_example.metadata or {}
            existing[question] = {
                "outputs": existing_example.outputs,
                "metadata": {
                    key: actual_metadata.get(key) for key in expected_metadata
                },
            }

        expected = {
            question: {
                "outputs": example["outputs"],
                "metadata": example.get("metadata", {}),
            }
            for question, example in local_by_question.items()
        }
        if existing == expected:
            print(f"LangSmith dataset is up to date: {LANGSMITH_DATASET_NAME}")
            return
        raise RuntimeError(
            f"LangSmith dataset is stale: {LANGSMITH_DATASET_NAME}. "
            "Delete it before rebuilding so old experiments are not changed silently."
        )

    langsmith_client.create_dataset(
        LANGSMITH_DATASET_NAME,
        description=(
            "Human-curated questions and source evidence for retriever evaluation."
        ),
    )
    for example in examples:
        langsmith_client.create_example(
            dataset_name=LANGSMITH_DATASET_NAME,
            inputs=example["inputs"],
            outputs=example["outputs"],
            metadata=example.get("metadata"),
            split="test",
        )

    print(f"Created LangSmith dataset: {LANGSMITH_DATASET_NAME}")


if __name__ == "__main__":
    create_retrieval_evidence_dataset()
