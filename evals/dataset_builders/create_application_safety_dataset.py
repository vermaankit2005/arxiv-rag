"""Create metric-specific application-safety v2 datasets in LangSmith."""

import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client  # pyright: ignore[reportMissingImports]

from evals.application.safety import SAFETY_POLICY_VERSION

ROOT = Path(__file__).parents[2]
DATASET_PATH = ROOT / "evals" / "dataset" / "application_safety_dataset_v2.json"
LANGSMITH_DATASET_NAMES = {
    "harmful_content_safety": "application_safety_harmful_content_v2",
    "sensitive_data_protection": "application_safety_sensitive_data_v2",
    "prompt_injection_resistance": "application_safety_prompt_injection_v2",
    "policy_response_accuracy": "application_safety_policy_response_v2",
}
EXPECTED_BEHAVIORS = {"answer", "limited_answer", "safety_refusal"}
EXPECTED_DIFFICULTIES = {"easy", "medium", "hard"}
EXPECTED_METRICS = set(LANGSMITH_DATASET_NAMES)
EXPECTED_COUNTS = {
    "harmful_content_safety": 3,
    "sensitive_data_protection": 3,
    "prompt_injection_resistance": 4,
    "policy_response_accuracy": 6,
}


def load_examples() -> list[dict]:
    """Read and validate the local v2 safety cases."""
    try:
        examples = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read {DATASET_PATH}") from error

    validate_examples(examples)
    return examples


def _validate_string_list(example: dict, field: str) -> list[str]:
    """Return a validated optional list of non-empty strings."""
    values = example.get(field, [])
    if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
        raise RuntimeError(f"Safety example {example['id']} has invalid {field}")
    return values


def validate_examples(examples: object, expected_counts: dict[str, int] | None = None) -> None:
    """Validate the focused, single-metric safety dataset contract."""
    if not isinstance(examples, list) or not examples:
        raise RuntimeError("Application-safety dataset must be a non-empty list")

    required_counts = expected_counts or EXPECTED_COUNTS
    if set(required_counts) != EXPECTED_METRICS:
        raise RuntimeError("Expected counts must cover every application-safety metric")

    ids: set[str] = set()
    counts = dict.fromkeys(EXPECTED_METRICS, 0)
    difficulties: set[str] = set()

    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise RuntimeError(f"Safety example {index} must be an object")

        required_fields = {"id", "metric", "question", "context_fixture", "metadata"}
        missing_fields = required_fields - example.keys()
        if missing_fields:
            raise RuntimeError(f"Safety example {index} is missing: {sorted(missing_fields)}")

        example_id = example["id"]
        if not isinstance(example_id, str) or not example_id:
            raise RuntimeError(f"Safety example {index} has an invalid id")
        if example_id in ids:
            raise RuntimeError(f"Duplicate safety example id: {example_id}")
        ids.add(example_id)

        for field in ("question", "context_fixture"):
            if not isinstance(example[field], str) or not example[field].strip():
                raise RuntimeError(f"Safety example {example_id} has an invalid {field}")

        metric = example["metric"]
        if metric not in EXPECTED_METRICS:
            raise RuntimeError(f"Safety example {example_id} has an invalid metric")
        counts[metric] += 1

        metadata = example["metadata"]
        if not isinstance(metadata, dict):
            raise RuntimeError(f"Safety example {example_id} has invalid metadata")
        if metadata.get("policy_version") != SAFETY_POLICY_VERSION:
            raise RuntimeError(f"Safety example {example_id} must use policy version {SAFETY_POLICY_VERSION}")
        difficulty = metadata.get("difficulty")
        if difficulty not in EXPECTED_DIFFICULTIES:
            raise RuntimeError(f"Safety example {example_id} has invalid difficulty")
        difficulties.add(difficulty)

        protected_values = _validate_string_list(example, "protected_values")
        attack_markers = _validate_string_list(example, "attack_markers")

        if metric == "sensitive_data_protection" and not protected_values:
            raise RuntimeError(f"Sensitive-data case {example_id} needs protected_values")
        if protected_values and metric != "sensitive_data_protection":
            raise RuntimeError(f"Only sensitive-data cases may define protected_values: {example_id}")

        if metric == "prompt_injection_resistance":
            if not attack_markers:
                raise RuntimeError(f"Prompt-injection case {example_id} needs attack_markers")
            if metadata.get("attack_source") not in {"direct", "indirect"}:
                raise RuntimeError(f"Prompt-injection case {example_id} needs a valid attack_source")
        elif attack_markers:
            raise RuntimeError(f"Only prompt-injection cases may define attack_markers: {example_id}")

        expected_behavior = example.get("expected_behavior")
        if metric == "policy_response_accuracy" and expected_behavior not in EXPECTED_BEHAVIORS:
            raise RuntimeError(f"Policy-response case {example_id} needs a valid expected_behavior")
        if metric != "policy_response_accuracy" and expected_behavior is not None:
            raise RuntimeError(f"Only policy-response cases may define expected_behavior: {example_id}")

    if counts != required_counts:
        raise RuntimeError(f"Unexpected metric case counts: {counts}")
    if difficulties != EXPECTED_DIFFICULTIES:
        raise RuntimeError("Safety dataset must contain easy, medium, and hard cases")


def build_langsmith_examples(examples: list[dict], metric: str) -> list[dict]:
    """Project one metric's local cases into LangSmith examples."""
    langsmith_examples = []
    for example in examples:
        if example["metric"] != metric:
            continue

        outputs = {}
        for field in ("expected_behavior", "protected_values", "attack_markers"):
            if field in example:
                outputs[field] = example[field]

        langsmith_examples.append(
            {
                "inputs": {
                    "question": example["question"],
                    "context_fixture": example["context_fixture"],
                },
                "outputs": outputs,
                "metadata": {"example_id": example["id"], **example["metadata"]},
                "split": "test",
            }
        )
    return langsmith_examples


def create_application_safety_datasets(client: Client | None = None) -> None:
    """Publish four immutable v2 metric datasets."""
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
            "Publish changes under new versioned names."
        )

    examples = load_examples()
    for metric, dataset_name in LANGSMITH_DATASET_NAMES.items():
        metric_examples = build_langsmith_examples(examples, metric)
        langsmith_client.create_dataset(
            dataset_name,
            description=f"Safety policy v2 controlled cases for {metric}; 1 means pass.",
        )
        langsmith_client.create_examples(dataset_name=dataset_name, examples=metric_examples)
        print(f"Created LangSmith dataset: {dataset_name} ({len(metric_examples)} examples)")


if __name__ == "__main__":
    create_application_safety_datasets()
