"""Shared validation and LangSmith example-building for application-safety datasets.

The v2 datasets these were originally written for have been deprecated in
LangSmith. `create_application_safety_dataset_v3.py` reuses `validate_examples`
and `build_langsmith_examples` below to publish the current v3 datasets.
"""

from evals.application.safety import LANGSMITH_DATASET_NAMES, SAFETY_POLICY_VERSION

EXPECTED_BEHAVIORS = {"answer", "limited_answer", "safety_refusal"}
EXPECTED_DIFFICULTIES = {"easy", "medium", "hard"}
EXPECTED_METRICS = set(LANGSMITH_DATASET_NAMES)


def _validate_string_list(example: dict, field: str) -> list[str]:
    """Return a validated optional list of non-empty strings."""
    values = example.get(field, [])
    if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
        raise RuntimeError(f"Safety example {example['id']} has invalid {field}")
    return values


def validate_examples(examples: object, expected_counts: dict[str, int]) -> None:
    """Validate the focused, single-metric safety dataset contract."""
    if not isinstance(examples, list) or not examples:
        raise RuntimeError("Application-safety dataset must be a non-empty list")

    if set(expected_counts) != EXPECTED_METRICS:
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

    if counts != expected_counts:
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
