# pyright: reportMissingImports=false

from types import SimpleNamespace
from typing import Any, cast

import pytest

from evals.regression.runner import _select_data
from evals.regression.suites import (
    BALANCED_GENERATION_EXAMPLE_IDS,
    EvaluationSpec,
)


def _target(inputs: dict) -> dict:
    return inputs


def test_balanced_subset_is_fixed_and_ordered() -> None:
    examples = [
        SimpleNamespace(metadata={"example_id": example_id})
        for example_id in reversed(BALANCED_GENERATION_EXAMPLE_IDS)
    ]
    client = SimpleNamespace(list_examples=lambda **_: iter(examples))
    spec = EvaluationSpec(
        name="generation",
        dataset_name="generation_quality_dataset",
        target=_target,
        evaluators=(),
        feedback_keys=("naturalness",),
        metadata={},
        experiment_prefix="generation",
        description="test",
        expected_examples=12,
        use_balanced_subset=True,
    )

    selected = _select_data(cast(Any, client), spec)

    assert len(selected) == 12
    assert tuple(example.metadata["example_id"] for example in selected) == (
        BALANCED_GENERATION_EXAMPLE_IDS
    )


def test_balanced_subset_rejects_missing_id() -> None:
    client = SimpleNamespace(list_examples=lambda **_: iter([]))
    spec = EvaluationSpec(
        name="generation",
        dataset_name="generation_quality_dataset",
        target=_target,
        evaluators=(),
        feedback_keys=("naturalness",),
        metadata={},
        experiment_prefix="generation",
        description="test",
        expected_examples=12,
        use_balanced_subset=True,
    )

    with pytest.raises(RuntimeError, match="missing balanced subset IDs"):
        _select_data(cast(Any, client), spec)
