from types import SimpleNamespace
from typing import Any, cast

import pytest  # pyright: ignore[reportMissingImports]

from evals.regression import runner
from evals.regression.runner import _collect_scores, _print_report, _select_data, _status
from evals.regression.suites import GENERATION_SUBSET, full_specs


def test_fixed_subset_is_selected_in_order() -> None:
    examples = [
        SimpleNamespace(metadata={"example_id": example_id})
        for example_id in reversed(GENERATION_SUBSET)
    ]
    client = SimpleNamespace(list_examples=lambda **_: iter(examples))

    selected = cast(
        list[Any],
        _select_data(cast(Any, client), "generation_quality_dataset", GENERATION_SUBSET),
    )

    assert tuple(example.metadata["example_id"] for example in selected) == GENERATION_SUBSET


def test_fixed_subset_rejects_missing_id() -> None:
    client = SimpleNamespace(list_examples=lambda **_: iter([]))

    with pytest.raises(RuntimeError, match="missing subset IDs"):
        _select_data(cast(Any, client), "generation_quality_dataset", GENERATION_SUBSET)


def test_collect_scores_keeps_only_safe_ids_and_requested_metric() -> None:
    example = SimpleNamespace(id="remote-id", metadata={"example_id": "gen-001"})
    results = [
        {
            "example": example,
            "evaluation_results": {
                "results": [
                    SimpleNamespace(key="naturalness", score=0.75),
                    SimpleNamespace(key="other", score=1),
                ]
            },
        }
    ]

    assert _collect_scores(results, "naturalness") == [("gen-001", 0.75)]


def test_incomplete_results_fail() -> None:
    assert _status("generation.naturalness", [("gen-001", 1.0)], 2) == ("FAIL", False)


def test_every_full_suite_metric_has_temporary_threshold() -> None:
    metric_ids = {
        f"{spec['name']}.{feedback_key}"
        for spec in full_specs()
        for feedback_key in spec["feedback_keys"]
    }

    assert set(runner.THRESHOLDS) == metric_ids
    assert set(runner.THRESHOLDS.values()) == {0.75}


def test_approved_threshold_is_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(runner.THRESHOLDS, "generation.naturalness", 0.8)

    assert _status("generation.naturalness", [("gen-001", 0.75)], 1) == ("FAIL", False)


def test_missing_threshold_is_report_only() -> None:
    assert _status("unknown.metric", [("gen-001", 0.1)], 1) == ("REPORT ONLY", True)


def test_print_report_returns_pass_or_fail(capsys: pytest.CaptureFixture[str]) -> None:
    assert _print_report("generation_naturalness.naturalness", [("gen-001", 1.0)], 1) is True
    assert _print_report("generation_naturalness.naturalness", [("gen-001", 1.0)], 2) is False

    output = capsys.readouterr().out
    assert "Status: PASS" in output
    assert "Status: FAIL" in output
