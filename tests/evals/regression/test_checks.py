import pytest  # pyright: ignore[reportMissingImports]

from evals.regression import checks  # pyright: ignore[reportAttributeAccessIssue]


def test_score_without_approved_policy_is_report_only() -> None:
    decision = checks.check_score("generation_naturalness.naturalness", 0.75, 12, 12)

    assert decision.passed is True
    assert decision.status == "REPORT ONLY"


def test_incomplete_metric_fails_without_an_approved_policy() -> None:
    decision = checks.check_score("generation_naturalness.naturalness", 0.75, 11, 12)

    assert decision.passed is False
    assert decision.status == "FAIL"
    assert decision.reasons == ("completed 11/12 expected results",)


def test_approved_minimum_and_regression_floor_are_both_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = checks.ScorePolicy(
        minimum_score=0.70, baseline_score=0.80, allowed_drop=0.05
    )
    monkeypatch.setitem(
        checks.APPROVED_POLICIES, "generation_naturalness.naturalness", policy
    )

    decision = checks.check_score("generation_naturalness.naturalness", 0.72, 12, 12)

    assert decision.passed is False
    assert decision.status == "FAIL"
    assert decision.reasons == ("score 0.7200 is below regression floor 0.7500",)


def test_invalid_policy_score_is_rejected() -> None:
    with pytest.raises(ValueError, match="minimum_score must be between 0 and 1"):
        checks.ScorePolicy(minimum_score=1.1)
