"""Compare evaluation scores with approved release policies."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScorePolicy:
    """An absolute floor plus an optional allowed drop from a baseline."""

    minimum_score: float
    baseline_score: float | None = None
    allowed_drop: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_score", self.minimum_score),
            ("allowed_drop", self.allowed_drop),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.baseline_score is not None and not 0 <= self.baseline_score <= 1:
            raise ValueError("baseline_score must be between 0 and 1")


@dataclass(frozen=True)
class ScoreDecision:
    """The deterministic release decision for one metric."""

    status: str
    passed: bool
    reasons: tuple[str, ...]


BASELINE_SCHEMA_VERSION = 1

# Add entries only after a run has been manually reviewed and approved.
# Keys use "<evaluation name>.<feedback key>" so component and pipeline metrics
# with similar names cannot overwrite one another.
APPROVED_POLICIES: dict[str, ScorePolicy] = {}


def check_score(
    metric_id: str, average_score: float | None, completed: int, expected: int
) -> ScoreDecision:
    """Check completion first, then any approved floor and baseline delta."""
    reasons = []
    if completed != expected:
        reasons.append(f"completed {completed}/{expected} expected results")
    if average_score is None:
        reasons.append("no numeric score was produced")

    policy = APPROVED_POLICIES.get(metric_id)
    if policy is None:
        if reasons:
            return ScoreDecision(status="FAIL", passed=False, reasons=tuple(reasons))
        return ScoreDecision(
            status="REPORT ONLY",
            passed=True,
            reasons=("no approved threshold has been recorded",),
        )

    if average_score is not None:
        if average_score < policy.minimum_score:
            reasons.append(
                f"score {average_score:.4f} is below minimum {policy.minimum_score:.4f}"
            )
        if policy.baseline_score is not None:
            regression_floor = policy.baseline_score - policy.allowed_drop
            if average_score < regression_floor:
                reasons.append(
                    f"score {average_score:.4f} is below regression floor {regression_floor:.4f}"
                )

    if reasons:
        return ScoreDecision(status="FAIL", passed=False, reasons=tuple(reasons))
    return ScoreDecision(
        status="PASS", passed=True, reasons=("all approved checks passed",)
    )
