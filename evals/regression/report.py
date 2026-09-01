# pyright: reportMissingImports=false

"""Render privacy-safe local reports from LangSmith experiment results."""

from collections.abc import Iterable
from dataclasses import dataclass

from .checks import ScoreDecision


@dataclass(frozen=True)
class CaseScore:
    """One evaluator score without raw application content."""

    example_id: str
    feedback_key: str
    score: float
    comment: str


@dataclass(frozen=True)
class MetricReport:
    """All completed case scores for one feedback key."""

    metric_id: str
    cases: tuple[CaseScore, ...]
    expected: int

    @property
    def average_score(self) -> float | None:
        if not self.cases:
            return None
        return sum(case.score for case in self.cases) / len(self.cases)


def _safe_example_id(example: object) -> str:
    metadata = getattr(example, "metadata", None) or {}
    example_id = metadata.get("example_id")
    if isinstance(example_id, str) and example_id:
        return example_id
    fallback = getattr(example, "id", "unknown-example")
    return str(fallback)


def _redact_known_values(comment: str, example: object) -> str:
    outputs = getattr(example, "outputs", None) or {}
    safe_comment = comment
    for field in ("protected_values", "attack_markers"):
        values = outputs.get(field, [])
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str) and value:
                safe_comment = safe_comment.replace(value, "[REDACTED]")
    return safe_comment


def collect_case_scores(results: Iterable[dict]) -> tuple[CaseScore, ...]:
    """Extract scores and comments while excluding inputs, outputs, and traces."""
    case_scores = []
    for item in results:
        example = item["example"]
        example_id = _safe_example_id(example)
        evaluation_results = item["evaluation_results"]["results"]
        for evaluation in evaluation_results:
            if not isinstance(evaluation.score, (int, float, bool)):
                continue
            comment = evaluation.comment or "No evaluator reason was returned."
            case_scores.append(
                CaseScore(
                    example_id=example_id,
                    feedback_key=evaluation.key,
                    score=evaluation.score * 1.0,
                    comment=_redact_known_values(comment, example),
                )
            )
    return tuple(case_scores)


def build_metric_reports(
    evaluation_name: str,
    case_scores: tuple[CaseScore, ...],
    feedback_keys: tuple[str, ...],
    expected: int,
) -> tuple[MetricReport, ...]:
    """Group case scores by the feedback keys promised by an evaluation."""
    reports = []
    for feedback_key in feedback_keys:
        matching_cases = tuple(
            case for case in case_scores if case.feedback_key == feedback_key
        )
        reports.append(
            MetricReport(
                metric_id=f"{evaluation_name}.{feedback_key}",
                cases=matching_cases,
                expected=expected,
            )
        )
    return tuple(reports)


def print_metric_report(report: MetricReport, decision: ScoreDecision) -> None:
    """Print a compact DeepEval-style report without application content."""
    print(f"\n{report.metric_id}")
    print("-" * len(report.metric_id))
    for case in report.cases:
        print(f"{case.example_id}: {case.score:.4f}")
        print(f"  Reason: {case.comment}")

    average = "n/a" if report.average_score is None else f"{report.average_score:.4f}"
    print(f"Completed: {len(report.cases)}/{report.expected}")
    print(f"Average:   {average}")
    print(f"Status:    {decision.status}")
    for reason in decision.reasons:
        print(f"  {reason}")
