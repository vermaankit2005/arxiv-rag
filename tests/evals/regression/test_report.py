# pyright: reportMissingImports=false

from types import SimpleNamespace

from evals.regression.checks import ScoreDecision
from evals.regression.report import (
    MetricReport,
    build_metric_reports,
    collect_case_scores,
    print_metric_report,
)


def test_collect_case_scores_uses_safe_id_and_redacts_known_values() -> None:
    protected_value = "controlled-protected-value"
    example = SimpleNamespace(
        id="remote-id",
        metadata={"example_id": "protected-data-001"},
        outputs={"protected_values": [protected_value]},
    )
    evaluation = SimpleNamespace(
        key="sensitive_data_protection",
        score=0,
        comment=f"The answer exposed {protected_value}.",
    )
    results = [
        {
            "example": example,
            "evaluation_results": {"results": [evaluation]},
        }
    ]

    cases = collect_case_scores(results)

    assert len(cases) == 1
    assert cases[0].example_id == "protected-data-001"
    assert cases[0].comment == "The answer exposed [REDACTED]."


def test_build_metric_reports_includes_missing_promised_metric() -> None:
    reports = build_metric_reports(
        "loading", (), ("anchor_coverage", "text_recall"), 12
    )

    assert reports == (
        MetricReport("loading.anchor_coverage", (), 12),
        MetricReport("loading.text_recall", (), 12),
    )


def test_print_metric_report_excludes_application_content(capsys) -> None:
    report = MetricReport("generation.naturalness", (), 12)
    decision = ScoreDecision("FAIL", False, ("completed 0/12 expected results",))

    print_metric_report(report, decision)

    output = capsys.readouterr().out
    assert "generation.naturalness" in output
    assert "Completed: 0/12" in output
    assert "Status:    FAIL" in output
