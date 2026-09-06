import sys

import pytest  # pyright: ignore[reportMissingImports]

from arxiv_rag.ingestion import vector_db_ingest
from evals.regression.runner import _error_record, _metric_record, write_results
from evals.regression.summary import build_summary
from evals.regression.suites import GENERATION_SUBSET, priority_specs


def test_suites_can_be_listed_without_a_chroma_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """CI lists the suites on a runner that has not ingested the corpus yet."""
    monkeypatch.setattr(
        vector_db_ingest, "CHROMA_DATABASE_FILE", tmp_path / "missing.sqlite3"
    )
    for module_name in list(sys.modules):
        if module_name.startswith("evals."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    from evals.regression.suites import full_specs as fresh_full_specs
    from evals.regression.suites import priority_specs as fresh_priority_specs

    assert fresh_full_specs()
    assert fresh_priority_specs()


def test_priority_suite_is_answering_and_pipeline_only() -> None:
    names = [spec["name"] for spec in priority_specs()]

    assert names == [
        "generation_groundedness",
        "generation_fact_citation",
        "generation_correctness",
        "pipeline_required_fact_coverage",
        "pipeline_fact_citation",
    ]


def test_priority_suite_uses_the_fixed_subset_everywhere() -> None:
    for spec in priority_specs():
        assert spec["subset_ids"] == GENERATION_SUBSET
        assert spec["expected"] == len(GENERATION_SUBSET)


def test_metric_record_carries_threshold_and_status() -> None:
    record = _metric_record("generation_correctness.correctness", [("gen-001", 1.0)], 1)

    assert record["average"] == 1.0
    assert record["threshold"] == 0.75
    assert record["status"] == "PASS"
    assert record["passed"] is True
    assert record["error"] is None


def test_error_record_reports_no_scores() -> None:
    record = _error_record("generation_correctness.correctness", 12, "RuntimeError: down")

    assert record["scores"] == []
    assert record["status"] == "ERROR"
    assert record["passed"] is False
    assert record["error"] == "RuntimeError: down"


def test_write_results_round_trips(tmp_path) -> None:
    import json

    results_path = tmp_path / "nested" / "results.json"
    record = _metric_record("generation_correctness.correctness", [("gen-001", 1.0)], 1)

    write_results(results_path, "priority", True, True, [record])
    results = json.loads(results_path.read_text(encoding="utf-8"))

    assert results["suite"] == "priority"
    assert results["uploaded"] is True
    assert results["status"] == "PASS"
    assert results["metrics"] == [record]


def test_summary_renders_a_row_for_each_metric() -> None:
    results = {
        "suite": "priority",
        "uploaded": True,
        "status": "FAIL",
        "metrics": [
            _metric_record("generation_correctness.correctness", [("gen-001", 1.0)], 1),
            _error_record("pipeline_fact_citation.fact_citation", 12, "RuntimeError: down"),
        ],
    }

    summary = build_summary(results)

    assert "priority suite — FAIL" in summary
    assert "| `generation_correctness.correctness` | 1.0000 | 0.7500 | 1/1 | ✅ PASS |" in summary
    assert "| `pipeline_fact_citation.fact_citation` | n/a | 0.9500 | 0/12 | 💥 ERROR |" in summary
    assert "- `pipeline_fact_citation.fact_citation`: RuntimeError: down" in summary


def test_summary_handles_an_empty_run() -> None:
    summary = build_summary({"suite": "priority", "uploaded": False, "status": "FAIL", "metrics": []})

    assert "_no metrics recorded_" in summary
    assert "Uploaded to LangSmith: no" in summary
