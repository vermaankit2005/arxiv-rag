"""Execute evaluation suites and apply deterministic regression checks."""

import argparse
from collections.abc import Sequence
from typing import Any

from dotenv import load_dotenv
from langsmith import Client

from .checks import (  # pyright: ignore[reportMissingImports]
    BASELINE_SCHEMA_VERSION,
    check_score,
)
from .report import (  # pyright: ignore[reportMissingImports]
    build_metric_reports,
    collect_case_scores,
    print_metric_report,
)
from .suites import (  # pyright: ignore[reportMissingImports]
    BALANCED_GENERATION_EXAMPLE_IDS,
    EvaluationSpec,
)


def _select_data(client: Client, spec: EvaluationSpec) -> Any:
    if not spec.use_balanced_subset:
        return spec.dataset_name

    examples = list(client.list_examples(dataset_name=spec.dataset_name))
    examples_by_id = {
        example.metadata.get("example_id"): example
        for example in examples
        if example.metadata
    }
    missing_ids = [
        example_id
        for example_id in BALANCED_GENERATION_EXAMPLE_IDS
        if example_id not in examples_by_id
    ]
    if missing_ids:
        raise RuntimeError(
            f"Dataset {spec.dataset_name} is missing balanced subset IDs: {missing_ids}"
        )
    return [
        examples_by_id[example_id] for example_id in BALANCED_GENERATION_EXAMPLE_IDS
    ]


def run_suite(name: str, specs: Sequence[EvaluationSpec], upload_results: bool) -> int:
    """Run each evaluation, print safe reports, and return a process exit code."""
    load_dotenv()
    client = Client()
    suite_passed = True

    print(f"Running {name} evaluation suite")
    print(f"Upload to LangSmith: {'yes' if upload_results else 'no'}")

    for spec in specs:
        print(f"\nStarting {spec.name}...")
        try:
            data = _select_data(client, spec)
            metadata = {
                **spec.metadata,
                "regression_suite": name,
                "baseline_schema_version": BASELINE_SCHEMA_VERSION,
                "case_selection": "balanced-12" if spec.use_balanced_subset else "full",
            }
            experiment_results = client.evaluate(
                spec.target,
                data=data,
                evaluators=list(spec.evaluators),
                metadata=metadata,
                experiment_prefix=spec.experiment_prefix,
                description=spec.description,
                max_concurrency=spec.max_concurrency,
                blocking=True,
                upload_results=upload_results,
            )
            case_scores = collect_case_scores(experiment_results)
        except Exception as error:  # noqa: BLE001 - suite boundary reports each failed eval
            suite_passed = False
            print(
                f"FAIL: {spec.name} could not complete ({type(error).__name__}: {error})"
            )
            continue

        reports = build_metric_reports(
            evaluation_name=spec.name,
            case_scores=case_scores,
            feedback_keys=spec.feedback_keys,
            expected=spec.expected_examples,
        )
        for report in reports:
            decision = check_score(
                metric_id=report.metric_id,
                average_score=report.average_score,
                completed=len(report.cases),
                expected=report.expected,
            )
            print_metric_report(report, decision)
            suite_passed = suite_passed and decision.passed

    print(f"\nSuite status: {'PASS' if suite_passed else 'FAIL'}")
    return 0 if suite_passed else 1


def parse_upload_flag(description: str) -> bool:
    """Parse the shared upload option used by both suite entry points."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload experiments to LangSmith. Local runs do not upload by default.",
    )
    return parser.parse_args().upload
