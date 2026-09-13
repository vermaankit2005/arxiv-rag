# pyright: reportMissingImports=false

"""Run the merge-gate generation and pipeline evals on the fixed 12-case subset."""
from evals.regression.runner import parse_arguments, run_suite
from evals.regression.suites import priority_specs


def main() -> int:
    """Run the small suite using the shared evaluation upload setting."""
    arguments = parse_arguments(
        "Run the generation and pipeline priority evals on the fixed 12-case subset."
    )
    return run_suite(
        "priority",
        priority_specs(),
        arguments.results_json,
        arguments.results_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
