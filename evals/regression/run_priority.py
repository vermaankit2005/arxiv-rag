# pyright: reportMissingImports=false

"""Run the merge-gate answering and pipeline evals on the fixed 12-case subset."""
from evals.regression.runner import parse_arguments, run_suite
from evals.regression.suites import priority_specs


def main() -> int:
    """Run the small suite locally unless explicit LangSmith upload is requested."""
    arguments = parse_arguments(
        "Run the answering and pipeline priority evals on the fixed 12-case subset."
    )
    return run_suite(
        "priority", priority_specs(), arguments.upload, arguments.results_json
    )


if __name__ == "__main__":
    raise SystemExit(main())
