# pyright: reportMissingImports=false

"""Run the merge-gate generation and pipeline evals on the fixed 12-case subset."""
from evals.regression.runner import run_suite
from evals.regression.suites import priority_specs


def main() -> int:
    return run_suite("priority", priority_specs())


if __name__ == "__main__":
    raise SystemExit(main())
