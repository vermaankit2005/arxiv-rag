# pyright: reportMissingImports=false

"""Run urgent evals with the fixed balanced 12-case generation subset."""

from .runner import parse_upload_flag, run_suite
from .suites import priority_specs


def main() -> int:
    """Run the small suite locally unless explicit LangSmith upload is requested."""
    upload_results = parse_upload_flag(
        "Run naturalness, sensitive-data, policy-response, and citation-support evals."
    )
    return run_suite("priority", priority_specs(), upload_results)


if __name__ == "__main__":
    raise SystemExit(main())
