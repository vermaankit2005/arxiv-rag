"""Run every active evaluation against its complete frozen dataset."""

from .runner import parse_arguments, run_suite  # pyright: ignore[reportMissingImports]
from .suites import full_specs  # pyright: ignore[reportMissingImports]


def main() -> int:
    """Run the complete suite using the shared evaluation upload setting."""
    arguments = parse_arguments(
        "Run all active component, pipeline, and application evaluations."
    )
    return run_suite(
        "full",
        full_specs(),
        arguments.results_json,
        arguments.results_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
