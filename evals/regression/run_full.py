"""Run every active evaluation against its complete frozen dataset."""

from .runner import (  # pyright: ignore[reportMissingImports]
    parse_upload_flag,
    run_suite,
)
from .suites import full_specs  # pyright: ignore[reportMissingImports]


def main() -> int:
    """Run the complete suite locally unless explicit LangSmith upload is requested."""
    upload_results = parse_upload_flag(
        "Run all active component, pipeline, and application evaluations."
    )
    return run_suite("full", full_specs(), upload_results)


if __name__ == "__main__":
    raise SystemExit(main())
