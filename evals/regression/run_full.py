"""Run every active evaluation against its complete frozen dataset."""

from .runner import (  # pyright: ignore[reportMissingImports]
    parse_arguments,
    resolve_upload,
    run_suite,
)
from .suites import full_specs  # pyright: ignore[reportMissingImports]


def main() -> int:
    """Run the complete suite locally unless explicit LangSmith upload is requested."""
    arguments = parse_arguments(
        "Run all active component, pipeline, and application evaluations."
    )
    return run_suite(
        "full",
        full_specs(),
        resolve_upload(arguments.upload),
        arguments.results_json,
        arguments.results_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
