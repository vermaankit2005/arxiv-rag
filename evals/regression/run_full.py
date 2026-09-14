"""Run every active evaluation against its complete frozen dataset."""

from .runner import run_suite  # pyright: ignore[reportMissingImports]
from .suites import full_specs  # pyright: ignore[reportMissingImports]


def main() -> int:
    return run_suite("full", full_specs())


if __name__ == "__main__":
    raise SystemExit(main())
