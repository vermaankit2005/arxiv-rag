"""One logger for the whole project.

Use it like this, at the top of any module:

    from arxiv_rag.logging import get_logger

    log = get_logger(__name__)
    log.info("loaded %s: %d passages", arxiv_id, len(doc.passages))

Two reasons this file exists rather than plain `print()`:

1. **Windows console encoding.** Paper text carries Greek letters and dashes.
   Printing them on the default Windows console raises UnicodeEncodeError and
   kills the run, which is why every command so far has had to be prefixed with
   PYTHONIOENCODING=utf-8. The handler below sets UTF-8 on its own stream, so
   that stops being something anyone has to remember.
2. **Only real problems should look like problems.** Long jobs -- fetching 40
   papers, scoring 12 -- need to say where they are, and an editor paints
   anything on the error stream red whatever it says. So DEBUG and INFO go to
   stdout and WARNING upwards goes to stderr: red then means something actually
   went wrong.

   The cost, worth knowing: piping a script's output to a file now catches the
   progress lines along with the results. If that ever gets annoying, the fix is
   `2>&1 >/dev/null` at the call site, or moving that script's real output off
   stdout -- not changing this file back.

Deliberately not here: config files, log files on disk, rotation, JSON output,
per-module levels. None of those are needed by anything running today. Add one
when a real run is hard to debug without it, not before.

Level comes from the ARXIV_RAG_LOG environment variable and defaults to INFO:

    ARXIV_RAG_LOG=debug uv run python experiments/01_loading/score.py html
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys

LEVEL_VAR = "ARXIV_RAG_LOG"
DEFAULT_LEVEL = "INFO"

# Every logger in the project hangs off this one, so configuring it once
# configures all of them and nothing leaks into libraries' own logging.
ROOT_NAME = "arxiv_rag"

_configured = False


def _stream_handler(stream) -> logging.StreamHandler:
    # Python 3.7+ lets us re-open the stream as UTF-8 in place. If it is already
    # UTF-8, or is something that cannot be reconfigured (a captured buffer in a
    # test), leave it alone rather than failing at import time.
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8", errors="replace")

    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    return handler


def _handlers() -> list[logging.Handler]:
    """Chatter on stdout, problems on stderr, so red always means red."""
    chatter = _stream_handler(sys.stdout)
    # Without this the same line goes out of both handlers and prints twice.
    chatter.addFilter(lambda record: record.levelno < logging.WARNING)

    problems = _stream_handler(sys.stderr)
    problems.setLevel(logging.WARNING)

    return [chatter, problems]


def setup(level: str | int | None = None) -> logging.Logger:
    """Configure the project's logging. Safe to call as often as you like.

    Called automatically by `get_logger`, so scripts normally never call it.
    Call it directly only to force a level in code, e.g. `setup("DEBUG")`.
    """
    global _configured

    root = logging.getLogger(ROOT_NAME)
    if not _configured:
        for handler in _handlers():
            root.addHandler(handler)
        # Ours is the only handler that should print our records. Without this,
        # anything that has called logging.basicConfig() prints every line twice.
        root.propagate = False
        _configured = True

    if level is None:
        level = os.environ.get(LEVEL_VAR, DEFAULT_LEVEL)
    if isinstance(level, str):
        level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    root.setLevel(level)
    return root


def get_logger(name: str | None = None) -> logging.Logger:
    """The logger to use in a module. Pass `__name__`.

    A module called `arxiv_rag.loading.arxiv` keeps that name in each record, so every
    line says which part of the project it came from and one switch controls
    them all.
    """
    setup()
    if not name or name == "__main__":
        return logging.getLogger(ROOT_NAME)
    return logging.getLogger(f"{ROOT_NAME}.{name}")
