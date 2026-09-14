"""Render regression results as Markdown for local runs and CI."""

import os
import sys
from pathlib import Path

STATUS_ICONS = {"PASS": "✅", "FAIL": "❌", "REPORT ONLY": "📊", "ERROR": "💥"}


def _cell(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}s"


def build_summary(results: dict) -> str:
    """Render the run as a heading plus one row per metric."""
    suite = results.get("suite", "unknown")
    status = results.get("status", "FAIL")
    uploaded = "yes" if results.get("uploaded") else "no"
    metrics = results.get("metrics", [])

    lines = [
        f"## {STATUS_ICONS.get(status, '')} {suite} suite — {status}".strip(),
        "",
        f"Commit: `{results.get('commit', 'unknown')}`",
        f"Generator: `{results.get('generator_model', 'unknown')}` · "
        f"Judge: `{results.get('judge_model', 'unknown')}`",
        f"Started: {results.get('started_at', 'unknown')} · "
        f"Took: {_seconds(results.get('duration_seconds'))}",
        f"Uploaded to LangSmith: {uploaded}",
        "",
        "| Metric | Average | Threshold | Cases | Took | Status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for metric in metrics:
        icon = STATUS_ICONS.get(metric["status"], "")
        cases = f"{metric['completed']}/{metric['expected']}"
        lines.append(
            f"| `{metric['metric']}` | {_cell(metric['average'])} "
            f"| {_cell(metric['threshold'])} | {cases} "
            f"| {_seconds(metric.get('duration_seconds'))} | {icon} {metric['status']} |"
        )

    if not metrics:
        lines.append("| _no metrics recorded_ | | | | | |")

    errors = [metric for metric in metrics if metric.get("error")]
    if errors:
        lines.extend(["", "### Errors", ""])
        for metric in errors:
            lines.append(f"- `{metric['metric']}`: {metric['error']}")

    return "\n".join(lines) + "\n"


def write_summary(results: dict) -> None:
    """Print the summary locally and add it to the GitHub job summary when available."""
    summary = build_summary(results)

    # Writing bytes keeps the status icons printable on a Windows console codepage.
    sys.stdout.flush()
    sys.stdout.buffer.write(summary.encode("utf-8"))

    github_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if github_summary:
        with Path(github_summary).open("a", encoding="utf-8") as output_file:
            output_file.write(summary)
