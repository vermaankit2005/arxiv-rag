"""Turn a regression results JSON file into a Markdown table for CI."""

import argparse
import json
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a Markdown regression summary.")
    parser.add_argument("results_json", type=Path, help="Results file from a suite run.")
    parser.add_argument("--output", type=Path, default=None, help="Append the table here.")
    arguments = parser.parse_args()

    results = json.loads(arguments.results_json.read_text(encoding="utf-8"))
    summary = build_summary(results)

    if arguments.output is None:
        # The status icons are not printable on a Windows console codepage.
        sys.stdout.buffer.write(summary.encode("utf-8"))
    else:
        with arguments.output.open("a", encoding="utf-8") as output_file:
            output_file.write(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
