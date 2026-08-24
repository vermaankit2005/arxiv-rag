"""Check that every loaded passage has a valid HTML anchor."""

import json
from pathlib import Path

import httpx

from arxiv_rag.loading import load_paper

ROOT = Path(__file__).parents[2]
CACHED_HTML_DIRECTORY = ROOT / "data" / "raw" / "sampled_html"
BENCHMARK_PAPERS_PATH = ROOT / "evals" / "dataset" / "papers.json"


def check_passage_anchor_validity() -> None:
    """Print anchor coverage across the cached benchmark papers."""
    try:
        papers = json.loads(BENCHMARK_PAPERS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not read benchmark papers from {BENCHMARK_PAPERS_PATH}"
        ) from error

    anchors_found = 0
    passage_count = 0

    with httpx.Client(follow_redirects=True) as http_client:
        for paper in papers:
            arxiv_id = paper["arxiv_id"]
            html_content = (CACHED_HTML_DIRECTORY / f"{arxiv_id}.html").read_text(
                encoding="utf-8"
            )
            loaded_paper = load_paper(arxiv_id, http_client)

            for passage in loaded_paper.passages:
                passage_count += 1
                anchor = passage.location.lstrip("#")
                if f'id="{anchor}"' in html_content:
                    anchors_found += 1
                else:
                    print(
                        f"Anchor not found in HTML: {passage.location} "
                        f"for passage: {passage.text[:60]}..."
                    )

    coverage = anchors_found / passage_count if passage_count else 0.0
    print(f"Anchors found in HTML: {anchors_found}/{passage_count}")
    print(f"Anchor coverage: {coverage:.2%}")


if __name__ == "__main__":
    check_passage_anchor_validity()
