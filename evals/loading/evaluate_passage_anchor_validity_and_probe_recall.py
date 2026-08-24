"""Evaluate passage-anchor validity and sampled text recall in LangSmith."""

import hashlib
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langsmith import Client  # pyright: ignore[reportMissingImports]

from arxiv_rag.loading import load_paper

ROOT = Path(__file__).parents[2]
CACHED_HTML_DIRECTORY = ROOT / "data" / "raw" / "sampled_html"
LANGSMITH_DATASET_NAME = "sampled_probe_recall_dataset"


def cached_html_path(arxiv_id: str) -> Path:
    """Return the cache path without allowing an id to escape the cache folder."""
    safe_filename = f"{arxiv_id.replace('/', '_')}.html"
    path = (CACHED_HTML_DIRECTORY / safe_filename).resolve()
    if path.parent != CACHED_HTML_DIRECTORY.resolve():
        raise ValueError(f"Invalid arXiv id: {arxiv_id}")
    return path


def load_passages_for_evaluation(inputs: dict) -> dict:
    """Run the paper-loading pipeline and return serializable values."""
    arxiv_id = inputs["arxiv_id"]
    html_content = cached_html_path(arxiv_id).read_bytes()

    with httpx.Client(follow_redirects=True) as http_client:
        paper = load_paper(arxiv_id, http_client)

    return {
        "html_sha256": hashlib.sha256(html_content).hexdigest(),
        "passage_anchors": [
            passage.location.lstrip("#") for passage in paper.passages
        ],
        "passage_texts": [passage.text for passage in paper.passages],
    }


def evaluate_passage_anchor_validity_and_probe_recall(inputs: dict, outputs: dict, reference_outputs: dict) -> list[dict]:
    """Score valid passage anchors and exact sampled-probe recall."""
    arxiv_id = inputs["arxiv_id"]

    if outputs["html_sha256"] != reference_outputs["html_sha256"]:
        raise ValueError(
            f"Cached HTML changed for {arxiv_id}; rebuild the frozen dataset first"
        )

    html_content = cached_html_path(arxiv_id).read_text(encoding="utf-8")
    passage_anchors = outputs["passage_anchors"]
    valid_anchor_count = sum(f'id="{anchor}"' in html_content for anchor in passage_anchors)

    anchor_coverage = (
        valid_anchor_count / len(passage_anchors) if passage_anchors else 0.0
    )

    probes = reference_outputs["probes"]
    passage_texts = outputs["passage_texts"]

    recalled_probe_count = sum(
        any(probe["text"] in passage_text for passage_text in passage_texts)
        for probe in probes
    )
    text_recall = recalled_probe_count / len(probes) if probes else 0.0

    return [
        {
            "key": "anchor_coverage",
            "score": anchor_coverage,
            "comment": f"Resolved {valid_anchor_count}/{len(passage_anchors)} anchors.",
        },
        {
            "key": "text_recall",
            "score": text_recall,
            "comment": f"Found {recalled_probe_count}/{len(probes)} probes.",
        },
    ]


def run_passage_anchor_and_probe_recall_evaluation() -> None:
    """Run both evaluators against the frozen LangSmith dataset."""
    load_dotenv()
    langsmith_client = Client()
    langsmith_client.evaluate(
        load_passages_for_evaluation,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[evaluate_passage_anchor_validity_and_probe_recall],
        experiment_prefix="sampled_probe_recall",
    )


if __name__ == "__main__":
    run_passage_anchor_and_probe_recall_evaluation()
