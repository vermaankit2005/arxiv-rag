import hashlib
import re
from collections import Counter
from pathlib import Path

import httpx
import unicodedata
from dotenv import load_dotenv
from langsmith import Client

from loader.load import load

ROOT = Path(__file__).parents[2]
CACHED_HTML_DIRECTORY = ROOT / "data" / "raw" / "sampled_html"
LANGSMITH_DATASET_NAME = "html_content_retention_dataset"
WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


def cached_html_path(arxiv_id: str) -> Path:
    """Return the cache path without allowing an id to escape the cache folder."""
    safe_filename = f"{arxiv_id.replace('/', '_')}.html"
    path = (CACHED_HTML_DIRECTORY / safe_filename).resolve()
    if path.parent != CACHED_HTML_DIRECTORY.resolve():
        raise ValueError(f"Invalid arXiv id: {arxiv_id}")
    return path


def normalized_words(text: str) -> list[str]:
    """Reduce harmless case, Unicode and punctuation differences."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return WORD_PATTERN.findall(normalized)


def load_paper_for_content_retention(inputs: dict) -> dict:
    """Run the shipping loader and return simple values LangSmith can store."""
    arxiv_id = inputs["arxiv_id"]
    html_content = cached_html_path(arxiv_id).read_bytes()

    with httpx.Client(follow_redirects=True) as http_client:
        loaded = load(arxiv_id, http_client)

    return {
        "html_sha256": hashlib.sha256(html_content).hexdigest(),
        "passages": [
            {
                "anchor": passage.location.lstrip("#"),
                "kind": passage.kind,
                "text": passage.text,
            }
            for passage in loaded.passages
        ],
    }


def evaluate_html_content_retention(inputs: dict, outputs: dict, reference_outputs: dict) -> list[dict]:
    """Compare every expected HTML block with the passage at the same anchor."""
    arxiv_id = inputs["arxiv_id"]

    if outputs["html_sha256"] != reference_outputs["html_sha256"]:
        raise ValueError(
            f"Cached HTML changed for {arxiv_id}; rebuild the frozen dataset first"
        )

    expected_blocks = reference_outputs["usable_blocks"]
    actual_passages = {
        (passage["anchor"], passage["kind"]): passage for passage in outputs["passages"]
    }

    found_blocks = 0
    expected_word_count = 0
    retained_word_count = 0
    missing_blocks: list[str] = []
    missing_words: Counter[str] = Counter()

    for expected in expected_blocks:
        key = (expected["anchor"], expected["kind"])
        expected_word_counts = Counter(normalized_words(expected["text"]))
        expected_word_count += sum(expected_word_counts.values())

        actual = actual_passages.get(key)
        if actual is None:
            missing_blocks.append(f"{expected['kind']} #{expected['anchor']}")
            missing_words.update(expected_word_counts)
            continue

        found_blocks += 1
        actual_word_counts = Counter(normalized_words(actual["text"]))
        retained_word_count += sum((expected_word_counts & actual_word_counts).values())
        missing_words.update(expected_word_counts - actual_word_counts)

    block_coverage = found_blocks / len(expected_blocks) if expected_blocks else 0.0
    word_retention = (
        retained_word_count / expected_word_count if expected_word_count else 0.0
    )

    block_comment = f"Found {found_blocks}/{len(expected_blocks)} expected blocks." + (
        f" Missing: {', '.join(missing_blocks[:5])}" if missing_blocks else ""
    )
    common_missing_words = ", ".join(
        f"{word} ({count})" for word, count in missing_words.most_common(10)
    )
    word_comment = (
            f"Retained {retained_word_count}/{expected_word_count} expected words."
            + (f" Most common missing: {common_missing_words}" if missing_words else "")
    )

    return [
        {
            "key": "html_block_coverage",
            "score": block_coverage,
            "comment": block_comment,
        },
        {
            "key": "html_word_retention",
            "score": word_retention,
            "comment": word_comment,
        },
    ]


def run_html_content_retention_evaluation() -> None:
    """Run the evaluator against the frozen LangSmith dataset."""
    load_dotenv()
    client = Client()
    client.evaluate(
        load_paper_for_content_retention,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[evaluate_html_content_retention],
        experiment_prefix="html_content_retention",
    )


if __name__ == "__main__":
    run_html_content_retention_evaluation()
