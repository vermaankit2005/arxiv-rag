"""Build the frozen HTML answer key for the loader retention eval.

The shipping loader uses Python's ``HTMLParser``. This builder deliberately uses
lxml instead, so a mistake in the loader cannot automatically appear in its own
answer key.

One LangSmith example represents one paper. Its reference output contains every
useful, citable HTML block in document order: prose and lists, anchored notes,
figure/table captions, and data tables.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client  # pyright: ignore[reportMissingImports]
from lxml import html  # pyright: ignore[reportMissingImports]
from lxml.html import HtmlElement  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).parents[2]
BENCHMARK_PAPERS_PATH = ROOT / "evals" / "dataset" / "papers.json"
CACHED_HTML_DIRECTORY = ROOT / "data" / "raw" / "sampled_html"
REFERENCE_DATASET_PATH = (
        ROOT / "evals" / "dataset" / "html_content_retention_dataset.json"
)
LANGSMITH_DATASET_NAME = "html_content_retention_dataset"

NOTE_CLASSES = {"ltx_note_content", "ltx_role_thanks", "ltx_role_footnote"}
NUMBERING_CLASSES = {"ltx_note_mark", "ltx_tag_note", "ltx_note_type"}
BIBLIOGRAPHY_CLASSES = {"ltx_bibliography", "ltx_bibitem"}
IGNORED_TAGS = {"head", "script", "style", "svg"}


def element_classes(element: HtmlElement) -> set[str]:
    """Return an element's space-separated CSS classes."""
    return set((element.get("class") or "").split())


def has_ancestor_with_any_class(element: HtmlElement, classes: set[str]) -> bool:
    """Return whether any parent carries one of the requested classes."""
    return any(element_classes(parent) & classes for parent in element.iterancestors())


def has_ancestor_with_any_tag(element: HtmlElement, tags: set[str]) -> bool:
    """Return whether this element sits inside an ignored HTML subtree."""
    return any(parent.tag in tags for parent in element.iterancestors())


def nearest_html_anchor(element: HtmlElement) -> str:
    """Find the closest id that can locate this block in the source HTML."""
    for candidate in (element, *element.iterancestors()):
        if candidate.get("id"):
            return candidate.get("id", "")
    return ""


def is_separate_content_block(element: HtmlElement) -> bool:
    """Return whether nested text must be emitted as its own reference block."""
    classes = element_classes(element)
    return bool(
        classes & NOTE_CLASSES
        or (element.tag == "figcaption" and "ltx_caption" in classes)
        or (element.tag == "table" and "ltx_tabular" in classes)
    )


def normalized_visible_text(element: HtmlElement, *, skip_nested_content_blocks: bool = True) -> str:
    """Read visible text once, without browser scaffolding or duplicate maths."""
    parts: list[str] = [element.text or ""]

    for child in element:
        tag = child.tag if isinstance(child.tag, str) else ""
        classes = element_classes(child)

        if tag == "math":
            parts.append(f" {child.get('alttext', '')} ")
        elif (
                tag in IGNORED_TAGS
                or classes & NUMBERING_CLASSES
                or "ltx_title" in classes
                or skip_nested_content_blocks
                and is_separate_content_block(child)
        ):
            pass
        else:
            parts.append(
                normalized_visible_text(
                    child,
                    skip_nested_content_blocks=skip_nested_content_blocks,
                )
            )

        parts.append(child.tail or "")

    return re.sub(r"\s+", " ", "".join(parts)).strip()


def serialized_data_table(table: HtmlElement) -> str:
    """Turn an HTML data table into the same readable row shape expected downstream."""
    rows: list[str] = []
    for row in table.xpath(".//tr"):
        cells = [
            normalized_visible_text(cell, skip_nested_content_blocks=False)
            for cell in row.xpath("./th | ./td")
        ]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def caption_kind(caption: HtmlElement) -> str:
    """Distinguish table captions from figure captions using their container."""
    for parent in caption.iterancestors():
        classes = element_classes(parent)
        if "ltx_table" in classes:
            return "table_caption"
        if "ltx_figure" in classes:
            return "figure_caption"
    return "figure_caption"


def classify_usable_block(element: HtmlElement) -> str | None:
    """Name a useful source block, or return None when it is not evidence."""
    classes = element_classes(element)

    if has_ancestor_with_any_tag(element, IGNORED_TAGS):
        return None

    if classes & BIBLIOGRAPHY_CLASSES or has_ancestor_with_any_class(
            element, BIBLIOGRAPHY_CLASSES
    ):
        return None

    if classes & NOTE_CLASSES:
        if has_ancestor_with_any_class(element, NOTE_CLASSES):
            return None
        return "note"

    if element.tag == "figcaption" and "ltx_caption" in classes:
        return caption_kind(element)

    if element.tag == "table" and "ltx_tabular" in classes:
        if has_ancestor_with_any_class(element, {"ltx_tabular"}):
            return None
        return "table"

    if "ltx_para" in classes:
        if has_ancestor_with_any_class(
                element,
                {"ltx_para", "ltx_tabular", "ltx_caption", *NOTE_CLASSES},
        ):
            return None
        return "prose"

    if element.tag == "p" and "ltx_p" in classes:
        if has_ancestor_with_any_class(
                element,
                {"ltx_para", "ltx_tabular", "ltx_caption", *NOTE_CLASSES},
        ):
            return None
        return "prose"

    return None


def cached_html_path(arxiv_id: str) -> Path:
    """Return the cache path without allowing an id to escape the cache folder."""
    safe_filename = f"{arxiv_id.replace('/', '_')}.html"
    path = (CACHED_HTML_DIRECTORY / safe_filename).resolve()
    if path.parent != CACHED_HTML_DIRECTORY.resolve():
        raise ValueError(f"Invalid arXiv id: {arxiv_id}")
    return path


def extract_usable_html_blocks(html_content: bytes) -> list[dict]:
    """Extract every useful textual block from one cached arXiv HTML page."""
    document = html.fromstring(html_content)
    articles = document.xpath(
        '//*[contains(concat(" ", normalize-space(@class), " "), " ltx_document ")]'
    )
    if not articles:
        raise ValueError("HTML has no ltx_document article")

    blocks: list[dict] = []
    for element in articles[0].iter():
        if not isinstance(element.tag, str):
            continue
        kind = classify_usable_block(element)
        if kind is None:
            continue

        anchor = nearest_html_anchor(element)
        if kind == "note" and not anchor:
            continue

        text = (
            serialized_data_table(element)
            if kind == "table"
            else normalized_visible_text(
                element,
                skip_nested_content_blocks=kind == "prose",
            )
        )
        if not text:
            continue

        blocks.append(
            {
                "order": len(blocks),
                "anchor": anchor,
                "kind": kind,
                "text": text,
            }
        )

    return blocks


def build_html_content_retention_reference() -> list[dict]:
    """Build and save the frozen local reference data for all benchmark papers."""
    try:
        papers = json.loads(BENCHMARK_PAPERS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not read benchmark papers from {BENCHMARK_PAPERS_PATH}"
        ) from error

    reference: list[dict] = []

    for paper in papers:
        arxiv_id = paper["arxiv_id"]
        html_content = cached_html_path(arxiv_id).read_bytes()
        blocks = extract_usable_html_blocks(html_content)
        reference.append(
            {
                "arxiv_id": arxiv_id,
                "html_sha256": hashlib.sha256(html_content).hexdigest(),
                "usable_blocks": blocks,
            }
        )
        print(f"{arxiv_id}: {len(blocks)} usable HTML blocks")

    REFERENCE_DATASET_PATH.write_text(
        json.dumps(reference, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {REFERENCE_DATASET_PATH.relative_to(ROOT)}")
    return reference


def create_html_content_retention_dataset() -> None:
    """Build the reference data and upload one LangSmith example per paper."""
    reference = build_html_content_retention_reference()
    load_dotenv()
    langsmith_client = Client()

    if langsmith_client.has_dataset(dataset_name=LANGSMITH_DATASET_NAME):

        existing = {
            existing_example.inputs["arxiv_id"]: existing_example.outputs
            for existing_example in langsmith_client.list_examples(dataset_name=LANGSMITH_DATASET_NAME)
        }

        expected = {
            paper["arxiv_id"]: {
                "html_sha256": paper["html_sha256"],
                "usable_blocks": paper["usable_blocks"],
            }
            for paper in reference
        }
        if existing == expected:
            print(f"LangSmith dataset is up to date: {LANGSMITH_DATASET_NAME}")
            return
        raise RuntimeError(
            f"LangSmith dataset is stale: {LANGSMITH_DATASET_NAME}. "
            "Delete it before rebuilding so old experiments are not changed silently."
        )

    langsmith_client.create_dataset(LANGSMITH_DATASET_NAME)
    for paper in reference:
        langsmith_client.create_example(
            dataset_name=LANGSMITH_DATASET_NAME,
            inputs={"arxiv_id": paper["arxiv_id"]},
            outputs={
                "html_sha256": paper["html_sha256"],
                "usable_blocks": paper["usable_blocks"],
            },
        )

    print(f"Created LangSmith dataset: {LANGSMITH_DATASET_NAME}")


if __name__ == "__main__":
    create_html_content_retention_dataset()
