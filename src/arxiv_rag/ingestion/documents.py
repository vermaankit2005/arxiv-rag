import json
import re
from dataclasses import replace
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document

from arxiv_rag.loading.models import LoadedPaper, Passage

PASSAGE_SPLIT_THRESHOLD_WORDS = 600
MAX_WORDS = 350


def _document_id(arxiv_id: str, location: str, content: str) -> str:
    source = json.dumps([arxiv_id, location, content], ensure_ascii=False)
    return str(uuid5(NAMESPACE_URL, source))


def _should_include(passage: Passage) -> bool:
    text = " ".join(passage.text.split())

    if not text:
        return False

    if not passage.location:
        return False

    return re.fullmatch(r"https?://\S+", text) is None


def _main_section(passage: Passage) -> str:
    if not passage.section_path:
        return ""
    return passage.section_path[0]


def _word_count(passage: Passage) -> int:
    return len(passage.text.split())


def _breadcrumb(passage: Passage) -> str:
    if not passage.section_path:
        return "Unsectioned"
    return " > ".join(passage.section_path)


def _build_page_content(group: list[Passage]) -> str:
    parts = []
    previous_section_path = None

    for passage in group:
        if passage.section_path != previous_section_path:
            parts.append(f"Section: {_breadcrumb(passage)}")
            previous_section_path = passage.section_path
        parts.append(passage.text)

    return "\n\n".join(parts)


def _process_oversize_passages(passage: Passage) -> list[Passage]:
    if _word_count(passage) <= PASSAGE_SPLIT_THRESHOLD_WORDS:
        return [passage]

    separator = "\n" if passage.kind == "table" else " "
    if passage.kind == "table":
        units = [line.strip() for line in passage.text.splitlines() if line.strip()]
    else:
        units = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", passage.text)
            if sentence.strip()
        ]

    parts = []
    current_units = []
    current_words = 0

    for unit in units:

        words = unit.split()
        normalized_unit = " ".join(words)

        if current_units and current_words + len(words) > MAX_WORDS:
            parts.append(separator.join(current_units))
            current_units = []
            current_words = 0

        current_units.append(normalized_unit)
        current_words += len(words)

    # To handle any remaining units that didn't exceed the MAX_WORDS limit
    if current_units:
        parts.append(separator.join(current_units))

    return [replace(passage, text=part) for part in parts]


def _group_passages(loaded_paper: LoadedPaper) -> list[list[Passage]]:

    """Group passages by section path."""
    groups = []
    # We will build the current group then add it to the groups list.
    current_group = []
    current_words = 0

    for passage in loaded_paper.passages:

        # This looks complicated, but it's just splitting up passages that are too long into smaller passages.
        # Split if the passage is too long, otherwise just return the passage as a list of one.
        passage_list = _process_oversize_passages(passage)

        for _passage in passage_list:
            if not _should_include(_passage):
                continue

            main_section_changed = current_group and _main_section(_passage) != _main_section(current_group[0])

            would_be_too_long = current_words + _word_count(_passage) > MAX_WORDS

            if current_group and (main_section_changed or would_be_too_long):
                groups.append(current_group)
                current_group = []
                current_words = 0

            current_group.append(_passage)
            current_words += _word_count(_passage)

    if current_group:
        groups.append(current_group)

    return groups


def _overlap_group_passages(groups: list[list[Passage]]) -> list[list[Passage]]:
    """Overlap groups of passages by one passage."""
    if not groups:
        return []

    overlapped_groups = [groups[0]]

    for i in range(1, len(groups)):

        previous_group = groups[i - 1]
        current_group = groups[i]

        same_main_section =  previous_group and current_group and _main_section(previous_group[-1]) == _main_section(current_group[0])

        if same_main_section:
            overlapped_groups.append([previous_group[-1]] + current_group)
        else:
            overlapped_groups.append(current_group)

    return overlapped_groups


# This is convert a given loaded paper into a list of Documents.
def convert_loaded_paper_to_documents(loaded_paper: LoadedPaper) -> list[Document]:
    """Convert a loaded paper into retrieval documents."""
    documents = []

    grouped_passages = _overlap_group_passages(_group_passages(loaded_paper))

    for group in grouped_passages:

        locations = [passage.location for passage in group]
        # Preserve the original passage text, location, section path, and kind in the metadata for each passage.
        # !!! Important for retrieval and context building.
        source_passages = [
            {
                "text": passage.text,
                "location": passage.location,
                "section_path": passage.section_path,
                "kind": passage.kind,
            }
            for passage in group
        ]

        doc_content = _build_page_content(group)

        doc = Document(
            id=_document_id(loaded_paper.arxiv_id, "|".join(locations), doc_content),
            page_content=doc_content,
            metadata={
                "arxiv_id": loaded_paper.arxiv_id,
                # if we ever need to re-embed , we can remove location as it is already in source_passages
                "locations": json.dumps(locations),
                "source_passages": json.dumps(source_passages),
                "images": json.dumps(
                    [
                        {"url": image.url, "location": image.location}
                        for passage in group
                        for image in passage.images
                    ]
                ),
            },
        )
        documents.append(doc)
    return documents
