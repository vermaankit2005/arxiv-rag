"""Turn the generator's [P1] markers into citations a browser can render.

The terminal renderer in arxiv_rag.answering writes ANSI escape codes, which a
browser shows as junk. This module is the browser's version of that job.
"""

from arxiv_rag.answering.generator import CITATION_ID_PATTERN, CITATION_MARKER_PATTERN, citation_ids_in_text
from arxiv_rag.retrieval import Citation


def citation_number(citation_id: str) -> str:
    """Turn a passage ID such as "P3" into the "3" a reader sees."""
    return citation_id.removeprefix("P")


def link_citation_markers(answer: str, citations: dict[str, Citation]) -> str:
    """Replace every [P1] marker with a Markdown link to its passage."""

    def replace_marker(match) -> str:
        citation_ids = CITATION_ID_PATTERN.findall(match.group())
        # An unknown ID means the model invented it, so leave the text alone.
        if not all(citation_id in citations for citation_id in citation_ids):
            return match.group()

        links = []
        for citation_id in citation_ids:
            number = citation_number(citation_id)
            links.append(f"[[{number}]]({citations[citation_id].url})")
        return " ".join(links)

    return CITATION_MARKER_PATTERN.sub(replace_marker, answer)


def build_sources(answer: str, citations: dict[str, Citation], passages_by_id: dict[str, str]) -> list[dict]:
    """List the cited passages, in the order the answer first uses them."""
    sources = []
    for citation_id in citation_ids_in_text(answer):
        citation = citations.get(citation_id)
        if citation is None:
            continue
        sources.append(
            {
                "number": citation_number(citation_id),
                "label": citation.label,
                "url": citation.url,
                "text": passages_by_id.get(citation_id, ""),
            }
        )
    return sources
