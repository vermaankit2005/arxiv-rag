from langsmith import traceable

from .generator import CITATION_ID_PATTERN, CITATION_MARKER_PATTERN
from arxiv_rag.retrieval import Citation


@traceable(
    name="render_answer",
    process_inputs=lambda inputs: {
        "answer": inputs["answer"],
        "clickable": inputs["clickable"],
    },
    process_outputs=lambda outputs: {"rendered": outputs},
)
def render_answer(answer: str, citations: dict[str, Citation], clickable: bool) -> str:
    """Replace passage IDs with compact trusted citations."""
    used_ids = []

    def replace_marker(match) -> str:
        citation_ids = CITATION_ID_PATTERN.findall(match.group())
        if not all(citation_id in citations for citation_id in citation_ids):
            return match.group()

        rendered = []
        for citation_id in citation_ids:
            if citation_id not in used_ids:
                used_ids.append(citation_id)
            citation = citations[citation_id]
            number = citation_id.removeprefix("P")
            if clickable:
                rendered.append(f"\033]8;;{citation.url}\033\\[{number}]\033]8;;\033\\")
            else:
                rendered.append(f"[{number}]")
        return " ".join(rendered)

    answer = CITATION_MARKER_PATTERN.sub(replace_marker, answer)
    if clickable or not used_ids:
        return answer

    sources = []
    for citation_id in used_ids:
        citation = citations[citation_id]
        number = citation_id.removeprefix("P")
        sources.append(f"[{number}] {citation.label}\n{citation.url}")
    return f"{answer}\n\nSources:\n" + "\n\n".join(sources)
