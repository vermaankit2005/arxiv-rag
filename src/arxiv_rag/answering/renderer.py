from arxiv_rag.retrieval import Citation


def render_answer(answer: str, citations: dict[str, Citation], clickable: bool) -> str:
    """Replace passage IDs with compact trusted citations."""
    sources = []

    for citation_id, citation in citations.items():
        marker = f"[{citation_id}]"
        if marker not in answer:
            continue

        number = citation_id.removeprefix("P")
        if clickable:
            link = f"\033]8;;{citation.url}\033\\[{number}]\033]8;;\033\\"
            answer = answer.replace(marker, link)
        else:
            answer = answer.replace(marker, f"[{number}]")
            sources.append(f"[{number}] {citation.label}\n{citation.url}")

    if not sources:
        return answer

    return f"{answer}\n\nSources:\n" + "\n\n".join(sources)
