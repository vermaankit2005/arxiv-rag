from arxiv_rag.answering import INSUFFICIENT_EVIDENCE_ANSWER
from arxiv_rag.retrieval import Citation

from ui import citations as ui_citations

CITATIONS = {
    "P1": Citation(label="1706.03762v7 — Attention", url="https://arxiv.org/html/1706.03762v7#S3"),
    "P2": Citation(label="1706.03762v7 — Results", url="https://arxiv.org/html/1706.03762v7#S6"),
}

PASSAGES = {
    "P1": "Multi-head attention runs several attention layers in parallel.",
    "P2": "The model achieved 28.4 BLEU.",
}


def test_link_citation_markers_turns_a_single_marker_into_a_numbered_link():
    answer = "Attention runs in parallel [P1]."

    linked = ui_citations.link_citation_markers(answer, CITATIONS)

    assert linked == "Attention runs in parallel [[1]](https://arxiv.org/html/1706.03762v7#S3)."


def test_link_citation_markers_splits_a_grouped_marker_into_separate_links():
    answer = "Both points hold [P1, P2]."

    linked = ui_citations.link_citation_markers(answer, CITATIONS)

    assert "[[1]](https://arxiv.org/html/1706.03762v7#S3)" in linked
    assert "[[2]](https://arxiv.org/html/1706.03762v7#S6)" in linked
    assert "[P1, P2]" not in linked


def test_link_citation_markers_leaves_an_unknown_marker_as_written():
    answer = "This claim has no passage [P9]."

    linked = ui_citations.link_citation_markers(answer, CITATIONS)

    assert linked == answer


def test_link_citation_markers_leaves_the_abstention_answer_alone():
    linked = ui_citations.link_citation_markers(INSUFFICIENT_EVIDENCE_ANSWER, CITATIONS)

    assert linked == INSUFFICIENT_EVIDENCE_ANSWER


def test_build_sources_lists_cited_passages_in_first_use_order():
    answer = "Results first [P2]. Then the mechanism [P1]. Results again [P2]."

    sources = ui_citations.build_sources(answer, CITATIONS, PASSAGES)

    assert [source["number"] for source in sources] == ["2", "1"]
    assert sources[0]["label"] == "1706.03762v7 — Results"
    assert sources[0]["url"] == "https://arxiv.org/html/1706.03762v7#S6"
    assert sources[1]["text"] == "Multi-head attention runs several attention layers in parallel."


def test_build_sources_skips_ids_that_have_no_citation():
    answer = "A real claim [P1]. An invented one [P9]."

    sources = ui_citations.build_sources(answer, CITATIONS, PASSAGES)

    assert [source["number"] for source in sources] == ["1"]


def test_build_sources_returns_nothing_for_an_uncited_answer():
    assert ui_citations.build_sources(INSUFFICIENT_EVIDENCE_ANSWER, CITATIONS, PASSAGES) == []
