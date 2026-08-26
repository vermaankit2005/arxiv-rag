import json

from arxiv_rag.ingestion.documents import (
    _build_page_content,
    _group_passages,
    _overlap_group_passages,
    convert_loaded_paper_to_documents,
)
from arxiv_rag.loading.models import LoadedPaper, Passage


def _passage(order: int, section_path: list[str], words: int = 2) -> Passage:
    return Passage(
        order=order,
        text=" ".join([f"passage-{order}"] * words),
        section=section_path[-1] if section_path else "",
        section_path=section_path,
        location=f"#p{order}",
    )


def _paper(passages: list[Passage]) -> LoadedPaper:
    return LoadedPaper(arxiv_id="test-paper", sections=[], passages=passages)


def test_page_content_adds_each_section_breadcrumb_once():
    first = _passage(1, ["Training", "Optimizer"])
    second = _passage(2, ["Training", "Optimizer"])
    third = _passage(3, ["Training", "Regularization"])

    content = _build_page_content([first, second, third])

    assert content == (
        "Section: Training > Optimizer\n\n"
        "passage-1 passage-1\n\n"
        "passage-2 passage-2\n\n"
        "Section: Training > Regularization\n\n"
        "passage-3 passage-3"
    )


def test_page_content_labels_passages_without_a_section():
    passage = _passage(1, [])

    content = _build_page_content([passage])

    assert content == "Section: Unsectioned\n\npassage-1 passage-1"


def test_document_metadata_preserves_each_source_passage_and_anchor():
    first = _passage(1, ["Training", "Optimizer"])
    second = _passage(2, ["Training", "Regularization"])

    documents = convert_loaded_paper_to_documents(_paper([first, second]))
    source_passages = json.loads(documents[0].metadata["source_passages"])

    assert source_passages == [
        {
            "text": first.text,
            "location": "#p1",
            "section_path": ["Training", "Optimizer"],
            "kind": "prose",
        },
        {
            "text": second.text,
            "location": "#p2",
            "section_path": ["Training", "Regularization"],
            "kind": "prose",
        },
    ]
    assert json.loads(documents[0].metadata["locations"]) == ["#p1", "#p2"]


def test_grouping_crosses_subsection_boundaries_when_content_fits():
    first = _passage(1, ["Training", "Optimizer"], words=200)
    second = _passage(2, ["Training", "Regularization"], words=100)

    groups = _group_passages(_paper([first, second]))

    assert groups == [[first, second]]


def test_grouping_splits_before_exceeding_max_words():
    first = _passage(1, ["Training", "Optimizer"], words=300)
    second = _passage(2, ["Training", "Optimizer"], words=100)

    groups = _group_passages(_paper([first, second]))

    assert groups == [[first], [second]]


def test_grouping_never_crosses_main_section_boundaries():
    first = _passage(1, ["Introduction"], words=20)
    second = _passage(2, ["Methods"], words=20)

    groups = _group_passages(_paper([first, second]))

    assert groups == [[first], [second]]


def test_overlap_stays_within_the_same_main_section():
    first = _passage(1, ["Introduction"])
    second = _passage(2, ["Introduction"])
    third = _passage(3, ["Methods"])
    groups = [[first], [second], [third]]

    overlapped = _overlap_group_passages(groups)

    assert overlapped == [[first], [first, second], [third]]
