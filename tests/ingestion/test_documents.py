import json

from arxiv_rag.ingestion.documents import (
    MAX_WORDS,
    _build_page_content,
    _group_passages,
    _overlap_group_passages,
    _process_oversize_passages,
    _should_include,
    _word_count,
    convert_loaded_paper_to_documents,
)
from arxiv_rag.loading.models import FigureImage, LoadedPaper, Passage


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


def test_passage_filter_rejects_content_that_cannot_be_retrieved_or_cited():
    whitespace = _passage(1, ["Introduction"])
    whitespace.text = "   \n"
    missing_anchor = _passage(2, ["Introduction"])
    missing_anchor.location = ""
    bare_url = _passage(3, ["Introduction"])
    bare_url.text = "https://example.com/paper"
    useful_prose = _passage(4, ["Introduction"])
    useful_prose.text = "See https://example.com for the complete results."

    assert not _should_include(whitespace)
    assert not _should_include(missing_anchor)
    assert not _should_include(bare_url)
    assert _should_include(useful_prose)


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


def test_oversized_prose_splits_at_sentence_boundaries():
    passage = _passage(1, ["Methods"])
    first_sentence = " ".join(["first"] * 200) + "."
    second_sentence = " ".join(["second"] * 200) + "."
    passage.text = f"{first_sentence} {second_sentence}"

    parts = _process_oversize_passages(passage)

    assert [part.text for part in parts] == [first_sentence, second_sentence]
    assert all(_word_count(part) <= MAX_WORDS for part in parts)
    assert all(part.location == passage.location for part in parts)
    assert all(part.section_path == passage.section_path for part in parts)


def test_oversized_table_splits_only_between_rows():
    passage = _passage(1, ["Results"])
    passage.kind = "table"
    header = "Model | Score"
    first_row = " ".join(["first-row"] * 200)
    second_row = " ".join(["second-row"] * 200)
    passage.text = f"{header}\n{first_row}\n{second_row}"

    parts = _process_oversize_passages(passage)

    assert [part.text for part in parts] == [f"{header}\n{first_row}", second_row]
    assert all(_word_count(part) <= MAX_WORDS for part in parts)


def test_one_oversized_sentence_is_kept_whole():
    passage = _passage(1, ["Methods"], words=701)

    parts = _process_oversize_passages(passage)

    assert len(parts) == 1
    assert parts[0].text.split() == passage.text.split()


def test_one_oversized_table_row_is_kept_whole():
    passage = _passage(1, ["Results"], words=701)
    passage.kind = "table"

    parts = _process_oversize_passages(passage)

    assert len(parts) == 1
    assert parts[0].text.split() == passage.text.split()


def test_grouping_keeps_one_oversized_unit_separate():
    oversized = _passage(1, ["Methods"], words=351)
    following = _passage(2, ["Methods"], words=20)

    groups = _group_passages(_paper([oversized, following]))

    assert [[_word_count(passage) for passage in group] for group in groups] == [
        [351],
        [20],
    ]


def test_grouping_never_crosses_main_section_boundaries():
    first = _passage(1, ["Introduction"], words=20)
    second = _passage(2, ["Methods"], words=20)

    groups = _group_passages(_paper([first, second]))

    assert groups == [[first], [second]]


def test_conversion_returns_no_documents_when_every_passage_is_filtered():
    empty = _passage(1, ["Introduction"])
    empty.text = ""
    missing_anchor = _passage(2, ["Introduction"])
    missing_anchor.location = ""

    assert convert_loaded_paper_to_documents(_paper([empty, missing_anchor])) == []


def test_document_metadata_collects_images_from_every_source_passage():
    first = _passage(1, ["Results"])
    first.images = [FigureImage(url="https://example.com/figure-1.png", location="#F1")]
    second = _passage(2, ["Results"])
    second.images = [FigureImage(url="https://example.com/figure-2.png", location="#F2")]

    document = convert_loaded_paper_to_documents(_paper([first, second]))[0]

    assert json.loads(document.metadata["images"]) == [
        {"url": "https://example.com/figure-1.png", "location": "#F1"},
        {"url": "https://example.com/figure-2.png", "location": "#F2"},
    ]


def test_document_ids_are_stable_and_change_with_the_paper():
    passages = [_passage(1, ["Results"]), _passage(2, ["Results"])]

    first_id = convert_loaded_paper_to_documents(_paper(passages))[0].id
    repeated_id = convert_loaded_paper_to_documents(_paper(passages))[0].id
    other_paper = LoadedPaper(arxiv_id="other-paper", sections=[], passages=passages)
    other_id = convert_loaded_paper_to_documents(other_paper)[0].id

    assert first_id == repeated_id
    assert first_id != other_id


def test_overlap_stays_within_the_same_main_section():
    first = _passage(1, ["Introduction"])
    second = _passage(2, ["Introduction"])
    third = _passage(3, ["Methods"])
    groups = [[first], [second], [third]]

    overlapped = _overlap_group_passages(groups)

    assert overlapped == [[first], [first, second], [third]]
