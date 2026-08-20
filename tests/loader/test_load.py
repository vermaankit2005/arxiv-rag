from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pytest

from loader.load import _fetch_html, load

FIXTURE = Path(__file__).parents[1] / "data" / "test_html.html"

# What arXiv actually sends back. A real page carries ltx_page_main; a paper
# with no HTML gets a 200 and a stub, never a 404, so the status code alone
# cannot tell the two apart.
REAL_PAGE = '<html><div class="ltx_page_main">the paper</div></html>'
STUB_PAGE = "<html><body>No HTML is available for this paper.</body></html>"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Send the download cache to a throwaway folder, never data/raw/html."""
    monkeypatch.setattr("loader.load.HTML_DIR", tmp_path)
    return tmp_path


def fake_response(status: int, text: str) -> Mock:
    reply = Mock()
    reply.status_code = status
    reply.text = text
    return reply


def test_load_html_returns_loaded_when_fetch_returns_html():
    client = Mock()
    mock_html = FIXTURE.read_text(encoding="utf-8")

    with patch("loader.load._fetch_html",
               return_value=mock_html) as mock_fetch_html:
        loaded = load("test_html", client)
        mock_fetch_html.assert_called_once_with("test_html", client)

        assert loaded.arxiv_id == "test_html"
        assert len(loaded.sections) == 31
        # 71 prose passages + 9 captions + 4 serialised data tables.
        assert len(loaded.passages) == 84


def test_load_html_returns_empty_loded_when_fetch_returns_none():
    client = Mock()

    with patch("loader.load._fetch_html",
               return_value=None) as mock_fetch_html:
        loaded = load("dummy_arxiv_id", client)
        mock_fetch_html.assert_called_once_with("dummy_arxiv_id", client)

        assert loaded.arxiv_id == "dummy_arxiv_id"
        assert loaded.sections == []
        assert loaded.passages == []
        assert loaded.note == "no arXiv HTML published"


def test_every_passage_has_a_location_that_is_in_the_page():
    mock_html = FIXTURE.read_text(encoding="utf-8")

    with patch("loader.load._fetch_html", return_value=mock_html):
        loaded = load("test_html", Mock())

    for passage in loaded.passages:
        assert passage.location, f"passage {passage.order} has no location: {passage.text[:60]}"

        anchor = passage.location.lstrip("#")
        assert f'id="{anchor}"' in mock_html, (
            f"passage {passage.order} points at {passage.location}, "
            f"which is not an id in the page"
        )


def test_loading_the_same_paper_twice_gives_the_same_result():
    mock_html = FIXTURE.read_text(encoding="utf-8")

    with patch("loader.load._fetch_html", return_value=mock_html):
        first = load("test_html", Mock())
        second = load("test_html", Mock())

    assert first == second


def load_fixture():
    mock_html = FIXTURE.read_text(encoding="utf-8")
    with patch("loader.load._fetch_html", return_value=mock_html):
        return load("test_html", Mock()), mock_html


def test_figure_two_caption_is_one_complete_citable_passage():
    """The sentence that explains Figure 2 used to be absent from every passage."""
    loaded, mock_html = load_fixture()
    expected = (
        "Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention "
        "consists of several attention layers running in parallel."
    )
    matches = [passage for passage in loaded.passages if passage.text == expected]

    assert len(matches) == 1
    assert matches[0].location == "#S3.F2"
    assert 'id="S3.F2"' in mock_html


def test_every_figure_and_table_caption_becomes_a_typed_passage():
    """All nine captions retain the type retrieval needs to present them."""
    loaded, _ = load_fixture()
    expected_kinds = {
        "#S3.F1": "figure_caption",
        "#S3.F2": "figure_caption",
        "#S4.T1": "table_caption",
        "#S6.T2": "table_caption",
        "#S6.T3": "table_caption",
        "#S6.T4": "table_caption",
        "#Sx1.F3": "figure_caption",
        "#Sx1.F4": "figure_caption",
        "#Sx1.F5": "figure_caption",
    }
    captions = {
        passage.location: passage.kind
        for passage in loaded.passages
        if passage.location in expected_kinds
    }

    assert captions == expected_kinds


def test_table_two_keeps_its_header_with_the_big_transformer_score():
    """A score without its BLEU header cannot answer what the big Transformer achieved."""
    loaded, _ = load_fixture()
    table = next(passage for passage in loaded.passages if passage.location == "#S6.T2.2")

    assert table.kind == "table"
    assert "Model | BLEU" in table.text
    assert "Transformer (big) | 28.4 | 41.8" in table.text


def test_equation_layout_tables_do_not_become_data_table_passages():
    """LaTeXML uses table markup for equations, which must not create duplicate passages."""
    loaded, _ = load_fixture()
    equation_locations = {"#S3.E1", "#S3.EGx1", "#S3.E2", "#S3.EGx2", "#S5.E3"}

    assert equation_locations.isdisjoint(
        passage.location for passage in loaded.passages
    )


def test_math_in_a_data_table_uses_alttext_exactly_once():
    """MathML repeats a formula visually; only its readable alttext belongs in a cell."""
    loaded, _ = load_fixture()
    table = next(passage for passage in loaded.passages if passage.location == "#S4.T1.2")

    assert table.text.count(r"O(n^{2}\cdot d)") == 1


def test_only_real_figure_images_are_recorded_with_urls_and_anchors():
    """Raster and SVG figures matter; arXiv logos and banner icons do not."""
    loaded, mock_html = load_fixture()
    expected_urls = {
        "https://arxiv.org/html/1706.03762v7/Figures/ModalNet-21.png",
        "https://arxiv.org/html/1706.03762v7/Figures/ModalNet-19.png",
        "https://arxiv.org/html/1706.03762v7/Figures/ModalNet-20.png",
        "https://arxiv.org/html/1706.03762v7/making_more_difficult5_new.svg",
        "https://arxiv.org/html/1706.03762v7/anaphora_resolution_new.svg",
        "https://arxiv.org/html/1706.03762v7/anaphora_resolution2_new.svg",
        "https://arxiv.org/html/1706.03762v7/attending_to_head_new.svg",
        "https://arxiv.org/html/1706.03762v7/attending_to_head2_new.svg",
    }

    assert {image.url for image in loaded.images} == expected_urls
    assert len(loaded.images) == 8
    for image in loaded.images:
        assert image.location.startswith("#")
        assert f'id="{image.location.lstrip("#")}"' in mock_html


def test_every_figure_image_is_linked_to_its_caption_passage():
    """A retrieved caption must carry every image the interface should show."""
    loaded, _ = load_fixture()
    figure_captions = [
        passage for passage in loaded.passages if passage.kind == "figure_caption"
    ]
    expected_locations = {
        "#S3.F1": ["#S3.F1.g1"],
        "#S3.F2": ["#S3.F2.g1", "#S3.F2.g2"],
        "#Sx1.F3": ["#Sx1.F3.g1"],
        "#Sx1.F4": ["#Sx1.F4.g1", "#Sx1.F4.g2"],
        "#Sx1.F5": ["#Sx1.F5.g1", "#Sx1.F5.g2"],
    }
    actual_locations = {
        passage.location: [image.location for image in passage.images]
        for passage in figure_captions
    }
    linked_images = [image for passage in figure_captions for image in passage.images]

    assert actual_locations == expected_locations
    assert linked_images == loaded.images
    assert all(
        not passage.images
        for passage in loaded.passages
        if passage.kind != "figure_caption"
    )


def test_prose_and_notes_retain_their_passage_kinds():
    loaded, _ = load_fixture()
    kinds_by_location = {passage.location: passage.kind for passage in loaded.passages}

    assert kinds_by_location["#S3.p1"] == "prose"
    assert kinds_by_location["#footnote1"] == "note"


def test_caption_is_emitted_beside_its_figure_in_document_order():
    """Captions appended after parsing would lose the reading order around a figure."""
    loaded, _ = load_fixture()
    caption = next(passage for passage in loaded.passages if passage.location == "#S3.F2")
    following_paragraph = next(
        passage
        for passage in loaded.passages
        if passage.location == "#S3.SS2.SSS2.p1"
    )

    assert [passage.order for passage in loaded.passages] == list(range(84))
    assert caption.order < following_paragraph.order


# -- downloading -----------------------------------------------------------
# Everything above fakes the download. These four are the only tests that
# exercise the code which actually talks to arXiv.


def test_a_downloaded_paper_is_returned_and_kept_on_disk(cache):
    client = Mock()
    client.get.return_value = fake_response(200, REAL_PAGE)

    assert _fetch_html("1706.03762v7", client) == REAL_PAGE
    assert (cache / "1706.03762v7.html").read_text(encoding="utf-8") == REAL_PAGE


def test_the_no_html_stub_is_not_mistaken_for_a_paper(cache):
    """arXiv answers 200 for a paper it has no HTML for. Trusting the status
    code would hand the parser a page with no paper in it."""
    client = Mock()
    client.get.return_value = fake_response(200, STUB_PAGE)

    assert _fetch_html("2101.00001", client) is None


def test_a_paper_already_on_disk_is_not_downloaded_again(cache):
    (cache / "1706.03762v7.html").write_text(REAL_PAGE, encoding="utf-8")
    client = Mock()

    assert _fetch_html("1706.03762v7", client) == REAL_PAGE
    client.get.assert_not_called()


def test_a_paper_known_to_have_no_html_is_not_asked_for_twice(cache):
    """The empty cache file means "arXiv has none of this one" -- without it
    every run re-asks arXiv for the papers it already said no to."""
    client = Mock()
    client.get.return_value = fake_response(200, STUB_PAGE)

    assert _fetch_html("2101.00001", client) is None
    assert _fetch_html("2101.00001", client) is None
    assert client.get.call_count == 1


def test_arxiv_being_unreachable_does_not_stop_the_run(cache):
    """One unreachable paper must not take the other eleven down with it."""
    client = Mock()
    client.get.side_effect = httpx.ConnectError("no route to host")

    assert _fetch_html("1706.03762v7", client) is None


def test_arxiv_being_unreachable_is_not_recorded_as_no_html(cache):
    """The empty cache file means "arXiv has none of this paper", which is
    permanent. A dropped connection is not. Writing one for the other would
    retire a perfectly good paper the first time the wifi blinks."""
    client = Mock()
    client.get.side_effect = httpx.ConnectError("no route to host")

    _fetch_html("1706.03762v7", client)
    assert not (cache / "1706.03762v7.html").exists()

    # Network comes back: the paper is fetched, not skipped forever.
    client.get.side_effect = None
    client.get.return_value = fake_response(200, REAL_PAGE)
    assert _fetch_html("1706.03762v7", client) == REAL_PAGE
