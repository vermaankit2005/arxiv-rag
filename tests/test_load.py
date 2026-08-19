from pathlib import Path
from unittest.mock import Mock, patch

from loader.load import load

FIXTURE = Path(__file__).parent / "data" / "test_html.html"


def test_load_html_returns_loaded_when_fetch_returns_html():
    client = Mock()
    mock_html = FIXTURE.read_text(encoding="utf-8")

    with patch("loader.load._fetch_html",
               return_value=mock_html) as mock_fetch_html:
        loaded = load("test_html", client)
        mock_fetch_html.assert_called_once_with("test_html", client)

        assert loaded.arxiv_id == "test_html"
        assert len(loaded.sections) == 31
        assert len(loaded.passages) == 83


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
