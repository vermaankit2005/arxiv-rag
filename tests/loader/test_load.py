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
        assert len(loaded.passages) == 71


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
