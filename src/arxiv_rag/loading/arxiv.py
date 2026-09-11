from pathlib import Path

import httpx

from arxiv_rag.loading.html_parser import ArxivHtmlParser
from arxiv_rag.loading.models import LoadedPaper
from arxiv_rag.logging import get_logger

log = get_logger(__name__)

NO_HTML_NOTE = "no arXiv HTML published"

# Fetch HTML from arXiv, caching it locally. The cache is a simple text file,
# empty if arXiv has no HTML for the paper.
def _fetch_arxiv_html(arxiv_id: str, client: httpx.Client, html_dir: Path) -> str | None:
    """Download the LaTeXML page. Returns None when arXiv published none."""

    html_dir.mkdir(parents=True, exist_ok=True)
    cached = html_dir / f"{arxiv_id.replace('/', '_')}.html"

    if cached.exists():
        text = cached.read_text(encoding="utf-8", errors="ignore")
        return None if text == "" else text

    r = client.get(f"https://arxiv.org/html/{arxiv_id}", timeout=60.0)

    if r.status_code != 200:
        r.raise_for_status()
        raise RuntimeError(f"Unexpected response from arXiv: HTTP {r.status_code}")

    # arXiv answers with a "no HTML for this paper" stub, not a 404.
    missing = "ltx_page_main" not in r.text

    cached.write_text("" if missing else r.text, encoding="utf-8")
    return None if missing else r.text


# Fetch and parse an arXiv paper into the application's loading model.
def _load_paper_from_html(arxiv_id: str, arxiv_html: str) -> LoadedPaper:
    parser = ArxivHtmlParser()
    parser.feed(arxiv_html)
    parser.passages.extend(parser.pending)

    for i, passage in enumerate(parser.passages):
        passage.order = i

    return LoadedPaper(
        arxiv_id,
        parser.passages,
        images=parser.images,
    )

# Fetch and parse an arXiv paper into the application's loading model.
def load_paper(arxiv_id: str, client: httpx.Client, html_dir: Path) -> LoadedPaper:
    html = _fetch_arxiv_html(arxiv_id, client, html_dir)

    if html is None:
        log.warning("arXiv %s has no HTML published", arxiv_id)
        return LoadedPaper(arxiv_id, [], note=NO_HTML_NOTE)

    return _load_paper_from_html(arxiv_id, html)
