from pathlib import Path

import httpx

from loader.renderer import Reader
from loader.shape import Loaded
from util.log import get_logger

ROOT = Path(__file__).resolve().parents[2]
HTML_DIR = ROOT / "data" / "raw" / "sampled_html"

log = get_logger(__name__)


# Fetch HTML from arXiv, caching it locally. The cache is a simple text file,
# empty if arXiv has no HTML for the paper.
def _fetch_html(arxiv_id: str, client: httpx.Client) -> str | None:
    """Download the LaTeXML page. Returns None when arXiv published none."""

    HTML_DIR.mkdir(parents=True, exist_ok=True)
    cached = HTML_DIR / f"{arxiv_id.replace('/', '_')}.html"

    if cached.exists():
        text = cached.read_text(encoding="utf-8", errors="ignore")
        return None if text == "" else text

    try:
        r = client.get(f"https://arxiv.org/html/{arxiv_id}", timeout=60.0)
    except httpx.RequestError as e:
        log.error("Error fetching arXiv %s: %s", arxiv_id, e)
        return None

    # arXiv answers with a "no HTML for this paper" stub, not a 404.
    missing = r.status_code != 200 or "ltx_page_main" not in r.text

    cached.write_text("" if missing else r.text, encoding="utf-8")
    return None if missing else r.text


# Loader class that fetches and parses HTML from arXiv, returning a Loaded object.
def load(arxiv_id: str, client: httpx.Client) -> Loaded:

    html = _fetch_html(arxiv_id, client)

    if html is None:
        log.warning("arXiv %s has no HTML published", arxiv_id)
        return Loaded(arxiv_id, [], [], note="no arXiv HTML published")

    reader = Reader()
    reader.feed(html)
    reader.passages.extend(reader.pending)

    for i, passage in enumerate(reader.passages):
        passage.order = i

    return Loaded(arxiv_id, reader.titles, reader.passages, images=reader.images)


if __name__ == "__main__":
    with httpx.Client() as client:
        loaded = load("1706.03762v7", client)
        print(f"\n=== {loaded.arxiv_id}  {len(loaded.sections)} sections  "
              f"{len(loaded.passages)} passages  {loaded.note}")
        for p in loaded.passages:
            print(f"  [{p.location or '-'}] ({p.section}) {p.text}")
