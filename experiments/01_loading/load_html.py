"""Candidate A: load a paper from arXiv's LaTeXML-generated HTML.

arXiv publishes https://arxiv.org/html/<id> for most papers since late 2023.
It is generated from the authors' LaTeX by LaTeXML, so the structure we need is
already in the markup instead of having to be guessed from a rendered page:

    <section id="S2" class="ltx_section">
      <h2 class="ltx_title ltx_title_section"><span class="ltx_tag">2 </span>Background</h2>
      <div class="ltx_para" id="S2.p1"><p class="ltx_p">...</p></div>

That gives a section tree, a paragraph id, and guaranteed reading order.

Why the standard library parser and not lxml/BeautifulSoup: nothing is broken
yet. LaTeXML emits regular, machine-written XHTML -- not the tag soup those
libraries exist to survive. If this parser turns out to mangle a real paper, that
is the demonstrated need, and we add one then.

The maths question from the sprint notes, answered:
LaTeXML writes every formula twice inside <math> -- once as MathML for the
browser and once as <annotation encoding="application/x-tex"> for copy-paste.
Naive tag-stripping therefore emits the same equation two or three times. We
never descend into <math> at all; we take its alttext attribute, which is the
original LaTeX, exactly once.

Run:  uv run python experiments/01_loading/load_html.py 2005.11401v4
"""

from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shape import Loaded, Passage

ROOT = Path(__file__).resolve().parents[2]
HTML_DIR = ROOT / "data" / "raw" / "html"

# Subtrees whose text is never prose a reader wants quoted.
SKIP_TAGS = {"script", "style", "math", "svg", "head"}
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
# Footnote-like prose: real content, but nested inside another paragraph.
NOTE_CLASSES = {"ltx_note_content", "ltx_role_thanks", "ltx_role_footnote"}
VOID_TAGS = {"br", "img", "hr", "meta", "link", "input", "source", "col"}


def fetch_html(arxiv_id: str, client: httpx.Client) -> str | None:
    """Download the LaTeXML page. Returns None when arXiv published none."""
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    cached = HTML_DIR / f"{arxiv_id.replace('/', '_')}.html"
    if cached.exists():
        text = cached.read_text(encoding="utf-8", errors="ignore")
        return None if text == "" else text
    r = client.get(f"https://arxiv.org/html/{arxiv_id}", timeout=60.0)
    # arXiv answers with a "no HTML for this paper" stub, not a 404.
    missing = r.status_code != 200 or "ltx_page_main" not in r.text
    cached.write_text("" if missing else r.text, encoding="utf-8")
    return None if missing else r.text


class Reader(HTMLParser):
    r"""Walk the page once, emitting one passage per <div class="ltx_para">.

    Two pieces of state do the work:

    * `skip_depth` -- while inside <math> or friends, count nested opens so we
      resume at the right place. Anything else double-counts nested <span>s and
      silently swallows the rest of the paper.
    * `sections` -- a stack of open <section> elements. The innermost one names
      the passage's section; the whole stack is its path.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.titles: list[str] = []
        self.passages: list[Passage] = []

        self.section_stack: list[str] = []
        self.open_sections: list[int] = []      # depth at which each was opened
        self.depth = 0
        self.skip_depth: int | None = None

        # A capture is (kind, depth, buffer, id). It is a stack because a
        # footnote lives *inside* a paragraph and must come out as its own
        # passage -- spliced inline it would cut a sentence in half.
        self.captures: list[tuple[str, int, list[str], str]] = []
        self.pending: list[Passage] = []        # notes, emitted after their host
        self.id_stack: list[tuple[int, str]] = []   # (depth, id) of open elements
        self.in_tag_span = False                # <span class="ltx_tag">2 </span>
        self.in_bibliography = False

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def classes(attrs: dict[str, str]) -> set[str]:
        return set((attrs.get("class") or "").split())

    @property
    def capturing(self) -> str | None:
        return self.captures[-1][0] if self.captures else None

    def start_capture(self, kind: str, node_id: str = "") -> None:
        self.captures.append((kind, self.depth, [], node_id))

    # -- parser callbacks ------------------------------------------------
    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = {k: (v or "") for k, v in attrs_list}
        if tag not in VOID_TAGS:
            self.depth += 1
        if self.skip_depth is not None:
            return

        if tag == "math":
            # One copy of the formula, from the attribute, then skip the subtree.
            if self.captures:
                self.captures[-1][2].append(f" {attrs.get('alttext', '')} ")
            self.skip_depth = self.depth
            return
        if tag in SKIP_TAGS:
            self.skip_depth = self.depth
            return

        if attrs.get("id") and tag not in VOID_TAGS:
            self.id_stack.append((self.depth, attrs["id"]))

        css = self.classes(attrs)
        if tag == "section" or "ltx_bibliography" in css or "ltx_appendix" in css:
            # The bibliography is a section too; mark it so its paragraphs are
            # labelled rather than mistaken for prose.
            if "ltx_bibliography" in css:
                self.in_bibliography = True
            self.open_sections.append(self.depth)
            self.section_stack.append("")

        if tag in HEADINGS and "ltx_title" in css:
            self.start_capture("title")
            return

        node_id = attrs.get("id") or (self.id_stack[-1][1] if self.id_stack else "")

        # Footnotes and author "thanks" notes are prose the paper really
        # contains -- the contribution note on page 1 of Attention Is All You
        # Need is one -- but they sit inside another paragraph, so they get
        # their own capture and are emitted after their host.
        if NOTE_CLASSES & css:
            self.start_capture("note", node_id)
            return

        # A <div class="ltx_para"> is the unit we want: it carries the #S2.p3
        # anchor and may hold several <p>. A bare <p class="ltx_p"> is the
        # fallback -- the abstract is written that way, with an <h6>Abstract</h6>
        # beside it, and capturing the wrapper instead loses the whole abstract.
        if self.capturing is None and ("ltx_para" in css or (tag == "p" and "ltx_p" in css)):
            self.start_capture("para", node_id)
            return

        if self.capturing == "title" and "ltx_tag" in css:
            self.in_tag_span = True

    def handle_endtag(self, tag: str) -> None:
        if self.skip_depth is not None:
            if self.depth <= self.skip_depth:
                self.skip_depth = None
            self.depth -= 1
            return

        if self.captures and self.depth == self.captures[-1][1]:
            kind, _, buffer, node_id = self.captures.pop()
            text = re.sub(r"\s+", " ", "".join(buffer)).strip()
            if kind == "title":
                self.finish_title(text)
            elif text:
                self.finish_para(text, node_id, note=kind == "note")

        if self.in_tag_span:
            self.in_tag_span = False

        while self.id_stack and self.depth <= self.id_stack[-1][0]:
            self.id_stack.pop()

        if self.open_sections and self.depth == self.open_sections[-1]:
            self.open_sections.pop()
            self.section_stack.pop()
            if not any(s == "References" for s in self.section_stack):
                self.in_bibliography = self.in_bibliography and bool(self.section_stack)

        self.depth -= 1

    def handle_data(self, data: str) -> None:
        if self.skip_depth is None and self.captures and not self.in_tag_span:
            self.captures[-1][2].append(data)

    # -- emitting --------------------------------------------------------
    def finish_title(self, text: str) -> None:
        if not text:
            return
        if self.section_stack:
            self.section_stack[-1] = text
        if text not in self.titles:
            self.titles.append(text)

    def finish_para(self, text: str, node_id: str, note: bool = False) -> None:
        passage = Passage(
            order=0,
            text=text,
            section=self.section_stack[-1] if self.section_stack else "",
            section_path=[s for s in self.section_stack if s],
            location=f"#{node_id}" if node_id else "",
        )
        if note and self.captures:
            self.pending.append(passage)        # comes out after its host
            return
        self.passages.append(passage)
        while self.pending:
            self.passages.append(self.pending.pop(0))


def load(arxiv_id: str, client: httpx.Client) -> Loaded:
    html = fetch_html(arxiv_id, client)
    if html is None:
        return Loaded(arxiv_id, [], [], note="no arXiv HTML published")
    reader = Reader()
    reader.feed(html)
    reader.passages.extend(reader.pending)
    for i, passage in enumerate(reader.passages):
        passage.order = i
    return Loaded(arxiv_id, reader.titles, reader.passages)


def main() -> None:
    ids = sys.argv[1:] or ["2005.11401v4"]
    with httpx.Client(follow_redirects=True) as client:
        for arxiv_id in ids:
            doc = load(arxiv_id, client)
            print(f"\n=== {arxiv_id}  {len(doc.sections)} sections  "
                  f"{len(doc.passages)} passages  {doc.note}")
            for p in doc.passages[:4]:
                print(f"  [{p.location or '-'}] ({p.section}) {p.text[:120]}")


if __name__ == "__main__":
    main()
