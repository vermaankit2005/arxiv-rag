"""Pull the paper out of one arXiv HTML page.

The page is read once, top to bottom, the way a person reads it. Three small
piles of state do all the work:

* `sections` -- the headings we are currently inside. The innermost one names a
  passage's section; the whole pile is its path.
* `captures` -- the text we are part way through collecting. A pile, not a
  single buffer, because a footnote sits *inside* a paragraph and has to come
  out as its own passage. Spliced in where it sits, it would cut a sentence in
  half.
* `skip_until` -- while inside <math> and friends, how deep we were when we
  closed our eyes.

Everything hangs off `depth`, the number of tags currently open. That is the
only dependable way to know when an element ends. Matching on the tag name
instead double-counts nested <span>s and silently swallows the rest of the
paper.
"""

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

from loader.shape import Passage

# Subtrees whose text is never prose a reader wants quoted.
SKIP_TAGS = {"script", "style", "math", "svg", "head"}
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
# Footnote-like prose: real content, but nested inside another paragraph.
NOTE_CLASSES = {"ltx_note_content", "ltx_role_thanks", "ltx_role_footnote"}
# Scaffolding LaTeXML writes around a footnote -- its number twice over, and a
# label saying what kind of note it is ("footnotemark: "). The author wrote none
# of it. See handle_starttag.
LABELS = {"ltx_note_mark", "ltx_tag_note", "ltx_note_type"}
VOID_TAGS = {"br", "img", "hr", "meta", "link", "input", "source", "col"}


@dataclass
class Section:
    """A <section> we have entered and not yet left."""

    depth: int
    title: str = ""       # filled in when its heading is read


@dataclass
class Capture:
    """Text being collected right now, and where it will point."""

    kind: str             # "title", "para" or "note"
    depth: int
    node_id: str = ""
    parts: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return re.sub(r"\s+", " ", "".join(self.parts)).strip()


class Reader(HTMLParser):
    """Walk the page once, emitting one passage per paragraph.

    Read `passages` and `pending` when the feed is done.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.titles: list[str] = []       # every heading, in order
        self.passages: list[Passage] = []
        self.pending: list[Passage] = []  # notes, emitted after their host

        self.depth = 0
        self.skip_until: int | None = None
        self.sections: list[Section] = []
        self.captures: list[Capture] = []
        self.open_ids: list[tuple[int, str]] = []
        self.in_tag_span = False          # <span class="ltx_tag">2 </span>

    # -- what we are in the middle of ------------------------------------
    @property
    def capturing(self) -> str:
        """Kind of text being collected, or "" if we are not collecting."""
        return self.captures[-1].kind if self.captures else ""

    @property
    def nearest_id(self) -> str:
        """The id of the closest open element that has one."""
        return self.open_ids[-1][1] if self.open_ids else ""

    @property
    def section_title(self) -> str:
        return self.sections[-1].title if self.sections else ""

    def start_capture(self, kind: str, node_id: str = "") -> None:
        self.captures.append(Capture(kind, self.depth, node_id))

    # -- reading ---------------------------------------------------------
    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = {k: (v or "") for k, v in attrs_list}
        if tag not in VOID_TAGS:
            self.depth += 1
        if self.skip_until is not None:
            return

        if tag == "math":
            # Keep one plain-text copy of the formula, then skip the markup.
            if self.captures:
                self.captures[-1].parts.append(f" {attrs.get('alttext', '')} ")
            self.skip_until = self.depth
            return
        if tag in SKIP_TAGS:
            self.skip_until = self.depth
            return

        if attrs.get("id") and tag not in VOID_TAGS:
            self.open_ids.append((self.depth, attrs["id"]))

        css = set((attrs.get("class") or "").split())

        # The bibliography and appendices are sections too, so their headings
        # land on the pile like any other. Their *entries* never reach here:
        # LaTeXML writes them as <li class="ltx_bibitem">, which is neither
        # ltx_para nor p.ltx_p, so nothing below captures them.
        if tag == "section" or "ltx_bibliography" in css or "ltx_appendix" in css:
            self.sections.append(Section(self.depth))

        if tag in HEADINGS and "ltx_title" in css:
            self.start_capture("title")
            return

        node_id = attrs.get("id") or self.nearest_id

        # Footnotes and author "thanks" notes are prose the paper really
        # contains -- the contribution note on page 1 of Attention Is All You
        # Need is one -- but they sit inside another paragraph, so they get
        # their own capture and come out after their host.
        if NOTE_CLASSES & css:
            self.start_capture("note", node_id)
            return

        # A <div class="ltx_para"> is the unit we want: it carries the #S2.p3
        # anchor and may hold several <p>. A bare <p class="ltx_p"> is the
        # fallback -- the abstract is written that way, with an <h6>Abstract</h6>
        # beside it, and capturing the wrapper instead loses the whole abstract.
        if not self.capturing and ("ltx_para" in css or (tag == "p" and "ltx_p" in css)):
            self.start_capture("para", node_id)
            return

        # Labels and numbering, not prose. A heading carries its number in
        # ltx_tag ("2 Background"). A footnote carries its number three times
        # over -- the superscript where it is referenced, the superscript
        # repeated at the start of the note, and again as ltx_tag_note -- plus
        # a "footnotemark: " label. A \footnotemark points at a footnote that
        # lives elsewhere and has no words of its own, so skipping the label
        # leaves it empty and it is never emitted.
        #
        # Only the footnote flavours of ltx_tag are skipped. ltx_tag_figure and
        # ltx_tag_table hold "Figure 1:" and "Table 2:", which are worth
        # keeping -- they say what the passage is.
        if LABELS & css or ("ltx_tag" in css and self.capturing == "title"):
            self.in_tag_span = True

    def handle_data(self, data: str) -> None:
        if self.skip_until is None and self.captures and not self.in_tag_span:
            self.captures[-1].parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.skip_until is not None:
            # Open our eyes only back at the depth where we closed them.
            if self.depth <= self.skip_until:
                self.skip_until = None
            self.depth -= 1
            return

        self.close_capture()
        self.in_tag_span = False
        self.close_ids()
        self.close_section()
        self.depth -= 1

    # -- leaving things --------------------------------------------------
    def close_capture(self) -> None:
        if not self.captures or self.depth != self.captures[-1].depth:
            return
        capture = self.captures.pop()
        if capture.kind == "title":
            self.name_section(capture.text)
        elif capture.text:
            self.emit(capture)

    def close_ids(self) -> None:
        while self.open_ids and self.depth <= self.open_ids[-1][0]:
            self.open_ids.pop()

    def close_section(self) -> None:
        if self.sections and self.depth == self.sections[-1].depth:
            self.sections.pop()

    # -- writing things down ---------------------------------------------
    def name_section(self, title: str) -> None:
        if not title:
            return
        if self.sections:
            self.sections[-1].title = title
        if title not in self.titles:
            self.titles.append(title)

    def emit(self, capture: Capture) -> None:
        # A note with no id cannot be cited. LaTeXML gives author "thanks"
        # notes no anchor, so there is no link to hand a reader, and this
        # project's promise is that everything it quotes can be clicked.
        #
        # Deliberately narrow: only *notes* are dropped. Across the 13 cached
        # papers all 7 anchorless passages are these notes and not one is body
        # prose. If a real paragraph ever loses its anchor, the "every passage
        # has a location" test should fail loudly rather than the text quietly
        # disappearing.
        if capture.kind == "note" and not capture.node_id:
            return

        passage = Passage(
            order=0,
            text=capture.text,
            section=self.section_title,
            section_path=[s.title for s in self.sections if s.title],
            location=f"#{capture.node_id}" if capture.node_id else "",
        )

        # A note is still inside its host paragraph, which has not been emitted
        # yet. Park it, and let the host bring it out.
        if capture.kind == "note" and self.captures:
            self.pending.append(passage)
            return

        self.passages.append(passage)
        self.passages.extend(self.pending)
        self.pending.clear()
