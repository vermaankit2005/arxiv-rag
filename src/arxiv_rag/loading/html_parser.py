"""Pull the paper out of one arXiv HTML page.

HTML content contract
---------------------
Passages:
* ``ltx_para`` containers and bare ``p.ltx_p`` elements become prose. Lists
  nested inside them are therefore prose too.
* anchored note classes become notes.
* ``figcaption.ltx_caption`` becomes a figure or table caption.
* ``table.ltx_tabular`` becomes a serialized data table.

Metadata only:
* headings name sections but are not passages;
* ``img.ltx_graphics`` and ``object.ltx_graphics`` become linked images, while
  their pixels and embedded SVG text are not passages.

Skipped:
* ``head``, ``script``, ``style`` and ``svg`` subtrees;
* numbering scaffolding around headings and notes;
* notes without an anchor;
* bibliography entries, author metadata, navigation and every other element
  that is neither a passage source nor metadata listed above.

Formulae are the one special case: keep ``math.alttext`` once, then skip the
MathML subtree so the same formula is not duplicated.

The page is read once, top to bottom, the way a person reads it. Four small
piles of state do all the work:

* `sections` -- the headings we are currently inside. The innermost one names a
  passage's section; the whole pile is its path.
* `captures` -- the text we are part way through collecting. A pile, not a
  single buffer, because a footnote sits *inside* a paragraph and has to come
  out as its own passage. Spliced in where it sits, it would cut a sentence in
  half.
* `figures` -- the images collected inside the current figure, ready to attach
  to its caption passage.
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
from urllib.parse import urljoin

from arxiv_rag.loading.models import FigureImage, Passage

# HTML CONTENT CONTRACT. The parser below uses these names directly; do not add
# a content rule in handle_starttag without recording it here.

# Passage sources.
PROSE_CONTAINER_CLASS = "ltx_para"
PROSE_FALLBACK_TAG = "p"
PROSE_FALLBACK_CLASS = "ltx_p"
NOTE_CLASSES = {"ltx_note_content", "ltx_role_thanks", "ltx_role_footnote"}
CAPTION_TAG = "figcaption"
CAPTION_CLASS = "ltx_caption"
DATA_TABLE_TAG = "table"
DATA_TABLE_CLASS = "ltx_tabular"

# Metadata sources: retained on LoadedPaper/Passage objects, not as passage text.
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
TITLE_CLASS = "ltx_title"
SECTION_TAG = "section"
SECTION_CLASSES = {"ltx_bibliography", "ltx_appendix"}
FIGURE_TAG = "figure"
FIGURE_CLASS = "ltx_figure"
TABLE_FIGURE_CLASS = "ltx_table"
FIGURE_CLASSES = {FIGURE_CLASS, TABLE_FIGURE_CLASS}
IMAGE_TAGS = {"img", "object"}
IMAGE_CLASS = "ltx_graphics"

# Excluded content. Everything not named above is ignored automatically because
# passages are allowlisted rather than obtained by stripping every HTML tag.
FORMULA_TAG = "math"  # alttext is retained once before its subtree is skipped
SKIPPED_SUBTREE_TAGS = {"script", "style", "svg", "head"}
SKIPPED_LABEL_CLASSES = {"ltx_note_mark", "ltx_tag_note", "ltx_note_type"}
HEADING_NUMBER_CLASS = "ltx_tag"
VOID_TAGS = {"br", "img", "hr", "meta", "link", "input", "source", "col"}


@dataclass
class Section:
    """A <section> we have entered and not yet left."""

    depth: int
    title: str = ""  # filled in when its heading is read


@dataclass
class FigureContext:
    """A figure or table whose caption and media belong together."""

    depth: int
    kind: str
    images: list[FigureImage] = field(default_factory=list)


@dataclass
class Capture:
    """Text being collected right now, and where it will point."""

    kind: str
    depth: int
    node_id: str = ""
    parts: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    cell_depth: int | None = None
    cell_parts: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        if self.kind == "table":
            return "\n".join(" | ".join(row) for row in self.rows if any(row))
        return re.sub(r"\s+", " ", "".join(self.parts)).strip()


class ArxivHtmlParser(HTMLParser):
    """Walk the page once, emitting prose, captions, tables and image records.

    Read `passages`, `images` and `pending` when the feed is done.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.titles: list[str] = []  # every heading, in order
        self.passages: list[Passage] = []
        self.images: list[FigureImage] = []
        self.pending: list[Passage] = []  # notes, emitted after their host

        self.depth = 0
        self.skip_until: int | None = None
        self.sections: list[Section] = []
        self.figures: list[FigureContext] = []
        self.captures: list[Capture] = []
        self.open_ids: list[tuple[int, str]] = []
        self.in_tag_span = False  # <span class="ltx_tag">2 </span>

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

    def append_text(self, text: str) -> None:
        if not self.captures:
            return
        capture = self.captures[-1]
        if capture.kind == "table":
            if capture.cell_depth is not None:
                capture.cell_parts.append(text)
        else:
            capture.parts.append(text)

    def start_table_row(self) -> None:
        self.captures[-1].rows.append([])

    def start_table_cell(self) -> None:
        capture = self.captures[-1]
        if not capture.rows:
            capture.rows.append([])
        capture.cell_depth = self.depth
        capture.cell_parts.clear()

    def close_table_cell(self) -> None:
        if not self.captures:
            return
        capture = self.captures[-1]
        if capture.kind != "table" or self.depth != capture.cell_depth:
            return
        text = re.sub(r"\s+", " ", "".join(capture.cell_parts)).strip()
        capture.rows[-1].append(text)
        capture.cell_depth = None
        capture.cell_parts.clear()

    # -- reading ---------------------------------------------------------
    def handle_starttag(self, tag: str, attrs) -> None:
        attrs = {key: (value or "") for key, value in attrs}
        if tag not in VOID_TAGS:
            self.depth += 1
        if self.skip_until is not None:
            return

        if tag == FORMULA_TAG:
            # Keep one plain-text copy of the formula, then skip the markup.
            self.append_text(f" {attrs.get('alttext', '')} ")
            self.skip_until = self.depth
            return
        if tag in SKIPPED_SUBTREE_TAGS:
            self.skip_until = self.depth
            return

        if attrs.get("id") and tag not in VOID_TAGS:
            self.open_ids.append((self.depth, attrs["id"]))

        css = set((attrs.get("class") or "").split())
        if tag == FIGURE_TAG and (FIGURE_CLASSES & css):
            figure_kind = "figure" if FIGURE_CLASS in css else "table"
            self.figures.append(FigureContext(self.depth, figure_kind))

        if tag in IMAGE_TAGS and IMAGE_CLASS in css:
            # LaTeXML uses <img> for raster figures and <object> for SVGs.
            # Page logos and banner icons lack ltx_graphics. Their alt text adds
            # no information, so retain only the URL and real anchor.
            source = attrs.get("src") or attrs.get("data")
            node_id = attrs.get("id") or self.nearest_id
            if source and node_id:
                image = FigureImage(
                    url=urljoin("https://arxiv.org/html/", source),
                    location=f"#{node_id}",
                )
                self.images.append(image)
                if self.figures:
                    self.figures[-1].images.append(image)

        # The bibliography and appendices are sections too, so their headings
        # land on the pile like any other. Their *entries* never reach here:
        # LaTeXML writes them as <li class="ltx_bibitem">, which is neither
        # ltx_para nor p.ltx_p, so nothing below captures them.
        if tag == SECTION_TAG or SECTION_CLASSES & css:
            self.sections.append(Section(self.depth))

        if tag in HEADING_TAGS and TITLE_CLASS in css:
            self.start_capture("title")
            return

        node_id = attrs.get("id") or self.nearest_id

        # LaTeXML also uses <table> for displayed equations. Only ltx_tabular is
        # a data table; math in its cells is kept once through the alttext rule
        # above, while ltx_eqn_table remains on the existing equation path.
        if not self.capturing and tag == DATA_TABLE_TAG and DATA_TABLE_CLASS in css:
            self.start_capture("table", node_id)
            return

        if self.capturing == "table":
            if tag == "tr":
                self.start_table_row()
            elif tag in {"td", "th"}:
                self.start_table_cell()
            return

        if not self.capturing and tag == CAPTION_TAG and CAPTION_CLASS in css:
            caption_kind = (
                "table_caption"
                if self.figures and self.figures[-1].kind == "table"
                else "figure_caption"
            )
            self.start_capture(caption_kind, node_id)
            return

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
        if not self.capturing and (
            PROSE_CONTAINER_CLASS in css
            or (tag == PROSE_FALLBACK_TAG and PROSE_FALLBACK_CLASS in css)
        ):
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
        if SKIPPED_LABEL_CLASSES & css or (
            HEADING_NUMBER_CLASS in css and self.capturing == "title"
        ):
            self.in_tag_span = True

    def handle_data(self, data: str) -> None:
        if self.skip_until is None and self.captures and not self.in_tag_span:
            self.append_text(data)

    def handle_endtag(self, tag: str) -> None:
        if self.skip_until is not None:
            # Open our eyes only back at the depth where we closed them.
            if self.depth <= self.skip_until:
                self.skip_until = None
            self.depth -= 1
            return

        self.close_table_cell()
        self.close_capture()
        self.in_tag_span = False
        self.close_ids()
        self.close_figure()
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

    def close_figure(self) -> None:
        if self.figures and self.depth == self.figures[-1].depth:
            self.figures.pop()

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

        if capture.kind == "title":
            return
        passage_kind = "prose" if capture.kind == "para" else capture.kind
        linked_images = (
            list(self.figures[-1].images)
            if passage_kind == "figure_caption" and self.figures
            else []
        )
        passage = Passage(
            order=0,
            text=capture.text,
            section=self.section_title,
            section_path=[s.title for s in self.sections if s.title],
            location=f"#{capture.node_id}" if capture.node_id else "",
            kind=passage_kind,
            images=linked_images,
        )

        # A note is still inside its host paragraph, which has not been emitted
        # yet. Park it, and let the host bring it out.
        if capture.kind == "note" and self.captures:
            self.pending.append(passage)
            return

        self.passages.append(passage)
        self.passages.extend(self.pending)
        self.pending.clear()
