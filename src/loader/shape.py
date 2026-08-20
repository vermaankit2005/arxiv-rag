"""What a loaded paper looks like.

Sprint 1's rule was that HTML, LaTeX and PDF all get graded by the same script,
so all three have to hand back the same thing:

    passage text  +  which section it belongs to  +  a location a reader clicks

arXiv HTML won and is the only source we read today, but the promise stays: if a
fallback is ever added for the ~3% of papers with no HTML, it fills in this same
pair of objects and nothing downstream has to know which source it came from.

`location` is deliberately a plain string. HTML gives a URL fragment like
"#S4.SS1.p1"; a PDF would give a page number. Nothing here cares about the
format, only that it is non-empty and resolves to something real.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Passage:
    order: int
    text: str
    section: str                      # innermost heading this sits under
    section_path: list[str] = field(default_factory=list)
    location: str = ""
    kind: str = "prose"
    images: list[FigureImage] = field(default_factory=list)


@dataclass
class FigureImage:
    url: str
    location: str


@dataclass
class Loaded:
    arxiv_id: str
    sections: list[str]               # headings in document order
    passages: list[Passage]
    note: str = ""
    images: list[FigureImage] = field(default_factory=list)
