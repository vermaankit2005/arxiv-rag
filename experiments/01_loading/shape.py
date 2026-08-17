"""The one shape every candidate source must produce.

The sprint's rule is that HTML, LaTeX and PDF are graded by the same script, so
they must hand back the same thing:

    passage text  +  which section it belongs to  +  a location a reader clicks

`location` is deliberately a free-form string. HTML gives a URL fragment, PDF
would give a page number. The scorer only checks that it is non-empty and
resolves; it does not care about the format.
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


@dataclass
class Loaded:
    arxiv_id: str
    sections: list[str]               # headings in document order
    passages: list[Passage]
    note: str = ""
