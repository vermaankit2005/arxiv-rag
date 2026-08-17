"""Build the golden set from arXiv LaTeX source.

The benchmark grades three PDF extractors. Its ground truth therefore must not
come from any of them -- otherwise we would be marking their homework with their
own answers.

arXiv publishes the LaTeX source for most papers. That source is upstream of
every PDF renderer, so it gives us three things for free and without opinion:

  * the real section headings, in order;
  * sentences that certainly exist in the paper ("probes");
  * the order those sentences must appear in.

A probe is only useful if a correct extractor would reproduce it verbatim, so we
keep only plain prose: no maths, no citations, no references, nothing that a
renderer is free to typeset differently.

Output: golden.json -- {arxiv_id: {sections: [...], probes: [...]}}

Run:  uv run python experiments/01_loading/golden.py
"""

from __future__ import annotations

import io
import json
import re
import tarfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
PAPERS = ROOT / "data" / "papers.json"
OUT = ROOT / "evals" / "golden" / "loading.json"
SRC_DIR = ROOT / "data" / "raw" / "source"

PROBES_PER_PAPER = 8
# Left where a formula or a URL was deleted, so a probe made of leftovers can be
# thrown away instead of being asked of an extractor that never saw the hole.
MATH_MARK = "\ue000"
# Inserted where a run-in heading ends one sentence and starts the next.
BREAK = "\ue001"
# Run-in headings: a bold label sitting on the same line as the prose after it.
RUNIN_RE = re.compile(
    r"\\(?:sub)?paragraph\*?\s*\{[^{}]*\}"                    # \paragraph{Ablations}
    r"|\\noindent\s*\{?\\(?:bf|it|em|sf|sc)\b[^{}]*\}"         # \noindent{\bf Takeaway-1:}
    r"|\\(?:textbf|textit|emph|textsc)\s*\{[^{}]*:\s*\}"       # \textbf{Takeaway-1:}
)
MIN_PROBE_CHARS = 60
MAX_PROBE_CHARS = 220

# Commands whose argument is a key, a colour or a page setting -- never words a
# reader sees. Each one that was missing leaked its argument into a probe:
# \autoref{fig:banner:stal} -> "fig:banner:stal shows that...",
# \color{red} -> "red Provided proper attribution...",
# \thispagestyle{citethis} -> "citethis An agent skill is...".
NON_PROSE_ARG = (
    "cite", "citep", "citet", "citeauthor", "citeyear", "citealp",
    "ref", "eqref", "cref", "Cref", "autoref", "nameref", "subref", "pageref",
    "label", "color", "pagecolor", "thispagestyle", "pagestyle", "pagenumbering",
    "setcounter", "addtocounter", "setlength", "includegraphics",
    "bibliographystyle", "documentclass", "usepackage", "hypersetup", "vspace",
    "hspace", "graphicspath",
)
NON_PROSE_ARG_RE = re.compile(
    r"\\(?:" + "|".join(NON_PROSE_ARG) + r")\*?\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}"
)

SECTION_RE = re.compile(r"\\(sub)?section\*?\s*\{", re.IGNORECASE)
COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
# Environments whose contents never survive into readable prose.
DROP_ENVS = (
    "equation", "align", "gather", "figure", "table", "tabular", "algorithm",
    "algorithmic", "lstlisting", "verbatim", "thebibliography", "tikzpicture",
    "comment",   # LaTeX-commented-out prose: in the file, not in the paper
)


@dataclass
class Probe:
    text: str
    order: int
    section: str


@dataclass
class Golden:
    arxiv_id: str
    sections: list[str]
    probes: list[Probe]
    note: str = ""


def fetch_source(arxiv_id: str, client: httpx.Client) -> bytes | None:
    """Download the e-print tarball. Returns None when arXiv has no source."""
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    cached = SRC_DIR / f"{arxiv_id.replace('/', '_')}.tar.gz"
    if cached.exists():
        return cached.read_bytes()
    r = client.get(f"https://arxiv.org/e-print/{arxiv_id}", timeout=60.0)
    if r.status_code != 200 or r.content[:2] != b"\x1f\x8b":
        return None
    cached.write_bytes(r.content)
    return r.content


def read_tex(blob: bytes) -> str:
    r"""Reconstruct the document in reading order, following \input and \include.

    An earlier version simply concatenated every .tex file with the
    \documentclass one first. That silently destroyed two papers: the main file
    ends with \bibliography{...}, and because the real content lived in
    \input-ed files placed *after* it, the bibliography cut in pick_probes threw
    that content away. BERT and Sarathi-Serve produced zero probes as a result.

    Papers also ship unused files -- older drafts, rebuttals, templates -- which
    a blind concatenation would happily mix into the text.
    """
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            files: dict[str, str] = {}
            for member in tar.getmembers():
                if not member.isfile() or not member.name.lower().endswith(".tex"):
                    continue
                handle = tar.extractfile(member)
                if handle is not None:
                    files[member.name] = handle.read().decode("utf-8", "ignore")
    except tarfile.ReadError:
        return strip_comments(blob.decode("utf-8", "ignore"))  # a bare .tex, gzipped

    if not files:
        return ""
    main = next((n for n, t in files.items() if "\\documentclass" in t), next(iter(files)))

    def resolve(name: str, seen: frozenset[str]) -> str:
        if name in seen:
            return ""
        text = strip_comments(files.get(name, ""))

        def swap(match: re.Match[str]) -> str:
            target = match.group(1).strip()
            for candidate in (target, f"{target}.tex", target.lstrip("./")):
                for key in files:
                    if key == candidate or key.endswith("/" + candidate):
                        return resolve(key, seen | {name})
            return ""

        return re.sub(r"\\(?:input|include)\s*\{([^}]*)\}", swap, text)

    return resolve(main, frozenset())


def strip_comments(tex: str) -> str:
    r"""Remove LaTeX % comments, which are not part of the paper.

    MX+ ships an acmart template carrying 6,615 comment markers. Left in, its
    boilerplate ("For submission and review of your manuscript...") became
    candidate probe sentences.
    """
    return COMMENT_RE.sub("", tex)


def strip_environments(tex: str) -> str:
    for env in DROP_ENVS:
        tex = re.sub(rf"\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}", " ",
                     tex, flags=re.DOTALL)
    return tex


def balanced_arg(text: str, open_at: int) -> tuple[str, int]:
    """Read a brace-balanced LaTeX argument starting at the opening brace."""
    depth, i = 0, open_at
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at + 1 : i], i + 1
        i += 1
    return "", len(text)


def clean_inline(s: str, mark_math: bool = False) -> str:
    r"""Turn a LaTeX fragment into the prose a renderer would produce.

    `mark_math` leaves a sentinel where a formula was deleted. Probes use it;
    section titles do not. Without it, "Then $A=U\Sigma V^T$ is a valid SVD of
    $M$ ($U$ and $V$ have orthonormal columns)" collapsed to "Then is a valid
    SVD of ( and have orthonormal columns)" -- a sentence no extractor could
    ever produce, marked wrong against every candidate.
    """
    hole = MATH_MARK if mark_math else " "
    # \begin{abstract} is punctuation, not a word. The generic "\cmd{x} -> x"
    # rule below turned it into the word "abstract", which was then glued to the
    # first sentence of six papers ("abstract Large pre-trained language...").
    s = re.sub(r"\\(?:begin|end)\s*\{[^}]*\}(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", s)
    s = re.sub(r"\\textcolor\s*\{[^}]*\}\s*\{([^{}]*)\}", r"\1", s)  # colour, not text
    s = re.sub(r"\\href\s*\{[^}]*\}\s*\{([^{}]*)\}", r"\1", s)   # keep the link text
    s = re.sub(r"\\url\s*\{[^}]*\}", hole, s)
    s = NON_PROSE_ARG_RE.sub("", s)
    s = re.sub(r"\\\[.*?\\\]|\\\(.*?\\\)", hole, s, flags=re.DOTALL)  # display maths
    s = re.sub(r"\$[^$]*\$", hole, s)             # inline maths
    s = re.sub(r"\\[a-zA-Z]+\s*\{([^{}]*)\}", r"\1", s)   # \textit{x} -> x
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)          # bare commands
    s = s.replace("``", '"').replace("''", '"').replace("~", " ")
    s = s.replace("\\%", "%").replace("\\&", "&").replace("\\_", "_")
    s = re.sub(r"[{}]", "", s)
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s).strip()


def expand_constant_macros(tex: str) -> str:
    r"""Substitute zero-argument \newcommand definitions.

    Papers name themselves with a macro -- MX+ defines \name and writes
    \section{\name{}: Enhancing the MX Formats}. Stripping the macro instead of
    expanding it left the section titled ": Enhancing the MX Formats", which
    would have marked every candidate wrong on a heading it got right.
    """
    # Collect every definition against the original text first. Substituting
    # inside the loop invalidates the offsets of later matches, which silently
    # turned "Sarathi-Serve" into "arathi-Serve".
    definitions: list[tuple[str, str]] = []
    for match in re.finditer(r"\\(?:new|renew)command\s*\{?\\(\w+)\}?\s*\{", tex):
        body, _ = balanced_arg(tex, match.end() - 1)
        if "#" in body:
            continue
        # The body is usually itself formatted -- MX+ has \name defined as
        # \textsc{MX+}\xspace -- so clean it down to the plain text a renderer
        # would show, and only accept a short result.
        plain = clean_inline(body).replace("\\", "")
        if plain and len(plain) <= 40:
            definitions.append((match.group(1), plain))

    # Longest name first, so \sysname is not clobbered by \sys.
    for name, plain in sorted(definitions, key=lambda d: -len(d[0])):
        tex = re.sub(rf"\\{name}(?:\{{\}})?(?![a-zA-Z])", lambda _m, p=plain: p, tex)
    return tex


def expand_section_macros(tex: str) -> str:
    r"""Rewrite paper-specific sectioning macros into plain \section{...}.

    Papers routinely wrap sectioning in their own command. MX+ (2510.14557)
    defines \putsec{label}{Title} -> \section{Title} and never writes \section
    itself, so a naive scan found zero sections in a 15-page paper.

    This expands only macros whose body contains a sectioning command -- not
    LaTeX macros in general, which is a rabbit hole we do not need to enter.
    """
    macros: dict[str, tuple[int, int, str]] = {}  # name -> (args, title index, level)
    pattern = r"\\(?:new|renew)command\s*\{?\\(\w+)\}?\s*\[(\d+)\]\s*\{"
    for match in re.finditer(pattern, tex):
        name, nargs = match.group(1), int(match.group(2))
        body, _ = balanced_arg(tex, match.end() - 1)
        # Keep the level the macro actually wraps. MX+ also defines a run-in
        # \paragraph macro; forcing that to \section would have invented a
        # section, and left "mx++Potential Use of Reserved Bits While MX+..."
        # -- a label, a heading and a sentence in one probe.
        hit = re.search(r"\\((?:sub)*(?:section|paragraph))\*?\s*\{#(\d)\}", body)
        if hit:
            macros[name] = (nargs, int(hit.group(2)), hit.group(1))

    for name, (nargs, title_index, level) in macros.items():
        out, pos = [], 0
        for call in re.finditer(rf"\\{name}\s*(?=\{{)", tex):
            if call.start() < pos:
                continue
            args, cursor = [], call.end()
            for _ in range(nargs):
                if cursor >= len(tex) or tex[cursor] != "{":
                    break
                arg, cursor = balanced_arg(tex, cursor)
                args.append(arg)
            if len(args) != nargs:
                continue
            out.append(tex[pos : call.start()])
            out.append(f"\\{level}{{{args[title_index - 1]}}}")
            pos = cursor
        out.append(tex[pos:])
        tex = "".join(out)
    return tex


def extract_sections(tex: str) -> list[str]:
    expanded = expand_section_macros(expand_constant_macros(tex))
    sections: list[str] = []
    for match in SECTION_RE.finditer(expanded):
        title = clean_inline(balanced_arg(expanded, match.end() - 1)[0])
        if title and 2 < len(title) < 90 and title not in sections:
            sections.append(title)
    return sections


NOT_BODY = (
    "title", "icmltitle", "author", "authors", "authornote", "authornotemark",
    "affil", "affiliation", "institution", "institute", "address", "email",
    "orcid", "country", "city", "department", "streetaddress", "postcode",
    "thanks", "keywords", "acks", "date", "footnote", "footnotetext",
)


def strip_definitions(tex: str) -> str:
    r"""Delete \newcommand and friends once their meaning has been expanded.

    A definition is not text. \renewcommand{\shortauthors}{Jungi Lee, Junyong
    Park, ...} otherwise had its braces stripped like any other command, and the
    five author names ran straight into the first sentence of the abstract.
    """
    out, pos = [], 0
    define = r"\\(?:new|renew|provide)command\s*\{?\\\w+\}?\s*(?:\[\d+\])*\s*(?=\{)"
    for match in re.finditer(define, tex):
        if match.start() < pos:
            continue
        _, cursor = balanced_arg(tex, match.end())
        out.append(tex[pos : match.start()])
        pos = cursor
    out.append(tex[pos:])
    return "".join(out)


def drop_non_body(tex: str) -> str:
    r"""Remove title matter and notes, which many classes put after \begin{document}.

    Brace-balanced, because these arguments contain other commands and a
    "{[^}]*}" pattern stops at the first inner brace.

    Two probes came from here: GestureGPT repeats \title{...} inside the
    document body, so the paper's own title became probe 0; and its
    \authornote{Corresponding author. Part of the work was conducted...} became
    a probe that lives in the author block, not the running text.
    """
    for name in NOT_BODY:
        out, pos = [], 0
        for call in re.finditer(rf"\\{name}\*?\s*(?=\{{)", tex):
            if call.start() < pos:
                continue
            _, cursor = balanced_arg(tex, call.end())
            out.append(tex[pos : call.start()])
            pos = cursor
        out.append(tex[pos:])
        tex = "".join(out)
    return tex


def is_good_probe(s: str) -> bool:
    """A probe must be plain prose a correct extractor reproduces verbatim."""
    if not MIN_PROBE_CHARS <= len(s) <= MAX_PROBE_CHARS:
        return False
    if not s.endswith("."):
        return False
    if any(ch in s for ch in "\\${}[]|<>^_"):
        return False
    # A hole left by a deleted formula or URL. The rendered paper still shows
    # the formula, so the sentence around it can never match, and marking a
    # candidate wrong for that would be measuring our own stripping.
    if MATH_MARK in s:
        return False
    if "http" in s or "www." in s:
        return False
    letters = sum(ch.isalpha() or ch.isspace() for ch in s)
    if letters / len(s) < 0.90:      # digit-heavy text is usually a table
        return False
    # Needs enough ordinary words to be locatable and unambiguous.
    return len(s.split()) >= 10


def pick_probes(tex: str, sections: list[str]) -> list[Probe]:
    """Take probes spread evenly through the document, not clustered at the top."""
    # Expand macros *before* the preamble is cut -- that is where they are
    # defined. Expanding afterwards left BERT's own name out of its abstract:
    # "a new language representation model called , which stands for...".
    whole = strip_definitions(expand_section_macros(expand_constant_macros(tex)))

    # The preamble is not the paper. \author{Aditya V. Nori} \affil{Microsoft
    # Research} \begin{document} became one "sentence" for 2311.03033, and an
    # acmart "ACM Reference Format" block became a bibliography entry probe.
    start = re.search(r"\\begin\{document\}", whole)
    body = drop_non_body(strip_environments(whole[start.end():] if start else whole))
    cut = re.search(r"\\begin\{thebibliography\}|\\bibliography\{|\\printbibliography", body)
    if cut:
        body = body[: cut.start()]
    # A run-in heading ends the sentence before the prose that follows it.
    # \paragraph{Retrieval Ablations} A key feature of RAG... was one probe;
    # no extractor can return a heading and a sentence as one string.
    body = RUNIN_RE.sub(BREAK, body)

    # Walk the body, remembering which section we are inside.
    current = sections[0] if sections else ""
    marks: list[tuple[str, str]] = []
    for chunk in re.split(r"(\\(?:sub)?section\*?\s*\{[^}]*\})", body):
        if chunk.startswith("\\section") or chunk.startswith("\\subsection"):
            inner = re.search(r"\{(.*)\}", chunk)
            current = clean_inline(inner.group(1)) if inner else current
            continue
        for raw in re.split(BREAK + r"|(?<=\.)\s+", clean_inline(chunk, mark_math=True)):
            sentence = raw.strip()   # a removed run-in heading leaves a space
            if is_good_probe(sentence):
                marks.append((sentence, current))

    # Drop sentences appearing more than once -- an ambiguous probe cannot
    # verify reading order.
    seen = [s for s, _ in marks]
    unique = [(s, sec) for s, sec in marks if seen.count(s) == 1]
    if not unique:
        return []

    step = max(1, len(unique) // PROBES_PER_PAPER)
    spread = unique[::step][:PROBES_PER_PAPER]
    return [Probe(text=s, order=i, section=sec) for i, (s, sec) in enumerate(spread)]


def main() -> None:
    if not PAPERS.exists():
        raise SystemExit(f"{PAPERS} not found -- pick the benchmark papers first.")
    paper_ids = [p["arxiv_id"] for p in json.loads(PAPERS.read_text(encoding="utf-8"))]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    results: list[Golden] = []
    with httpx.Client(follow_redirects=True) as client:
        for arxiv_id in paper_ids:
            blob = fetch_source(arxiv_id, client)
            if blob is None:
                results.append(Golden(arxiv_id, [], [], note="no LaTeX source published"))
                print(f"{arxiv_id:<12} -- no source (excluded from golden set)")
                continue
            tex = read_tex(blob)
            sections = extract_sections(tex)
            probes = pick_probes(tex, sections)
            results.append(Golden(arxiv_id, sections, probes))
            print(f"{arxiv_id:<12} {len(sections):>3} sections  {len(probes):>2} probes")

    OUT.write_text(
        json.dumps([asdict(g) for g in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    usable = [g for g in results if g.probes]
    print(f"\n{len(usable)}/{len(results)} papers usable as ground truth")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
