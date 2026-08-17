"""Survey the PDFs our corpus will actually contain.

We are about to choose a PDF extractor. That choice should be made against the
population we will really ingest -- in-domain arXiv papers -- not against a
hand-picked set of famous papers, which are unusually well typeset.

What actually breaks extractors is not the topic. It is the PDF *producer*
(pdfTeX / XeTeX / LuaTeX / Word / Ghostscript), the column layout, the length,
and the amount of maths. So we measure those, and let the numbers pick the
benchmark set.

Output: survey.json -- one record per sampled paper. Nothing here is a decision.

Run:  uv run python experiments/01_loading/survey.py
"""

from __future__ import annotations

import contextlib
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import arxiv
import httpx
import pymupdf
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = ROOT / "data" / "raw" / "survey"
OUT = Path(__file__).parent / "survey.json"

SAMPLE_SIZE = 40
UA = "arxiv-rag/0.0.1 (research prototype; contact via arxiv-rag repository)"
SEED = 20260817

# In-domain topics, drawn from the corpus scope in docs/PRODUCT.md. Deliberately
# spread across the topic families rather than concentrated on RAG.
TOPICS = [
    "large language model",
    "retrieval-augmented generation",
    "chain-of-thought reasoning",
    "preference optimization",
    "language model agent",
    "parameter-efficient fine-tuning",
    "language model evaluation",
    "efficient llm inference",
]
# Temporal spread matters: LaTeX toolchains and templates drift year to year.
YEARS = [2023, 2024, 2025, 2026]
CATEGORIES = ["cs.CL", "cs.LG", "cs.AI"]

# Fonts that only appear when a document sets real mathematics.
MATH_FONTS = ("CMMI", "CMSY", "CMEX", "MSAM", "MSBM", "MathItalic", "STIXMath", "XITSMath")


@dataclass
class Record:
    arxiv_id: str
    title: str
    published: str
    primary_category: str
    producer: str
    creator: str
    pages: int
    page_width: float
    page_height: float
    columns: int
    math_font_ratio: float
    tables_found: int
    latex_source: bool
    error: str = ""


def sample_papers() -> list[arxiv.Result]:
    """Collect candidates across topics and years, then take a seeded sample."""
    client = arxiv.Client(page_size=50, delay_seconds=3.0, num_retries=3)
    pool: dict[str, arxiv.Result] = {}
    cats = " OR ".join(f"cat:{c}" for c in CATEGORIES)

    for topic in TOPICS:
        for year in YEARS:
            window = f"submittedDate:[{year}01010000 TO {year}12312359]"
            query = f'({cats}) AND abs:"{topic}" AND {window}'
            search = arxiv.Search(
                query=query, max_results=8, sort_by=arxiv.SortCriterion.SubmittedDate
            )
            try:
                for result in client.results(search):
                    pool[result.get_short_id()] = result
            except Exception as exc:  # a failed slice is not fatal
                print(f"  ! query failed ({topic} {year}): {exc}")

    print(f"pool: {len(pool)} candidates")
    rng = random.Random(SEED)
    chosen = rng.sample(sorted(pool), min(SAMPLE_SIZE, len(pool)))
    return [pool[cid] for cid in chosen]


def page_gutter(page: pymupdf.Page) -> int | None:
    """Width of the widest word-free vertical strip near the page centre."""
    width = int(page.rect.width)
    occupied = [False] * (width + 1)
    words = 0
    for x0, _y0, x1, _y1, text, *_ in page.get_text("words"):
        if not text.strip():
            continue
        words += 1
        for x in range(max(0, int(x0)), min(width, int(x1)) + 1):
            occupied[x] = True
    if words < 120:      # a figure page says nothing about the text layout
        return None
    best = run = 0
    for x in range(int(width * 0.38), int(width * 0.62)):
        run = 0 if occupied[x] else run + 1
        best = max(best, run)
    return best


def count_columns(doc: pymupdf.Document) -> int:
    """1 or 2 columns, from the share of pages showing a centre gutter.

    Earlier attempts -- clustering block left-edges, and comparing median block
    width -- disagreed with each other on 22 of 40 papers, so both were wrong.
    Full-width figures also sink a median-gutter measure, which is why this
    counts *pages that show a gutter* instead of averaging the gutter width.

    Validated against three pages inspected by eye: 2403.02310v3 (two column),
    2509.24832v2 (two column, narrow 7pt gutter), 2310.12821v5 (one column).
    """
    gaps = [g for i in range(1, min(9, doc.page_count)) if (g := page_gutter(doc[i])) is not None]
    if not gaps:
        return 0
    share = sum(1 for g in gaps if g >= 6) / len(gaps)
    return 2 if share >= 0.35 else 1


def math_ratio(doc: pymupdf.Document, pages: int) -> float:
    """Share of text spans set in a mathematics font."""
    total = math = 0
    for i in range(min(pages, 8)):
        for block in doc[i].get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    total += 1
                    if any(f in span["font"] for f in MATH_FONTS):
                        math += 1
    return round(math / total, 4) if total else 0.0


def count_tables(doc: pymupdf.Document, pages: int) -> int:
    found = 0
    for i in range(min(pages, 8)):
        # Table detection is best effort; a page it cannot parse is not a failure.
        with contextlib.suppress(Exception):
            found += len(doc[i].find_tables().tables)
    return found


def has_latex_source(arxiv_id: str, client: httpx.Client) -> bool:
    """arXiv serves e-print source; a gzip/tar body means real LaTeX exists."""
    try:
        r = client.get(
            f"https://arxiv.org/e-print/{arxiv_id}",
            headers={"Range": "bytes=0-3", "User-Agent": UA},
            timeout=20.0,
        )
        return r.status_code in (200, 206) and r.content[:2] == b"\x1f\x8b"
    except Exception:
        return False


def download(url: str, path: Path, client: httpx.Client) -> None:
    r = client.get(url, timeout=90.0, headers={"User-Agent": UA})
    r.raise_for_status()
    if not r.content.startswith(b"%PDF"):
        raise ValueError(f"not a PDF (got {r.headers.get('content-type')})")
    path.write_bytes(r.content)


def inspect(result: arxiv.Result, client: httpx.Client) -> Record:
    short_id = result.get_short_id()
    safe = short_id.replace("/", "_")
    path = PDF_DIR / f"{safe}.pdf"
    base = {
        "arxiv_id": short_id,
        "title": result.title.strip().replace("\n", " "),
        "published": result.published.date().isoformat(),
        "primary_category": result.primary_category,
    }
    try:
        if not path.exists():
            download(result.pdf_url, path, client)
            time.sleep(1.5)  # arXiv asks callers not to hammer the mirrors

        meta = PdfReader(str(path)).metadata or {}
        doc = pymupdf.open(str(path))
        pages = doc.page_count
        rec = Record(
            **base,
            producer=str(meta.get("/Producer", "") or "").strip()[:90],
            creator=str(meta.get("/Creator", "") or "").strip()[:90],
            pages=pages,
            page_width=round(doc[0].rect.width, 1),
            page_height=round(doc[0].rect.height, 1),
            columns=count_columns(doc),
            math_font_ratio=math_ratio(doc, pages),
            tables_found=count_tables(doc, pages),
            latex_source=has_latex_source(short_id, client),
        )
        doc.close()
        return rec
    except Exception as exc:  # a broken paper is itself a finding
        return Record(
            **base,
            producer="",
            creator="",
            pages=0,
            page_width=0.0,
            page_height=0.0,
            columns=0,
            math_font_ratio=0.0,
            tables_found=0,
            latex_source=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def normalise_producer(p: str) -> str:
    low = p.lower()
    for name in (
        "pdftex",
        "xetex",
        "luatex",
        "ghostscript",
        "dvips",
        "quartz",
        "microsoft",
        "word",
        "acrobat",
        "distiller",
        "skia",
        "cairo",
    ):
        if name in low:
            return name
    return low.split()[0] if low else "(none)"


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    papers = sample_papers()
    print(f"sampled {len(papers)} papers\n")

    records: list[Record] = []
    with httpx.Client(follow_redirects=True) as client:
        for n, paper in enumerate(papers, 1):
            rec = inspect(paper, client)
            records.append(rec)
            flag = "ERR" if rec.error else f"{rec.pages:>3}p {rec.columns}col"
            print(
                f"[{n:>2}/{len(papers)}] {rec.arxiv_id:<12} {flag}  "
                f"{normalise_producer(rec.producer)}"
            )

    OUT.write_text(json.dumps([asdict(r) for r in records], indent=2), encoding="utf-8")

    ok = [r for r in records if not r.error]
    print(f"\n--- {len(ok)}/{len(records)} inspected -----------------------")
    print("\nproducer:")
    for name, count in Counter(normalise_producer(r.producer) for r in ok).most_common():
        print(f"  {name:<14} {count:>3}  {count / len(ok):>5.0%}")
    print("\ncolumns:")
    for cols, count in sorted(Counter(r.columns for r in ok).items()):
        print(f"  {cols}              {count:>3}  {count / len(ok):>5.0%}")
    if ok:
        lengths = sorted(r.pages for r in ok)
        print(
            f"\npages: min {lengths[0]}  median {lengths[len(lengths) // 2]}  max {lengths[-1]}"
        )
        maths = sorted(r.math_font_ratio for r in ok)
        print(f"math font ratio: median {maths[len(maths) // 2]:.3f}  max {maths[-1]:.3f}")
        print(f"latex source available: {sum(r.latex_source for r in ok)}/{len(ok)}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
