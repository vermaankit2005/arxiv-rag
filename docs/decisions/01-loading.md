# 01 — Loading: which source do we read papers from?

**Decided:** 2026-08-17

## The question

Every claim this product makes must point at a real location in a real paper. If
the text arrives scrambled, or the location is wrong, every later component
inherits garbage and no amount of retrieval or prompting repairs it.

The sprint opened as *"which PDF parser is best?"*. Surveying 40 in-domain
papers answered a different question first: **39 of 40 have structured HTML
published by arXiv itself**, generated from the authors' LaTeX. A PDF is what
LaTeX renders *to*. Reading it means rebuilding structure that was thrown away
at render time — the two-column reading-order problem exists only because we
chose to read the rendered artefact.

So the real question became: **which source do we build the corpus from?**

## Candidates

| # | Source | What it gives | What it costs |
|---|---|---|---|
| **A** | **arXiv HTML** (LaTeXML), read with Python's built-in `html.parser` | section tree, per-paragraph `id` anchors, guaranteed reading order | ~3% of papers have none; every formula is written twice unless parsed carefully |
| **B** | **LaTeX source** (`arxiv.org/e-print`) | the true original | custom macros, `\input` chains, per-paper quirks — parsing arbitrary LaTeX is its own hard problem |
| **C** | **PDF** (`pypdf` 6.x, `pymupdf4llm` 0.0.30, `docling` 2.x) | always available | reading order, page furniture, hyphenation and column scrambling are all ours to fix |

## How we measured

Ground truth is independent of every candidate: it is built from the **LaTeX
source**, which is upstream of both the HTML and the PDF, so it favours neither.

- `experiments/01_loading/survey.py` — surveyed 40 in-domain papers and chose
  the 12 benchmark papers to match that population on producer, layout and
  length, not on fame. Recorded in `data/papers.json` with a reason each.
- `experiments/01_loading/golden.py` — builds the answer key:
  **12 papers, 309 section titles, 96 probe sentences** (8 per paper), in
  `evals/golden/loading.json`.
- `experiments/01_loading/load_html.py` — candidate A.
- `experiments/01_loading/score.py` — grades any candidate. All candidates
  return the same shape (`shape.py`), so one scorer covers all three.

Re-run with `uv run python experiments/01_loading/score.py html`.

**The matcher is checked before any score is printed.** Probe text and rendered
text never match character-for-character, so both sides are reduced to lowercase
word tokens and a probe counts as found when ≥90% of its tokens line up *in
order* inside a window 1.6× its length. `verify()` requires a real probe to be
found, and requires both a shuffled probe and a probe from another paper to be
rejected. If any of the three fails, the script exits without printing scores.

### The rule, fixed before any result was seen

1. The primary source must reach **≥95% coverage** and **≥0.95** on section
   fidelity and reading order.
2. A **fallback** must exist for the remainder, and papers loaded by fallback
   must be recorded as such.
3. If two sources tie, the simpler one wins.

## Numbers

Candidate A — arXiv HTML, 12 benchmark papers:

| paper | found | order | sections | junk/1k | passages |
|---|---|---|---|---|---|
| 1706.03762v7 | 1.00 | 1.00 | 1.00 | 2.26 | 83 |
| 1810.04805v2 | 1.00 | 1.00 | 1.00 | 0.00 | 127 |
| 2305.18290v3 | 1.00 | 1.00 | 0.97 | 1.10 | 110 |
| 2005.11401v4 | 1.00 | 1.00 | 1.00 | 0.17 | 60 |
| 2510.14557v1 | 1.00 | 1.00 | 1.00 | 0.00 | 90 |
| 2509.24832v2 | 1.00 | 1.00 | 1.00 | 0.00 | 79 |
| 2608.14492v1 | 1.00 | 1.00 | 1.00 | 0.81 | 49 |
| 2608.10906v1 | 1.00 | 1.00 | 1.00 | 0.57 | 31 |
| 2601.02404v1 | 1.00 | 1.00 | 0.98 | 0.17 | 78 |
| 2403.02310v3 | 1.00 | 1.00 | 0.97 | 0.13 | 97 |
| 2310.12821v5 | 1.00 | 1.00 | 1.00 | 0.08 | 147 |
| 2311.03033v1 | 1.00 | 1.00 | 1.00 | 0.00 | 70 |
| **mean** | **1.00** | **1.00** | **0.99** | **0.44** | 1021 |

Against the rule:

| Rule | Required | Measured |
|---|---|---|
| coverage | ≥95% | **12/12 = 100%** |
| section fidelity | ≥0.95 | **0.99** (306 of 309 titles) |
| reading order | ≥0.95 | **1.00** |

Also measured:

- **Text recall 96/96.** Every probe sentence was found.
- **Locations: 1,015 of 1,021 passages** carry a `#S2.p3`-style anchor. The 6
  without one are front matter that sits outside any numbered section.
- **Junk 0.44 per 1,000 words** across 78,052 words: 9 hyphen-split words,
  4 duplicated strings, 14 repeated short passages.
- **Maths is not duplicated.** LaTeXML writes each formula twice — MathML for
  the browser, `<annotation encoding="application/x-tex">` for copy-paste. The
  loader never descends into `<math>`; it takes the `alttext` attribute once.
  This was the sprint's open question and it is now closed.
- The 3 missed headings are a run-on table caption, an appendix title split
  across two lines, and one heading LaTeXML numbers differently.

## Decision

**Read papers from arXiv HTML.** It cleared every threshold with margin — 100%
coverage against a required 95%, 1.00 reading order and 0.99 section fidelity
against a required 0.95, and every probe sentence found — and it is the only
candidate that yields a per-paragraph anchor a reader can click.

The loader will live in `src/arxiv_rag/loader.py`.

### What we did not measure, and why

**Candidates B and C were never run.** Measuring them could only have changed
the decision if A had failed the rule; A cleared it on every metric, and rule 3
says the simpler source wins a tie. Running B and C would have cost a day and
produced numbers nobody would act on.

This is an honest gap, and it has one consequence: **we do not know how good the
fallback is** for the ~3% of papers with no HTML. That is deferred, not solved —
see below.

### What this changes elsewhere

Losing PDFs means losing page numbers. `docs/PRODUCT.md`'s citation contract
moves from *page or section* to *section path plus paragraph anchor*, which
resolves to a link like:

```
https://arxiv.org/html/2005.11401v4#S2.p3
```

A reader clicks a claim and lands on the sentence. A page number cannot do that.

## What would change our mind

- **Coverage drops on the real corpus.** The benchmark is 12 papers and all 12
  had HTML; the 40-paper survey found 1 in 40 without. If the true rate over
  8,000 papers is materially worse than ~3%, HTML alone stops being enough.
- **Older papers.** arXiv began generating HTML in late 2023. It backfilled far
  enough to cover both 2017 and 2018 papers in this set, but a foundational
  paper with no HTML would need the fallback.
- **The parser breaks.** We used Python's built-in `html.parser`, on the
  grounds that LaTeXML output is machine-generated and regular. A paper that
  comes out scrambled or loses its anchors is the demonstrated need to switch to
  `lxml` or BeautifulSoup — a parser, not a framework.
- **Anchors stop being stable.** If arXiv changes its `id` scheme, stored
  citations rot. Worth re-checking whenever a version is re-fetched.

## Open, and deliberately so

**The fallback for papers with no HTML.** Rule 2 requires one, and it does not
exist yet. It is deferred until a paper we actually want is missing HTML —
at which point candidates B and C get measured on exactly that set, which is the
only set where their numbers matter.
