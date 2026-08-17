# Sprint 1 — Loading docs

This file holds **only the current sprint**. It is overwritten when the sprint
ends; the lasting result goes to `docs/decisions/`.

## The component

Paper in, citable text out. Nothing else.

## Why this first

Every claim the product makes must point at a real location in a real paper. If
the text is scrambled or the location is wrong, every later component inherits
garbage and no amount of retrieval or prompting repairs it.

---

## The question changed mid-sprint

It started as **"which PDF parser is best?"**

Step A answered a different question first. While surveying 40 in-domain papers
we checked whether arXiv publishes the authors' LaTeX source. It does — and it
also publishes **HTML generated from that LaTeX**.

- **39 of 40** papers have structured arXiv HTML (median 30 sections each).
- **12 of 12** benchmark papers have real LaTeX source in their e-print archive.

A PDF is what LaTeX *renders to*. Parsing it means reconstructing structure that
was thrown away at render time — the two-column reading-order problem exists
only because we chose to read the rendered artefact.

So the question is now: **which source do we build the corpus from?**

## Candidates

| # | Source | What it gives | What it costs |
|---|---|---|---|
| **A** | **arXiv HTML** (LaTeXML) | numbered section tree, real `<table>`, MathML, per-paragraph anchors, guaranteed reading order | ~3% of papers have none; maths appears twice unless parsed carefully |
| **B** | **LaTeX source** | the true original; explicit sections, tables, equations | custom macros, `\input` chains, per-paper quirks — parsing arbitrary LaTeX is its own hard problem |
| **C** | **PDF** (`pypdf`, `pymupdf4llm`, `docling`) | always available | reading order, page furniture, hyphenation, and column scrambling are all ours to fix |

Expected shape of the answer: **A primary, C as the fallback** for papers with no
HTML. B probably loses to A because LaTeXML has already solved the macro problem
for us. That is a hypothesis, not a decision — it gets measured.

## What we give up, and what we gain

Losing PDFs means losing **page numbers**. `PRODUCT.md` promises a citation
resolves to "a stored location such as page or section".

arXiv HTML gives something better:

```
https://arxiv.org/html/2005.11401v4#S2.p3
```

That is a **deep link to the exact paragraph**, on arXiv's own site. A reader
clicks a claim and lands on the sentence. A page number cannot do that.

If we adopt this, `PRODUCT.md`'s citation contract changes from *page* to
*section path + paragraph anchor*. That is an improvement to write down, not a
compromise to hide.

## How we decide

Ground truth stays independent of every candidate: it comes from the **LaTeX
source**, which is upstream of both the HTML and the PDF.

| Metric | Why |
|---|---|
| **Coverage** | share of the corpus this source exists for. A source that fails 30% of papers cannot be primary. |
| **Section fidelity** | do the extracted headings match `\section{...}` in the source, in order? |
| **Text recall** | do sentences taken from the LaTeX body appear in the output? |
| **Reading order** | do they appear in the right order? |
| **Cleanliness** | duplicated maths, citation noise, running headers, hyphen breaks per 1k words |
| **Location quality** | can a citation resolve to something a reader can click? |

### The rule, fixed before we run it

1. The primary source must reach **≥95% coverage** of the corpus and **≥0.95**
   on section fidelity and reading order.
2. Whatever the primary is, a **fallback** must exist for the remainder, and
   papers loaded by fallback must be recorded as such.
3. If two sources tie, the simpler one wins.

## Papers

12, chosen in Step A to match the surveyed population rather than by fame.
Recorded with their reasons in `data/papers.json`. Composition versus the
40-paper survey:

| | Population | Benchmark |
|---|---|---|
| producer (recent track) | 60% arXiv-GenPDF / 35% pdfTeX / 5% Ghostscript | 62 / 25 / 12 |
| two-column | 38% | 33% |
| pages | 3–53, median 16 | 3–38, median 18 |

Known gap: nothing above 38 pages. Add one if speed or memory ends up deciding.

## Deliverables

1. `data/papers.json` — the 12 papers and why each is in the set. ✅ done
2. `evals/golden/loading.json` — sections and probe sentences from LaTeX source. ✅ done
3. `experiments/01_loading/` — survey, golden, and benchmark scripts with results. ✅ done
4. `docs/decisions/01-loading.md` — the numbers and the decision. ✅ done
5. `src/arxiv_rag/loader.py` — the winner, behind one clear entry point.
6. `tests/test_loader.py` — the invariants below.

## Tests — the invariants

Fast tests run against one small committed fixture, never the real 12.

- every extracted passage carries a location that resolves to something real
- loading the same paper twice produces identical output
- the references section is identified, so it can be kept out of evidence
- a paper with no HTML falls back, and records that it fell back
- a corrupt or missing source raises a clear, typed error instead of junk
- **the benchmark runs as a slow test** — the golden probes must still be found
  and ordered, so a library upgrade that breaks us is noticed

## Done when

We can load any of the 12 papers, get text with correct sections and resolvable
locations, the decision is recorded with its numbers, and the invariants pass.

## Progress

- ✅ **Step A** — survey 40 papers, pick the 12. Found arXiv HTML; question changed.
- ✅ **Step B** — golden set built from LaTeX source: 12/12 papers, 309 sections,
  96 probes. Four bugs found and fixed while checking it.
- ✅ **Step C** — scored arXiv HTML against the key: coverage 12/12, probes
  96/96, reading order 1.00, section fidelity 0.99, junk 0.44 per 1k words.
  Twelve probes turned out not to be body prose; the rules that let them in
  were fixed and the key rebuilt. B and C were not run — see the decision.
- ⬜ **Step D** — implement the winner in `src/arxiv_rag/loader.py` and write
  the invariants. ✅ decision recorded in `docs/decisions/01-loading.md`.

## Open question for the end of the sprint

What does the extracted output tell us the chunker must respect? If the source
is HTML with a real section tree, the answer may look very different from
chunking a wall of PDF text. That answer picks Sprint 2.
