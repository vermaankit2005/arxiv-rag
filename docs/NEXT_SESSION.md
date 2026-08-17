# Prompt for the next session

Start a fresh Claude Code session in `D:\Ankit\Python\arxiv-rag` and say:

> Read `docs/NEXT_SESSION.md` and continue from there.

---

## How I want you to work

- **Small steps.** Do one thing, show me the result in a few lines, then stop
  and ask before the next thing. I lose the thread when answers get long.
- **Simple language.** Numbered steps. No jargon unless you explain it.
- Decide with data, not taste. If you state a number, show where it came from.
- If a measurement contradicts an earlier assumption, say so plainly and change
  the plan. That has already happened twice and both times it was right.
- No new tool, library, or service without something currently broken that it
  fixes. Hold me to this too.

## Read these first

1. `CLAUDE.md` — the rules
2. `docs/PROCESS.md` — how we build one component at a time
3. `docs/SPRINT.md` — the current sprint, its metrics, and its pass rule

Do **not** read the old repo at `D:\Ankit\Projects\arxiv-rag`. Its useful ideas
are already carried into `docs/`.

---

## The project in three lines

An evidence-grounded reading assistant for language-centric GenAI papers. Every
claim it makes must point at the passage it came from. Corpus is capped at
8,000–10,000 arXiv papers, one domain, with the full paper list published.

## Sprint 1 — "loading docs"

**Goal: source in, citable text out.** Nothing else. No AI, no search, no answers.

"Loading" means producing this shape, whatever the source:

```
passage text  +  which section it belongs to  +  a location a reader can click
```

All candidates must produce that same shape, so one scoring script grades them all.

### The three candidates

| Source | Passages from | Sections from | Location |
|---|---|---|---|
| **arXiv HTML** | `<p>` tags | `<h2>`/`<h3>` tags | `#S2.p3` anchor — deep link |
| **LaTeX source** | paragraphs | `\section{...}` | section path only |
| **PDF** (`pypdf`/`pymupdf4llm`/`docling`) | text blocks | guessed from font size | page number |

Expected answer: **HTML primary, PDF fallback or dropped entirely.** That is a
hypothesis, not a decision — it must be measured. See `docs/SPRINT.md` for the
metrics and the pass rule, which were fixed before any result was seen. Do not
loosen them afterwards.

---

## What is already done

### Step A — survey, done

`experiments/01_loading/survey.py` sampled 40 in-domain arXiv papers and
measured what our corpus actually looks like. Results in `survey.json`.

- Producers: **60% arXiv-GenPDF, 35% author pdfTeX, 5% Ghostscript.** arXiv now
  compiles most recent PDFs itself. Famous old papers are 100% pdfTeX, so
  benchmarking on famous papers alone would have tested the wrong toolchain.
- **38% two-column**, 62% single.
- Pages: 3 to 53, median 16.
- **39 of 40 papers have structured arXiv HTML** (median 30 sections each).
  This discovery changed the sprint from "which PDF parser" to "which source".

The column detector took four attempts; the first three disagreed with each
other. It is now validated against three pages inspected by eye, and the failed
attempts are documented in the script's docstring. Do not "simplify" it back.

The 12 benchmark papers are in `data/papers.json`, each with the reason it was
chosen. They match the surveyed population on producer, layout, and length.

### Step B — the answer key, done

`experiments/01_loading/golden.py` builds `data/golden/loading.json` from each
paper's **LaTeX source**, which is upstream of both the HTML and the PDF, so it
favours no candidate.

Result: **12/12 papers, 309 section titles, 96 probe sentences** (8 per paper).

Per paper it holds:

```json
{
  "arxiv_id": "2005.11401v4",
  "sections": ["Introduction", "Methods", "Retriever: DPR", ...],
  "probes": [
    {"order": 0, "section": "Introduction", "text": "However, their ability to access..."},
    {"order": 1, "section": "Introduction", "text": "Despite these being extractive..."}
  ]
}
```

Four real bugs were found and fixed while checking it — read the docstrings in
`golden.py` before changing that file:

1. `.tex` files were concatenated in the wrong order, so a bibliography cut
   deleted most of two papers (BERT and Sarathi produced zero probes).
2. LaTeX `%` comments were never stripped; MX+ leaked 6,615 template comments.
3. MX+ defines its own `\putsec{}` instead of `\section{}` — zero sections found
   until macro expansion was added.
4. The macro-expansion fix mutated text while holding offsets into the old text,
   turning "Sarathi-Serve" into "arathi-Serve".

Lesson worth keeping: running a script and *checking its output* are two
different steps.

---

## Next: Step C

Score each candidate against the answer key, on the same 12 papers.

### The four checks

1. **Sentences found** — is each of the 8 probe texts present in the output?
   Score = found / 8.
2. **Right order** — record where each probe was found. Positions must increase
   with `order`. A probe appearing before its predecessor means scrambled text.
   Score = correctly-ordered adjacent pairs / total pairs.
3. **Sections found** — how many of the paper's real section titles appear as
   headings in the output.
4. **Junk** — repeated running headers/footers, hyphen-split words
   (`intro-duction`), duplicated maths, per 1,000 words. Lower is better.

### Do this first, before trusting any score

Probe text and extracted text will never match character-for-character —
spacing, quote marks and dashes are typeset differently. So normalise both
sides (collapse whitespace, unify quotes and dashes, casefold) before matching.

**Verify the matcher on a case you know should pass** — take a probe, find it by
hand in the HTML, and confirm the matcher says "found". If matching is too
strict everything scores zero; too loose and everything scores one. Do not
report numbers until this is checked.

### Order of work

1. Write the scorer. Verify the matcher as above.
2. Run **arXiv HTML only**. Show me one paper's score first, then all 12.
3. Stop. Show me the numbers before running anything else.
4. Then LaTeX. Then PDF only if the pass rule requires it.
5. Write `docs/decisions/01-loading.md` — the question, candidates, method,
   the numbers, the decision, and what would change our mind. Template is in
   `docs/decisions/README.md`.

### A known open question for the decision

arXiv HTML renders maths twice (once visually, once as LaTeX), so a naive text
extraction duplicates it. Confirm whether a proper HTML parse avoids this. It
affects the junk score and is not yet resolved.

---

## Environment

- Windows, `uv`. `uv sync --group fetch --group extract` installs what is needed.
- `docling` sits in its own group (`extract-docling`, ~1 GB of models) and is
  **not installed**. Only install it if Step C actually reaches PDFs.
- Prefix Python runs with `PYTHONIOENCODING=utf-8` — paper text contains Greek
  letters that crash the default Windows console encoding.
- Heredocs in the Bash tool mangle backslashes on this machine. Write Python
  files with the Write tool instead of `cat <<'EOF'`.
- Downloaded PDFs and LaTeX archives live in `data/raw/` and are gitignored.
  They are cached, so re-running scripts does not re-download.

## Repo state

`git init` has been run but **nothing is committed yet.** Worth committing the
docs, `data/papers.json`, `data/golden/loading.json`, and `experiments/` early
so the work is safe.
