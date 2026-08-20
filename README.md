# arxiv-rag

An evidence-grounded reading assistant for language-centric GenAI research
papers. Every claim it makes points at the passage it came from.

**Status: sprint 1 — document loading.** Nothing works end to end yet.

## What exists

`src/loader/` reads a paper from arXiv's HTML and returns its passages, each
carrying the section it sits in and a paragraph anchor. A citation resolves to a
clickable link like `https://arxiv.org/html/2005.11401v4#S4.SS1.p1` — the exact
paragraph, not a page number.

The source was chosen by measurement, not preference: 40 papers surveyed, 39 of
them have arXiv HTML, and the loader was scored against an answer key built from
the LaTeX source — deliberately a third source, so it favours neither the HTML
nor the PDF.

## How this is built

Bottom-up. One component at a time, each decided with data and tested before the
next begins. No component enters without a demonstrated need: before any
database, queue, cache, vector store or framework, something in the running
program has to be broken, slow, or wrong.

Working notes — the product definition, the runtime map, the seven principles,
the sprint records and the decision log — are kept locally and are not published
with the code.
