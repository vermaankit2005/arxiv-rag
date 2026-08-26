# arxiv-rag

An evidence-grounded reading assistant for language-centric GenAI research
papers. Every claim it makes points at the passage it came from.

**Status: sprint 1 complete — paper loading finished.** Nothing works end to end yet;
chunking is the next question.

## What exists

`src/arxiv_rag/loading/` reads arXiv HTML and returns citable prose, figure/table captions
and serialized data tables with their section path and anchor. Figure captions
carry their image URL-and-anchor records, which are also available as a flat
image list. A citation resolves to a link like
`https://arxiv.org/html/2005.11401v4#S4.SS1.p1` — the exact passage, not a page.

The source was chosen by measurement, not preference: 40 papers surveyed, 39 have
arXiv HTML, and the source decision used a 12-paper answer key built from LaTeX.
The shipping loading pipeline finds all 120 sampled passages, all 1,211 passage anchors
resolve, and all 1,211 useful HTML blocks are emitted. It retains 99.91% of
reference words overall; the worst benchmark paper retains 99.53% against the
95% rule.

## How this is built

Components are developed one at a time, with decisions informed by data and
tests.

Working notes — the product definition, the runtime map, the seven principles,
the sprint records and the decision log — are kept locally and are not published
with the code.
