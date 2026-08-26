# Sprint 02 — Corpus ingestion

**Status:** 🟡 active · oversized passages now split · segment identity still open · 52 tests passing locally

## Goal

`LoadedPaper` objects in; useful retrieval documents with exact passage-level
citation provenance in Chroma.

The pipeline loads cached arXiv HTML, removes content that cannot become useful
citable evidence, groups related passages, adds section breadcrumbs, embeds the
result, and stores enough metadata to map an answer back to the exact source
passage and arXiv anchor.

Raw HTML remains the recoverable source. We do not save intermediate Document
files.

## Decisions to make

| Call | What decided it | Outcome |
|---|---|---|
| Evidence unit | One source passage was often too small; headings and short neighbouring paragraphs retrieved poorly in isolation. | ✅ Group passages into retrieval Documents. Preserve each original passage separately in metadata. |
| Target size | The improved documents must carry enough context without becoming broad sections. | ✅ Pack source passages up to 350 words. Prefer natural boundaries; one sentence or table row may exceed the target. |
| Section boundary | Treating every subsection as hard would recreate many tiny Documents. | ✅ Main section is hard. Subsections may share a Document and are identified by breadcrumbs. |
| Overlap | Adjacent Documents need continuity, but overlap must not cross an unrelated section. | ✅ Add the previous complete passage only within the same main section. Apply the target while splitting and grouping, not again after overlap. |
| Oversized source passage | 9/1,206 sampled passages exceed 350 words, but only 2 exceed 600; the largest is a 1,330-word table. | 🟡 Split only source passages above 600 words. Prose uses sentence boundaries and tables use row boundaries, packing parts toward 350 words. A single sentence or row above the target stays whole. Source anchors are retained; internal segment identity is still open. |
| Unsectioned content | Front matter such as the abstract has no normal section path but can still be useful evidence. | ✅ Use the explicit breadcrumb `Unsectioned`. |
| Citation mapping | A list of anchors beside combined text does not tell the answer model which text belongs to which anchor. | ✅ Store `text`, `section_path`, `location`, and `kind` for every constituent passage in `source_passages`. |
| Embedding model | `qwen3-embedding:0.6b` was fast but retrieval was weak. The 4B model was tested on the rebuilt sample corpus. | ✅ `qwen3-embedding:4b`. Offline ingestion is slower; query latency remained practical. |
| Vector store | Smallest local stack compatible with Ollama and LangChain. | ✅ Chroma in `chroma_db/`. |
| Intermediate storage | Raw HTML deterministically regenerates Documents. | ✅ Save raw HTML, not Documents. |
| Stable Document identity | Rebuilding the same representation must produce the same IDs. | ✅ arXiv ID + ordered source anchors. Oversized segments must add their segment index before that work is complete. |
| Retrieval eval | The five-question check proved the new representation is better, but it is not a frozen answer key. | 🔶 Open separately. Do not call the manual check an eval. |

## Steps

- ✅ Convert loaded passages into deterministic LangChain Documents.
- ✅ Exclude blank text, missing anchors, and bare-URL passages.
- ✅ Prevent visual labels inside figures from becoming prose passages.
- ✅ Group passages up to 350 words without crossing a main-section boundary.
- ✅ Add one-passage overlap only inside the same main section.
- ✅ Add subsection breadcrumbs and the `Unsectioned` fallback to `page_content`.
- ✅ Preserve every source passage and anchor in JSON metadata for exact citations.
- ✅ Aggregate linked image metadata from every passage in a retrieval Document.
- ✅ Move the embedding model from `qwen3-embedding:0.6b` to `qwen3-embedding:4b`.
- ✅ Rebuild and manually inspect the 12-paper sample corpus: 1,206 loaded passages became 379 retrieval Documents.
- ✅ Make a real answer-model call from the local quick retriever and render exact clickable passage links.
- ✅ Split an individual source passage only when it exceeds 600 words; do not cap the later overlap stage.
- ⬜ Add stable segment identity so multiple parts sharing one source anchor cannot collide or be deduplicated.
- ⬜ Rebuild Chroma once more after oversized-passage metadata and IDs are final.
- ⬜ Prove a safe rerun or make fresh-database rebuilding the explicit supported behaviour.
- ⏸ Indexing a 500-paper development corpus is deferred until the ingestion representation is stable.

## Tests

`uv run pytest` currently reports **52 passing tests**: 32 loader tests, 16
tracked ingestion-document tests, and 4 local quick-retriever tests under the
gitignored `experiments` path.

| Invariant | State |
|---|---|
| Empty, anchorless, and bare-URL passages do not become Documents | ✅ passing |
| Passages merge across subsection boundaries when they fit | ✅ passing |
| A group splits before exceeding 350 words | ✅ passing |
| A group never crosses a main-section boundary | ✅ passing |
| Overlap never crosses a main-section boundary | ✅ passing |
| Breadcrumbs appear once per consecutive section path | ✅ passing |
| Missing section paths become `Section: Unsectioned` | ✅ passing |
| Metadata preserves every source text, section path, kind, and anchor | ✅ passing |
| Image metadata is collected from every constituent passage | ✅ passing |
| Document IDs are deterministic and paper-specific | ✅ passing |
| Figure panel labels do not leak into prose | ✅ passing |
| Oversized prose splits at sentence boundaries where possible | ✅ passing |
| Oversized tables split between rows where possible | ✅ passing |
| One sentence or table row above 350 words stays whole | ✅ passing |
| Split parts have stable unique identities while retaining one source anchor | ⬜ not implemented |
| A second ingestion produces no duplicate records | ⬜ not proved |
| Stored metadata survives a Chroma write/read round trip | ⬜ only checked manually |

The oversized-passage tests now cover prose boundaries, table row boundaries,
oversized natural units, source anchors, and grouping after splitting. Segment IDs and a
full multi-segment text-retention check remain open.

## Evals

There is **no tracked retrieval-quality eval yet**.

The completed manual probe used five questions after rebuilding with
`qwen3-embedding:4b`:

- multi-head attention: useful evidence at rank 1
- positional encoding: useful evidence at rank 1
- scaled dot-product attention: useful evidence at rank 1
- self-attention versus recurrent layers: useful evidence at rank 1
- translation results: abstract at rank 1 and the exact results table at rank 2

Observed query latency was roughly **0.76–1.23 seconds**. This showed a major
improvement over the old passage-per-Document representation, but five hand-picked
questions are a smoke check, not an eval. The retrieval eval, its answer key, and
its scoring rules remain a separate decision.

## Done when

1. Every included source passage above 600 words is split near 350 words when a sentence or row boundary allows it. ✅
2. Oversized prose and tables keep all source text and the original clickable anchor. ✅ splitting and anchors; segmented identity still open
3. Grouping, breadcrumbs, overlap, metadata, images, and stable IDs have passing tests. ✅ except segmented IDs
4. The final representation is freshly ingested across all 12 sampled papers. 🔶 rebuild needed after oversized splitting
5. A retrieved Document can be expanded into exact source passages and clickable citations. ✅ manually and in local quick-retriever tests
6. The supported rerun behaviour is explicit and tested. 🔶
7. This document records the final numbers and leaves one clear next question. 🔶

The 500-paper run is not a Sprint 02 closure condition anymore. Running it before
the representation is final would create expensive throwaway embeddings.

## Log

- Sprint opened with one loader passage mapped to one vector-store Document.
- Early retrieval was poor: tiny prose blocks, standalone headings, and bare URLs
  occupied useful ranks.
- Filtering was kept narrow. Blank text, missing locations, and bare URLs are
  excluded; figures and image metadata remain until data shows they hurt.
- Figure panel labels inside a figure were found leaking through `ltx_p` capture.
  The parser now avoids starting prose capture while inside a figure. The fixture
  changed from 84 to 82 passages.
- Main sections became hard boundaries. Subsections remain soft boundaries so
  short neighbouring passages can form useful evidence.
- The grouping target became 350 words. Source passages up to 600 words stay
  whole; only passages above that separate split threshold are divided.
- One complete passage is overlapped only when both groups share a main section.
- Breadcrumbs were added to embedding text. Repeated passages in the same
  subsection do not repeat the breadcrumb; missing paths use `Unsectioned`.
- The sample corpus produced 379 retrieval Documents from 1,206 loaded passages.
- `qwen3-embedding:4b` took about 2.5 GB on disk and about 9.96 GB VRAM while
  embedding. The observed sample run was fast enough for offline rebuilding.
- Manual retrieval improved all five development questions. Exact translation
  results appeared at rank 2; the other four useful passages appeared at rank 1.
- A citation-design bug remained after retrieval improved: combined text was
  paired with a list of URLs, so the model could not know which claim belonged
  to which anchor.
- Retrieval Documents now keep clean combined `page_content` for embedding and a
  JSON `source_passages` mapping for answering. The local quick retriever assigns
  temporary passage IDs and renders them into exact clickable arXiv links.
- Existing Chroma data must be rebuilt after metadata-shape changes. The quick
  retriever rejects old records with a clear re-ingestion message.
- Oversized-passage measurement found 9/1,206 source passages above 350 words,
  but only 2 above the agreed 600-word split threshold. The largest is 1,330
  words. We split those before grouping and remain lenient after overlap.
- `_group_passages()` now expands an oversized source passage into bounded parts
  before applying its existing merge rules. Prose prefers sentence boundaries,
  tables prefer row boundaries, and a single sentence or row above 350 words
  stays whole. The parts still need stable internal segment identity.
- Retrieval-eval metrics were discussed but not accepted. Keep that decision
  separate instead of quietly treating suggested metrics as agreed work.

## Next question

Once oversized passages and rerun behaviour are settled, **what production
retrieval contract should the answering layer consume, and what evidence would
convince us that retrieval is good enough to build on?**
