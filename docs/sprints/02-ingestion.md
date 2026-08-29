# Sprint 02 — Corpus ingestion

**Status:** ✅ closed · final Chroma rebuilt with 384 Documents from 12 papers · 59 tests passing locally

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
| Oversized source passage | 9/1,206 sampled passages exceed 350 words, but only 2 exceed 600; the largest is a 1,330-word table. | ✅ Split only source passages above 600 words. Prose uses sentence boundaries and tables use row boundaries, packing parts toward 350 words. A single sentence or row above the target stays whole. Source anchors are retained. |
| Unsectioned content | Front matter such as the abstract has no normal section path but can still be useful evidence. | ✅ Use the explicit breadcrumb `Unsectioned`. |
| Citation mapping | A list of anchors beside combined text does not tell the answer model which text belongs to which anchor. | ✅ Store `text`, `section_path`, `location`, and `kind` for every constituent passage in `source_passages` metadata. |
| Embedding model | `qwen3-embedding:0.6b` was fast but retrieval was weak. The 4B model was tested on the rebuilt sample corpus. | ✅ `qwen3-embedding:4b`. Offline ingestion is slower; query latency remained practical. |
| Vector store | Smallest local stack compatible with Ollama and LangChain. | ✅ Chroma in `chroma_db/`. |
| Intermediate storage | Raw HTML deterministically regenerates Documents. | ✅ Save raw HTML, not Documents. |
| Stable Document identity | Split parts can share the same source anchor, so anchors alone can collide. | ✅ UUID input includes arXiv ID, ordered source anchors, and Document content. Same input stays stable; different split content gets a different ID. |
| Sprint boundary | The only retriever is an experimental script; Sprint 02 should not claim to evaluate production retrieval. | ✅ Close ingestion first. Keep the curated dataset as prepared input for Sprint 03, where production retrieval and its evals belong. |

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
- ✅ Include Document content in stable IDs and include passage text in quick-retriever deduplication.
- ✅ Rebuild Chroma once more after oversized-passage metadata and IDs are final: 384 Documents from 12 papers.
- ✅ Curate a minimal frozen dataset for the next sprint: 24 questions and 36 required evidence units across all 12 benchmark papers.
- ⏭ Implement retrieval metrics and record LangSmith runs in Sprint 03, after a production retriever exists.
- ✅ Make fresh-database rebuilding the explicit, tested rerun behaviour.
- ✅ Prove stored metadata survives a Chroma write/read round trip.
- ⏸ Indexing a 500-paper development corpus is deferred until the ingestion representation is stable.

## Tests

`uv run pytest` reports **59 passing tests**: 32 loader tests, 19 ingestion-
document tests, 2 real Chroma integration tests, and 6 production retrieval
tests. The 53 loading and ingestion tests close this sprint.

| Invariant | State |
| --- | --- |
| Empty, anchorless, and bare-URL passages do not become Documents | ✅ passing |
| Passages merge across subsection boundaries when they fit | ✅ passing |
| A group splits before exceeding 350 words | ✅ passing |
| A group never crosses a main-section boundary | ✅ passing |
| Overlap never crosses a main-section boundary | ✅ passing |
| Breadcrumbs appear once per consecutive section path | ✅ passing |
| Missing section paths become `Section: Unsectioned` | ✅ passing |
| Metadata preserves every source text, section path, kind, and anchor | ✅ passing |
| Every included passage remains present after overlap deduplication | ✅ passing |
| Image metadata is collected from every constituent passage | ✅ passing |
| Documents are non-empty and IDs stay unique after oversized splitting | ✅ passing |
| Document IDs are deterministic and paper-specific | ✅ passing |
| Figure panel labels do not leak into prose | ✅ passing |
| Oversized prose splits at sentence boundaries where possible | ✅ passing |
| Oversized tables split between rows where possible | ✅ passing |
| One sentence or table row above 350 words stays whole | ✅ passing |
| Same anchors with different Document content produce different stable IDs | ✅ passing |
| Retrieval dataset has two unique questions per benchmark paper | 🔶 checked during curation; automated test not tracked |
| Every evidence quote resolves to exactly one loaded source passage | 🔶 checked during curation; automated test not tracked |
| A fresh rebuild removes old records before adding the corpus | ✅ passing with a real temporary Chroma collection |
| Stored metadata survives a Chroma write/read round trip | ✅ passing with a real temporary Chroma collection |

The oversized-passage tests now cover prose boundaries, table row boundaries,
oversized natural units, source anchors, grouping after splitting, and IDs that
change with content. The quick retriever also keeps different text segments that
share one anchor while still deduplicating exact overlap.

## Evals

There was **no production retriever when the frozen dataset was prepared**, and
there is still no tracked retrieval-quality run. Production retrieval has now
opened under Sprint 03; its dataset and metric contract below are not Sprint 02
completion evidence.

### Evidence Recall@5 contract

For each frozen question, the answer key contains one or more required evidence
units. Each unit identifies the source evidence by arXiv ID, source location, and
a short exact evidence quote. A unit may list accepted alternatives when more
than one source passage supports the same fact.

The evaluator retrieves the top five Documents, expands their `source_passages`
metadata, and removes overlap duplicates using `(arxiv_id, location, text)`. A required unit
is covered when a deduplicated source passage has the expected paper and location
and contains one of the unit's normalized evidence quotes. The answer key does not
use retrieval Document IDs or a hash of the complete grouped text, so regrouping
Documents does not invalidate the evidence.

`Evidence Recall@5 = covered required evidence units / all required evidence units`

This deliberately scores source evidence rather than retrieval Document IDs.
Document IDs change when grouping changes, and overlap can place the same source
passage in more than one Document. Questions with one required unit therefore
score either 0 or 1; questions needing several pieces of evidence receive partial
credit when only some are found.

The curated dataset lives at
`../../evals/dataset/retrieval_evidence_dataset.json`. It contains 24 questions,
two for each of the 12 benchmark papers, and 36 required evidence units. Thirteen
questions need one evidence unit; eleven need multiple units. The question mix is
9 method, 8 result, 4 definition, 2 comparison, and 1 explanation question. The
five development probes were not reused as frozen questions.

Every quote was checked against the shipping loader and resolves to exactly one
passage at its recorded paper and location. The builder at
`../../evals/dataset_builders/create_retrieval_evidence_dataset.py` uploads the
examples to LangSmith without silently replacing a stale dataset. OpenEvals will
implement the evaluator in the next step.

The completed manual probe used five questions after rebuilding with
`qwen3-embedding:4b`:

- multi-head attention: useful evidence at rank 1
- positional encoding: useful evidence at rank 1
- scaled dot-product attention: useful evidence at rank 1
- self-attention versus recurrent layers: useful evidence at rank 1
- translation results: abstract at rank 1 and the exact results table at rank 2

Observed query latency was roughly **0.76–1.23 seconds**. This showed a major
improvement over the old passage-per-Document representation, but five hand-picked
questions remain a smoke check, not an eval.

## Done when

1. Every included source passage above 600 words is split near 350 words when a sentence or row boundary allows it. ✅
2. Oversized prose and tables keep all source text and the original clickable anchor. ✅
3. Grouping, breadcrumbs, overlap, metadata, images, and stable IDs have passing tests. ✅
4. The final representation is freshly ingested across all 12 sampled papers. ✅ 384 Documents
5. A retrieved Document can be expanded into exact source passages and clickable citations. ✅ proven again by Sprint 03 production tests
6. Fresh rebuilding is the explicit rerun behaviour and is tested. ✅
7. Stored metadata survives a real Chroma write/read round trip. ✅
8. This document records the final numbers and hands production retrieval to Sprint 03. ✅

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
  stays whole.
- Split Documents can share the same anchor sequence. Document IDs now also use
  content, producing 384 unique IDs for 384 sampled Documents. The quick
  retriever deduplicates by paper, anchor, and text so exact overlap is removed
  without dropping different segments from one anchor.
- Retrieval-eval metrics were discussed but not accepted. Keep that decision
  separate instead of quietly treating suggested metrics as agreed work.
- The retrieval suite is now accepted: Evidence Recall@5, MRR@5, and Context
  Precision@5. Evidence Recall@5 comes first, using source-level evidence after
  overlap deduplication. LangSmith will hold the dataset and runs; OpenEvals will
  provide the evaluators.
- The frozen retrieval dataset now has 24 questions and 36 evidence units across
  all 12 benchmark papers. Curation checks confirmed paper balance, unique
  questions and IDs, and that every exact quote resolves to one shipping-loader
  passage. It is retained as prepared Sprint 03 input; it does not prove retrieval.
- We corrected the sprint boundary after noticing that retrieval still lived only
  under `experiments/`. Production retrieval and retrieval-quality runs belong to
  Sprint 03.
- Reruns now explicitly reset the Chroma collection before rebuilding the complete
  cached corpus. Real temporary-Chroma tests prove old records disappear and the
  full Document content and metadata survive write/read.
- The final local rebuild completed in about 77 seconds with
  `qwen3-embedding:4b`: 384 Documents from 12 papers. Counts by paper were 18,
  31, 32, 48, 55, 22, 35, 24, 46, 36, 9, and 28.

## Next question

Can the production retriever preserve exact source-passage citation behavior and
establish a fully reproducible Evidence Recall@5 baseline? See Sprint 03.
