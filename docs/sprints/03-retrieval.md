# Sprint 03 — Retrieval

**Status:** ✅ closed · Recall@5 0.9375 · corrected MRR@5 0.8368 · Document Precision@5 0.275 · 59 tests passing at close

## Goal

Question in; ranked retrieval Documents and exact, deduplicated source passages
out through production code.

The retriever must use the final Sprint 02 Chroma representation, preserve exact
arXiv citation links, and be measurable against the frozen 24-question evidence
dataset. The answer model is not part of this sprint until retrieval itself has
a trustworthy baseline.

## Decisions to make

| Call | What decides it | Outcome |
| --- | --- | --- |
| Production boundary | Retrieval started only in `experiments/02_retrieval/quick_retrieval.py`. Experiments are evidence, not shipping code. | ✅ Search, metadata parsing, overlap deduplication, and exact citation context now live in `src/arxiv_rag/retrieval/`. The interactive answer demo remains experimental and imports production retrieval. |
| First metric | Completeness must work before rank and noise metrics are useful. | ✅ Evidence Recall@5 first. MRR@5 and precision follow after its contract is trusted. |
| Frozen answer key | Retrieval needs questions tied to stable source evidence rather than unstable grouped Document IDs. | ✅ Use `evals/dataset/retrieval_evidence_dataset.json`: 24 questions and 36 required evidence units across 12 papers. |
| Experiment tracking | Runs should be easy to recognize and compare in LangSmith. | ✅ Use a stable metric prefix and a few configuration tags. |
| Ingestion quality boundary | Ingestion invariants have known correct outputs; chunk quality matters only through downstream retrieval. | ✅ Keep ingestion correctness in tests and evaluate quality through the production retriever. |
| Precision unit | Passage-level precision averaged 0.07 because the answer key labels minimal evidence while Documents intentionally carry useful neighbours. | ✅ Keep the historical run, explain why it misfit the dataset, and replace it with Document Precision@5. |

## Steps

- ✅ Promote the useful manual-retrieval behavior into production code.
- ✅ Parse and validate `source_passages` metadata at the production boundary.
- ✅ Retrieve the top five Documents through the production vector-store interface.
- ✅ Deduplicate overlap by `(arxiv_id, location, text)` without dropping different split parts that share one anchor.
- ✅ Build exact source-passage context and clickable arXiv citations.
- ✅ Add production retrieval tests; do not use the experimental script as the tested contract.
- ✅ Record a manual production-retriever smoke check against the rebuilt 12-paper corpus.
- ✅ Implement Evidence Recall@5 against production retrieval code.
- ✅ Add simple comparison tags under the stable experiment prefix `retriever-evidence-recall-at-5`.
- ✅ Run the first Evidence Recall@5 LangSmith baseline: 0.9375.
- ✅ Manually inspect the Recall@5 misses.
- ✅ Correct and rerun MRR@5: the original evidence-first evaluator scored 0.7847; ranked-Document-first matching scores 0.8368.
- ✅ Run passage-level Context Precision@5 and record its 0.07085 baseline.
- ✅ Replace the misaligned passage metric with Document Precision@5 under a new experiment prefix.
- ✅ Run the first Document Precision@5 baseline: 0.275.
- ✅ Manually compare passage and Document precision examples and accept Documents as the precision unit.

## Tests

Six production retrieval tests pass. They cover:

- top-k is passed to the vector store
- missing source-passage metadata fails clearly
- each source passage remains paired with its exact paper and anchor
- exact overlap is deduplicated
- different text parts sharing one source anchor are retained
- rendered citations are meaningful and clickable

`uv run pytest` reports **59 passing tests** across the repository: 32 loader,
21 ingestion, and 6 production retrieval tests.

## Evals

The production retrieval baselines are complete across all 24 frozen questions.
The corrected MRR evaluator scores 0.8368 in LangSmith. The five manual questions
from Sprint 02 remain a smoke check only.

All metrics score the top five ranked Documents. A source passage matches when
its arXiv ID and location match and its text contains an accepted quote as an
exact substring. Exact overlap is deduplicated by `(arxiv_id, location, text)`.
Each metric is calculated per question and macro-averaged across questions.

- **Evidence Recall@5** measures completeness: `covered required evidence units / all required evidence units`.
- **MRR@5** measures first useful rank: `1 / rank of first relevant Document`, or `0` when none appears in the top five.
- **Document Precision@5** measures useful retrieval units: `Documents containing required evidence / 5`.

Evidence Recall uses source evidence rather than Document IDs, so normal
regrouping does not invalidate it. Document Precision judges each retrieved
Document separately, including Documents that contain evidence through overlap.

The earlier passage-level Context Precision formula was `relevant deduplicated
source passages / all deduplicated source passages`. Its first LangSmith run,
`retriever-context-precision-at-5-b7438284`, averaged **0.07085**. The dataset labels
minimal evidence rather than every useful neighbouring passage, so the metric
systematically treated valid context as noise. The run is retained as the reason
for changing the precision unit, not presented as a retriever regression.

### Experiment identification

Each metric uses its own stable prefix. The replacement precision experiment uses
`retriever-document-precision-at-5`; the historical passage-level run keeps its
old prefix. Metadata tags identify the metric, evaluation unit, dataset,
embedding model, top-k, corpus, and vector database.

## Done when

1. Retrieval runs from production code rather than importing an experiment. ✅
2. Exact source passages and clickable citations survive retrieval. ✅
3. The production retrieval invariants have passing tests. ✅
4. The rebuilt 12-paper corpus passes a production smoke check. ✅
5. Evidence Recall@5 has one identifiable LangSmith baseline run. ✅
6. The sprint records measured results and leaves one clear next question. ✅

## Log

- Sprint opened after correcting the boundary between ingestion and retrieval.
  The frozen dataset was created early, but it is an acceptance key, not proof
  that retrieval exists.
- Sprint 02 rebuilt Chroma from the final ingestion representation: 12 papers and
  384 Documents.
- The manual script already demonstrated useful source-passage parsing, exact
  overlap deduplication, context labels, and clickable citations. Those behaviors
  are candidates for promotion; its chat prompt and interactive UI remain an
  experiment.
- LangSmith runs use a stable metric prefix and a few readable configuration tags
  so experiments are easy to recognize without adding metadata machinery.
- `PaperRetriever` now owns top-k search. Production helpers validate and expand
  `source_passages` through `get_source_passage_for_a_document()`, deduplicate
  exact overlap, build citation context, and render clickable links. The
  interactive script now consumes those helpers instead of defining the
  retrieval contract.
- A production smoke query, “Why is scaled dot-product attention scaled?”,
  returned 5 Documents and 24 deduplicated source passages. The top result was
  from `1706.03762v7` at distance 0.4738. This is a manual check, not an eval.
- The source-passage reader was simplified after review. It keeps only boundary
  checks that matter: metadata must decode and contain the required text,
  location, and section-path fields.
- Citation URLs now have one source of truth: `RetrievalContext.citations`. The
  model context contains only passage ID, section, and text; trusted URLs are
  attached when rendering the answer instead of being duplicated in the prompt.
- Evidence Recall@5 now runs through the exported production source-passage
  reader. The metric logic was left unchanged; the run adds only simple tags and
  a stable LangSmith experiment prefix.
- Passage Context Precision@5 averaged 0.07085 in
  `retriever-context-precision-at-5-b7438284`. Inspection showed a metric/dataset
  mismatch: minimally labelled evidence made useful neighbouring passages look
  like noise. We kept that run and introduced Document Precision@5 under a new
  prefix instead of overwriting the history.
- After clearing the earlier experiments, four runs were created in order, all
  with plain-English descriptions and formulas: Evidence Recall@5 scored 0.9375
  in `retriever-evidence-recall-at-5-2b3b3489`; the historical MRR@5 run scored
  0.7847 in `retriever-mrr-at-5-3ccdd954`; passage precision scored 0.07085 in
  `retriever-context-precision-at-5-b7438284`; and Document Precision@5 scored
  0.275 in `retriever-document-precision-at-5-3590ed14`. The MRR run is retained
  as history but used evidence ordering rather than retrieval rank ordering.
  The corrected LangSmith run `retriever-mrr-at-5-9c2543fe` completed 24/24
  examples and scored 0.8368.
- Manual review confirmed the Recall@5 misses were scored as intended and that
  Document Precision fits this dataset better than passage-level precision. No
  first-run score was turned into a release threshold.

## Next question

Can answer generation use the retrieved source passages to produce grounded
answers with exact, correctly placed citations?
