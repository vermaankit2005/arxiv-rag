# Sprint 04 — Grounded answer generation

**Status:** 🟡 active · grounded answer path implemented · all four answer-quality evaluators implemented · answer-quality dataset frozen · first baseline pending

**Working rule:** Update this file when each decision, implementation step, test,
manual check, or eval run happens. Do not reconstruct the sprint at the end. If
this document disagrees with the code, the code wins and this file is corrected.

## Goal

Question and retrieved source passages in; a useful answer with exact, correctly
placed citations out.

Every factual claim must be supported by the source passage it cites. Citation
IDs must resolve through `RetrievalContext.citations` to the exact arXiv paper and
anchor. Unsupported claims must not survive into the final answer.

This sprint builds the plain answer-generation path only. Scope resolution,
evidence gating, query rewriting, model routing, and the full graph remain later
components unless answer-generation evidence proves one is required now.

## Decisions to make

| Call | What decides it | Outcome |
| --- | --- | --- |
| Production boundary | The graph still contains unproved branches, while retrieval already returns a complete `RetrievalContext`. | ✅ Start with plain production functions under `src/arxiv_rag/`; do not build the graph in this sprint. |
| Answer contract | We need readable answers without losing the claim-to-source connection. | ✅ Normal answer text containing only supplied inline passage IDs such as `[P1]`. No JSON claim structure unless this proves hard to validate. |
| Citation placement | A citation list at the end does not show which passage supports which claim. | ✅ Put `[P#]` immediately after the supported sentence; allow multiple IDs; production code renders compact trusted links. |
| Generation model | The model must follow the answer contract and stay grounded at practical latency and cost. | ✅ Use Ollama `qwen3.8:27b` at temperature 0 through `OLLAMA_BASE_URL`; model comparisons remain evidence-driven. |
| Answer-quality judge | The judge should be fast, locally available, and replaceable without changing evaluator contracts. | ✅ Use Ollama `qwen3.8:27b`. Restrict ordinal scores to `0`, `0.25`, `0.5`, `0.75`, or `1`, and pair/fact decisions to binary choices. |
| Evidence use | The model receives deduplicated source passages with temporary IDs, but it may still ignore or misuse them. | ✅ Prompt allows only supplied passages and IDs, requires inline citations, forbids URLs, and uses one exact insufficient-evidence response. |
| Unsupported claims | The architecture map says unsupported claims are deleted rather than repaired, but this has not been proved in shipping code. | ⬜ Confirm the behavior with examples before accepting it. |
| Partial answers | Retrieved evidence may support only part of a question. | ⬜ Decide whether to answer the supported part, state the limitation, or refuse. |
| Citation validation | String-valid citation IDs do not prove that the cited passage supports the claim. | ✅ Keep citation-ID validity in deterministic tests. Add a semantic citation-support evaluator that checks each cited claim against its attached passage. |
| Corpus activation | Clearing the active collection before parsing and embedding can leave retrieval empty or partial after a failure. | ✅ Parse every paper first, build a uniquely named staging collection, then atomically update the active-collection pointer only after every write succeeds. |
| Answer eval dataset | The generation eval must not inherit the retriever's ranking or misses. | ✅ Curate the 24 contexts independently from cached source HTML through the shipping loader. Freeze each manually supported passage with its immediate source neighbours; do not call the retriever or consume retrieved Documents. |
| First answer metrics | The manual probe showed grounded answers but does not measure whether every factual claim is supported, whether each attached citation supports its claim, or whether required facts are correct and complete. | ✅ Start with groundedness, citation support, correctness, and completeness. Defer directness until the core quality baseline exists. |

## Steps

- ✅ Start from the production `RetrievalContext`: deduplicated passage text plus
  trusted citation IDs and URLs.
- ✅ Define the smallest answer contract: normal text with supplied inline `[P#]` IDs.
- ✅ Create a small fixed probe set from the frozen retrieval questions and inspect
  the production answers manually.
- ✅ Implement answer drafting in `src/arxiv_rag/answering/generator.py`.
- ✅ Add `python -m arxiv_rag.answering` as a small interactive end-to-end entry point.
- ✅ Validate that every emitted citation ID exists in the supplied retrieval
  context.
- ✅ Reject model-written URLs; production code renders supplied IDs as compact
  numbered links such as `[1]`.
- ⬜ Decide and implement the handling of unsupported and partially supported
  claims.
- ✅ Add deterministic tests for the accepted answer and citation contract.
- ✅ Choose the first semantic evaluator contracts: groundedness, citation
  support, correctness, and completeness.
- ✅ Freeze the minimum answer-quality dataset in `evals/dataset/generation_quality_dataset.json`.
- ✅ Add a LangSmith dataset builder for publishing the frozen examples to the UI.
- ✅ Implement the accepted evaluators against shipping code: groundedness, citation support, correctness, and completeness are implemented.
- ⬜ Run the first described LangSmith baseline against shipping code.
- ⬜ Manually inspect failures, record measured results, and choose the next change.

## Tests

Nine production answer-generation tests and six citation-support evaluator tests
pass without calling Ollama. Seven focused correctness and completeness tests were
added but were not run at the user's request. The command-line test covers question input,
retrieval-context handoff, generation, trusted citation rendering, and output.
Citation-support tests cover pair extraction, multiple citations, Markdown and
decimal punctuation, mixed support, missing citations, unknown IDs, and stable
frozen passage IDs. Three focused ingestion-pipeline tests cover parse failure,
embedding/write failure, and activation only after every staged write succeeds.
A storage test checks the active pointer replacement. Three focused MRR tests
cover rank-first matching across multiple evidence units, exact
paper/location/quote identity, and no-match scoring. The full suite has **80
passing tests**.

| Invariant | State |
| --- | --- |
| Empty or missing retrieval context cannot produce an unsupported factual answer | ✅ fixed insufficient-evidence response |
| Every citation ID in the draft exists in the supplied context | ✅ passing |
| Unknown or model-invented citation IDs fail clearly | ✅ passing |
| Model-written URLs fail clearly, including mixed and uppercase schemes | ✅ passing |
| A normal answer without any citation fails clearly | ✅ passing |
| Prompt contains the question, passages, citation rules, and concise formatting rules | ✅ passing |
| Ollama model configuration stays pinned to `qwen3.8:27b` | ✅ passing |
| Whether answers are grounded, correctly cited, correct, and complete | 🟡 all four evaluators implemented; baseline needed |

Prompt quality and whether a passage truly supports a generated claim are eval
questions, not unit-test assertions.

## Evals

The groundedness, citation-support, correctness, and completeness evaluators are
implemented, but there is no accepted automated answer-generation baseline yet. The first run used `gemma4:26b` as both generator
and judge with OpenEvals continuous scoring. It is invalid because the judge
returned values up to `5` despite the intended `0`–`1` range. The configured
judge is now `qwen3.8:27b`; ordinal scores are restricted to five explicit values
from `0` to `1`, while citation-support and fact-coverage decisions are binary.

A six-question manual probe used production retrieval and `qwen3.8:27b`. It covered
method, comparison, and numeric-result questions from the frozen retrieval dataset.
Using a 1–5 judgement of correctness, completeness, citation support, and directness,
the answers scored **5, 4, 5, 5, 4, and 4**, for a **4.5/5 average**. All six answers
were correct, complete, and cited supporting passages. The first repeated weakness
was over-answering: three answers added grounded but unnecessary background instead
of stopping after the requested facts.

The accepted automated evaluation plan has four semantic scores:

- **Groundedness:** judge whether every factual statement in the generated answer
  is supported by the frozen passages.
- **Citation support:** inspect each citation attached to a factual statement. A
  statement-passage pair passes only when that exact cited passage supports the
  statement.
- **Correctness:** judge the generated answer against the frozen required facts
  and passages.
- **Completeness:** report the share of frozen required facts covered by the
  generated answer.

### How each metric uses the dataset

- **Groundedness:** generated answer + frozen passages.
- **Citation support:** generated answer + passage selected by each `[P#]`.
- **Correctness:** generated answer + required facts + passages.
- **Completeness:** covered required facts divided by total required facts.

Citation-ID validity has a known correct output and remains in deterministic unit
tests. It is not an LLM-judged metric. Directness is also deferred for now: the
manual probe exposed over-answering, but semantic support, correctness, and
completeness are the minimum quality baseline.

`evals/dataset/generation_quality_dataset.json` is now the frozen evaluator input.
It keeps the existing 24 questions and question-type metadata, 92 exact source
passages, and 84 atomic required facts. Passage IDs are stable within each example,
and every fact names one or more supporting passage IDs. Evaluator implementation
must follow these references rather than create labels dynamically.

### How the dataset was curated

The frozen context is source-curated and independent of retrieval. No
`PaperRetriever`, vector search, ranking, retrieved Document, or production
`RetrievalContext` was used to select its passages:

1. The existing questions and required-evidence locations were taken from
   `retrieval_evidence_dataset.json`; retrieval output was not taken from it.
2. For every question, required facts and the exact source passages supporting
   them were manually identified and reviewed against the cached arXiv HTML.
   Table facts include the heading, caption, or surrounding passage needed to
   establish what an otherwise unlabeled row measures.
3. Each manually supported passage was frozen together with its immediate previous
   and next passage in source order. Overlapping three-passage neighbourhoods were
   merged. This produced 92 realistic source passages across the 24 examples.
4. Exact text, arXiv ID, location, and section path were exported from the cached
   HTML with the shipping `load_paper_from_html()` parser. Text and anchors were
   not retyped, and passage order follows the source document rather than a search
   rank.
5. A one-off automated validation parsed the JSON, checked unique and sequential
   local IDs, resolved every supporting passage reference, confirmed all original
   evidence units remained represented, and compared every frozen passage's text,
   location, and section path with fresh shipping-loader output. Semantic
   fact-to-passage support was manually reviewed.

Every LangSmith run must use a clear description in plain English, including what
the metric asks and how its score is calculated. Old runs remain immutable when
the contract or metric changes.

## Done when

1. Answer generation runs from production code using `RetrievalContext`. ✅
2. The answer contract makes claim-to-citation mapping explicit. ✅
3. Model-invented citation IDs and URLs cannot reach the rendered answer. ✅
4. Unsupported and partially supported claims have an explicit, tested behavior. ⬜
5. Deterministic answer and citation invariants have passing tests. ✅
6. At least one answer-quality evaluator runs against a frozen dataset and shipping code. ⬜
7. The first LangSmith baseline has a plain-English description, measured results, and manual review. ⬜
8. This sprint records the final decisions and leaves one clear next question. ⬜

## Log

- Sprint opened after Retrieval Sprint 03 closed with Evidence Recall@5 `0.9375`,
  historical MRR@5 `0.7847`, and Document Precision@5 `0.275` across 24 frozen
  questions. The corrected MRR evaluator later scored `0.8368` in LangSmith.
- Retrieval owns source-passage parsing, exact overlap deduplication, temporary
  citation IDs, and trusted arXiv URLs. Answering owns final link rendering.
- The answer model is not trusted to create URLs or decide citation identity. It
  may select supplied citation IDs; production code owns their meaning and
  rendering.
- The accepted draft is normal prose with `[P#]` immediately after supported
  sentences. Multiple passages may support one sentence, and separate factual
  claims repeat their citations.
- LangGraph is deliberately deferred. The current path is linear, and `GRAPH.md`
  says to build plain functions until real branches or the bounded rewrite loop
  make normal control flow unreadable. Tested functions can become graph nodes
  later without moving their logic into the graph.
- The architecture graph remains a map, not an implementation plan. This sprint
  builds plain functions first and adds no branch that has not earned its place.
- No answer metric or threshold has been accepted yet. The sprint will inspect
  actual generated failures before deciding what deserves an eval.
- `src/arxiv_rag/answering/generator.py` now owns the Ollama model setup, prompt,
  response-type check, and deterministic citation-ID and URL validation. Tests
  inject a tiny fake model instead of calling the network.
- A real `qwen3.8:27b` smoke check answered the supplied Transformer question in
  one sentence and placed `[P1]` immediately after the supported claim.
- `src/arxiv_rag/answering/__main__.py` provides the smallest manual path: ask one
  question, retrieve context, generate the cited draft, call the production answer
  renderer, and print the answer.
- The first open-ended definition answer was grounded but too long and visually
  dominated by repeated full citation labels. Markdown rendering now uses compact
  numbered links. A real terminal uses an OSC 8 hyperlink so only clickable `[1]`
  is visible; non-terminal consoles keep `[1]` in the answer and print each URL
  once in a Sources section. The prompt asks for direct Markdown, short
  paragraphs, useful headings only, and no repeated points.
- Rendering was briefly duplicated between retrieval and the temporary CLI. It is
  now owned once by `src/arxiv_rag/answering/renderer.py`; retrieval only supplies
  the trusted citation map, and every caller uses `render_answer()`.
- Session handoff verified the complete working tree before commit: all **67 tests**
  passed in 3.98 seconds, primary LSP diagnostics were clean across the changed
  production and test files, and session diagnostics reported no issues. The
  remaining Sprint 04 work is the fixed probe set, unsupported/partial-answer
  policy, and the first answer-quality eval.
- Corpus rebuilding no longer clears the active Chroma collection first. It now
  parses all papers before creating a uniquely named staging collection, deletes
  an incomplete staging collection after an embedding or write failure, and
  atomically replaces `chroma_db/active_collection.txt` only after the complete
  corpus is stored. Existing databases without the pointer keep using the legacy
  `arxiv_papers` collection until the first successful rebuild.
- URL validation now matches `http://` and `https://` without regard to scheme
  case. Focused tests cover lowercase, uppercase, and mixed-case model output.
- MRR evaluation now checks Documents in retrieval-rank order and returns the
  reciprocal rank of the first Document containing any required evidence. A
  match requires the same arXiv ID and passage location plus the accepted quote,
  consistent with the other retrieval metrics. Three focused evaluator tests
  guard the earlier evidence-ordering and identity-matching defects. The corrected
  LangSmith run `retriever-mrr-at-5-9c2543fe` completed all 24 questions and scored
  **0.8368**. The evaluator now loads `.env` explicitly before constructing its
  LangSmith client. The full suite passes **74 tests**.
- A manual production probe ran six frozen questions: Transformer masking, BERT
  GLUE averages, DPO/PPO out-of-distribution win rates, RAG-Sequence versus
  RAG-Token, GitSkills collection and deduplication, and physical-circuit success
  by complexity. Scores were **5, 4, 5, 5, 4, and 4**, averaging **4.5/5**. Every
  answer contained the required facts with supported citations. The repeated
  weakness was grounded but unnecessary expansion in three answers; no unsupported
  factual claim was observed in this sample.
- The first generation evaluator scope is now fixed. Groundedness will judge
  whether every factual statement is supported by the frozen passages. Citation
  support will judge statement-passage pairs. Answer quality will judge correctness
  and coverage of frozen required facts. Citation-ID validity stays in unit tests,
  and directness is deferred until after the core semantic baseline.
- The first dataset draft incorrectly froze production retrieval contexts, making
  the generation eval dependent on retriever ranking and misses. That draft was
  replaced before any evaluator or baseline used it.
- The answer-quality dataset is frozen at
  `evals/dataset/generation_quality_dataset.json`. It preserves all 24 questions
  and their existing type metadata, but its context is now curated independently
  from cached source HTML. Each manually supported passage is accompanied by its
  immediate source neighbours, with overlapping neighbourhoods merged. No
  retriever, vector search result, retrieved Document, or `RetrievalContext` was
  used to select the final 92 passages.
- The dataset contains 84 atomic required facts. Supporting passage references
  were manually reviewed, including table captions where a table body alone did
  not establish the experiment. Automated validation confirmed valid JSON, 24
  unique example IDs, sequential unique local passage and fact IDs, non-empty and
  resolvable support lists, unchanged questions and metadata, all 36 original
  evidence units represented, and exact text/location/section-path identity
  against the shipping loader for all 92 passages. No evaluator, LangSmith
  dataset, generation baseline, or production-code change was made.
- `GenerationQualityDatasetBuilder` now publishes the frozen questions and source
  passages as LangSmith inputs, required facts as reference outputs, and preserves
  the example ID and question type as metadata. It refuses to replace an existing
  dataset so historical evaluation inputs cannot change silently.
- The first OpenEvals groundedness run used `gemma4:26b` as both generator and
  judge with `continuous=True`. Its feedback contained scores of `1` and `5`, so
  the reported aggregate was outside the metric's intended `0`–`1` range and the
  run is not an accepted baseline. OpenEvals described the range but did not
  enforce numeric bounds in its schema. The judge is now pinned to Ollama
  `qwen3.8:27b`; ordinal evaluators permit only `0`, `0.25`, `0.5`, `0.75`, or
  `1`, and binary evaluators use explicit choices.
- Citation support is implemented in
  `evals/answering/evaluate_generation_citation_support.py`. It deterministically
  attaches every `[P#]` marker to the preceding statement, resolves the frozen
  passage ID in code, asks the configured judge for a binary support decision for
  each statement-passage pair, and reports supported pairs divided by all cited
  pairs. Six focused tests cover statement extraction, multiple citations,
  Markdown and decimal punctuation, mixed support, missing citations, unknown
  IDs, and stable frozen passage IDs. The last full-suite run passed **80 tests**.
- Correctness is implemented in
  `evals/answering/evaluate_generation_correctness.py`. It judges the generated
  answer against the frozen required facts and only their named supporting
  passages, uses the restricted five-value score, and leaves omitted facts to the
  completeness metric.
- Completeness is implemented in
  `evals/answering/evaluate_generation_completeness.py`. It makes one binary
  coverage decision per frozen required fact and reports covered facts divided by
  all required facts. Seven focused tests were added for both metrics, including
  frozen-reference construction, score aggregation, invalid scores, missing
  facts, and unknown passage IDs. They were not run at the user's request, and no
  LangSmith baseline has been run yet.

## Next question

Run the first described LangSmith baseline for all four implemented metrics,
inspect the failures manually, and record the measured results.
