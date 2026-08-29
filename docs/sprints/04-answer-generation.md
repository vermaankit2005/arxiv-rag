# Sprint 04 — Grounded answer generation

**Status:** 🟡 active · generator and one production answer renderer implemented · compact clickable terminal citations added · 67 tests passing

**Working rule:** Update this file when each decision, implementation step, test,
manual check, or eval run happens. Do not reconstruct the sprint at the end. If
this document disagrees with the code, the code wins and this file is corrected.

## Next-session review queue

Keep these review findings deferred for the next session; they are not part of
this commit:

1. Correct MRR evaluation so ranked Documents are checked first and evidence is
   matched by arXiv ID, location, and quote; add focused evaluator tests.
2. Make corpus rebuilding failure-safe so a failed parse, embedding, or write
   cannot leave the active Chroma corpus empty or partial.
3. Reject model-written URL schemes case-insensitively and add mixed/uppercase
   URL validation tests.

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
| Generation model | The model must follow the answer contract and stay grounded at practical latency and cost. | ✅ Start with Ollama `qwen3.8:27b` at temperature 0 using `OLLAMA_BASE_URL`; compare models only if measured failures justify it. |
| Evidence use | The model receives deduplicated source passages with temporary IDs, but it may still ignore or misuse them. | ✅ Prompt allows only supplied passages and IDs, requires inline citations, forbids URLs, and uses one exact insufficient-evidence response. |
| Unsupported claims | The architecture map says unsupported claims are deleted rather than repaired, but this has not been proved in shipping code. | ⬜ Confirm the behavior with examples before accepting it. |
| Partial answers | Retrieved evidence may support only part of a question. | ⬜ Decide whether to answer the supported part, state the limitation, or refuse. |
| Citation validation | String-valid citation IDs do not prove that the cited passage supports the claim. | 🔶 Deterministic validation now rejects missing IDs, unknown IDs, and model-written URLs; semantic claim support still needs an eval. |
| Answer eval dataset | The 24 retrieval questions identify required evidence, not complete reference answers. | ⬜ Decide whether to extend those examples or freeze a smaller answer-specific set. |
| First answer metric | A single clear quality failure should be measured before adding a metric suite. | ⬜ Choose after inspecting the first generated answers. |

## Steps

- ✅ Start from the production `RetrievalContext`: deduplicated passage text plus
  trusted citation IDs and URLs.
- ✅ Define the smallest answer contract: normal text with supplied inline `[P#]` IDs.
- ⬜ Create a small fixed probe set from the frozen retrieval questions and inspect
  one baseline prompt manually.
- ✅ Implement answer drafting in `src/arxiv_rag/answering/generator.py`.
- ✅ Add `python -m arxiv_rag.answering` as a small interactive end-to-end entry point.
- ✅ Validate that every emitted citation ID exists in the supplied retrieval
  context.
- ✅ Reject model-written URLs; production code renders supplied IDs as compact
  numbered links such as `[1]`.
- ⬜ Decide and implement the handling of unsupported and partially supported
  claims.
- ✅ Add deterministic tests for the accepted answer and citation contract.
- ⬜ Freeze the minimum answer-quality dataset and evaluator only after the failure
  modes are visible.
- ⬜ Run the first described LangSmith baseline against shipping code.
- ⬜ Manually inspect failures, record measured results, and choose the next change.

## Tests

Nine answer-generation tests pass. They use an injected fake model, so normal
test runs never call Ollama. The command-line test covers question input,
retrieval-context handoff, generation, trusted citation rendering, and output.

| Invariant | State |
| --- | --- |
| Empty or missing retrieval context cannot produce an unsupported factual answer | ✅ fixed insufficient-evidence response |
| Every citation ID in the draft exists in the supplied context | ✅ passing |
| Unknown or model-invented citation IDs fail clearly | ✅ passing |
| Model-written URLs fail clearly | ✅ passing |
| A normal answer without any citation fails clearly | ✅ passing |
| Prompt contains the question, passages, citation rules, and concise formatting rules | ✅ passing |
| Ollama model configuration stays pinned to `qwen3.8:27b` | ✅ passing |
| Whether each cited passage semantically supports its claim | ⬜ eval needed |

Prompt quality and whether a passage truly supports a generated claim are eval
questions, not unit-test assertions.

## Evals

There is no accepted answer-generation eval yet. The retrieval scores establish
that evidence can usually be found; they do not prove that an answer uses it
correctly.

The first answer eval must run against shipping code and separate three ideas:

- **Citation validity:** does every citation resolve to a supplied source passage?
- **Citation support:** does that source passage support the claim carrying it?
- **Answer completeness:** did the answer cover the parts of the question that the
  available evidence supports?

Citation validity has a known correct output and belongs in tests. Citation
support and answer completeness require an eval because wording can vary while
remaining correct. Do not accept formulas, thresholds, or a large metric suite
until baseline answers expose the real failure modes.

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
  MRR@5 `0.7847`, and Document Precision@5 `0.275` across 24 frozen questions.
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

## Next question

Does `qwen3.8:27b` follow the inline citation contract across a small fixed set of
real retrieval questions, and what is the first repeated failure mode?
