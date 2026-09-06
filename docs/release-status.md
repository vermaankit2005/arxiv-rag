# v1.0 release status

**Released:** 6 September 2026

**Tag:** `v1.0.0`

**Status:** Released as a measured local prototype

v1.0 freezes the first complete, reviewable version of arxiv-rag. It is a local
research-paper assistant and engineering case study, not a production service.
The release deliberately keeps the corpus small so its provenance and evaluation
evidence remain inspectable.

## Release scope

- Load 12 frozen arXiv HTML papers while preserving section structure and exact
  paragraph anchors.
- Build 384 source-aware Chroma retrieval documents through staged ingestion and
  atomic index activation.
- Retrieve the top eight documents and preserve the original passages needed for
  trustworthy citations.
- Generate grounded answers with code-resolved source links, partial-answer and
  insufficient-evidence behavior, and Standard or Easy presentation modes.
- Route chat and paper questions through a bounded LangGraph workflow. Paper
  follow-ups always retrieve fresh evidence rather than treating conversation
  history as evidence.
- Provide both a Streamlit interface and a command-line entry point through the
  same backend answer path.
- Ship frozen component, pipeline, and application-safety datasets, metric-specific
  release thresholds, priority/full regression runners, and a public benchmark
  scorecard.
- Run deterministic tests and semantic regression checks in GitHub Actions.
  LangSmith uploads are opt-in, while manual and master eval results remain in
  `evals/results/`.

## Release verification

- The deterministic suite passed **173 tests** on 6 September 2026.
- The local corpus contains **12 papers and 384 retrieval documents**.
- All **1,211 published paragraph anchors** are valid.
- The reviewed 11-turn conversation acceptance run correctly separated four chat
  turns from seven paper turns and retrieved fresh evidence for paper follow-ups.
- The latest published evaluation snapshot reports **16 of 20 checks** meeting
  their recommended benchmark.

A new full semantic evaluation was intentionally **not** run for this release.
The published evaluation snapshot therefore describes the latest observed system
quality; it is not a certification of the exact tagged commit. This limitation is
recorded rather than hidden.

## Known limitations

- The benchmark covers 12 GenAI papers, not a broad or production-scale corpus.
- The latest snapshot includes known misses in HTML block coverage, one retrieval
  precision case, a legacy live-citation monitoring metric, and one policy-response
  case.
- Two newer fact-citation checks reached 100% on stored-answer replay but still
  require broader manual review.
- There is no public API, production traffic validation, durable conversation
  memory, or measured fallback for papers without arXiv HTML.
- Citation validation rejects invented IDs and URLs, but does not provide a formal
  sentence-level proof for every generated claim.

## LangSmith and result retention

- Application tracing is off by default.
- Deterministic tests permanently disable LangSmith tracing, test tracking, and
  credentials.
- Pull-request eval runs never upload to LangSmith.
- Manual eval runs can opt in through the workflow checkbox.
- Master eval runs upload only when the `EVAL_UPLOAD_TO_LANGSMITH` repository
  variable is set to `true`.
- Manual and master eval runs commit their complete JSON result to
  `evals/results/`; every run also exposes a GitHub Actions summary and artifact.

## Deferred beyond v1.0

- A 5,000-paper corpus experiment.
- Weaviate or another vector-database migration.
- Kubernetes, distributed queues, and production-scale API infrastructure.
- Durable conversation memory and autonomous tool-calling loops.
