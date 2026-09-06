# v1.0 release status

**Updated:** 6 September 2026
**Status:** Release hardening in progress — not yet tagged `v1.0`

This is the single current sprint and release-status document. The older detailed
sprint files are working history, not competing statements of current status.

## What is built

| Sprint | Outcome | Current state |
| --- | --- | --- |
| 01 — Loading | arXiv HTML loading with exact paragraph anchors | Complete |
| 02 — Ingestion | Source-aware chunking, Chroma ingestion, and atomic index activation | Complete |
| 03 — Retrieval | Evidence-based retrieval evaluation, currently using top eight results | Complete; one precision case remains below the recommended bar |
| 04 — Generation | Grounded answers, citations, partial answers, and refusal behavior | Complete |
| 05 — Pipeline evaluation | Required-fact, fact-citation, and evidence-behavior evaluation | Implemented; canonical release run pending |
| 06 — Application safety | Harm, sensitive-data, prompt-injection, and policy-response checks | Implemented; one policy-response case remains below the recommended bar |
| 07 — Tracing and regression | LangSmith tracing and local regression suites | Implemented; final gates, CI, and release artifact remain |
| 08 — Conversation | Two-route LangGraph workflow, Standard/Easy modes, and fresh retrieval on paper follow-ups | Complete; real 11-turn conversation accepted |

## Current verified snapshot

- The local corpus contains 12 papers and 384 retrieval documents.
- All 1,211 published paragraph anchors are valid.
- The deterministic suite passed **159 tests in 5.77 seconds** on 6 September 2026.
- The real Sprint 08 acceptance conversation completed all 11 turns in one thread:
  four chat turns used no retrieval, seven paper turns retrieved fresh evidence,
  non-latest topic recall worked, and chat answers contained no passage markers or
  model-written URLs.
- Invalid router output now falls back to RAG with the raw user question.
- Generation and pipeline regression suites now use the newer fact-citation
  evaluator. Their current local release floor is **95%**.
- The last published evaluation snapshot reports 16 of 20 checks meeting their
  recommended benchmark. It predates the next canonical run of the current
  top-eight and fact-citation release suite.

## Overall release sprint

Complete these in order. Do not add scaling infrastructure before this list is
closed.

- [x] Align retrieval evaluation names and execution with the production top-eight contract.
- [x] Replace legacy citation-support checks in the active regression suites with fact citation.
- [x] Set a practical 95% release floor for generation and pipeline fact citation.
- [x] Add invalid-router fallback and deterministic chat URL/passage-marker rejection.
- [x] Run and manually review the real 11-turn conversation.
- [ ] Replace the remaining shared 75% regression thresholds with metric-specific release rules.
- [ ] Run one canonical full evaluation against the current code and record completion, latency, models, commit SHA, and dataset hashes.
- [ ] Review every failure and approve the release baselines.
- [ ] Add GitHub Actions for `uv sync` and the deterministic pytest suite.
- [ ] Change `.env.example` so LangSmith tracing is off by default.
- [ ] Remove stray local artifacts, confirm a clean release diff, and tag `v1.0`.

## Current release blockers

1. Most regression metrics still use temporary 75% thresholds rather than their
   metric-specific standards.
2. The current top-eight and fact-citation suite does not yet have one canonical,
   machine-readable release result.
3. The latest published snapshot still has known misses in HTML block coverage,
   retrieval precision, and policy-response accuracy.
4. Normal test CI is not configured; the existing workflow publishes only the
   evaluation dashboard.
5. LangSmith tracing is still enabled by default in `.env.example`.

## Public documentation policy

The public narrative documentation is intentionally limited to:

1. `README.md` — stable project overview, architecture, results, and local setup.
2. `docs/release-status.md` — this concise overall sprint and current release state.

The generated benchmark site under `evals/docs/` remains a published supporting
artifact, not another planning or status document. Detailed sprint notes,
experiments, and production-readiness drafts stay local and ignored by Git. When
status changes, update this file instead of adding another sprint-status document.

## Deferred until after v1.0

- A 5,000-paper corpus experiment.
- Weaviate or another vector-database migration.
- Kubernetes, distributed queues, and production-scale API infrastructure.
- Durable conversation memory and autonomous tool-calling loops.
