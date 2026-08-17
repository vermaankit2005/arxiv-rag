# Product

## In one sentence

An evidence-grounded reading assistant for language-centric GenAI research: it
explains papers without flattening the technical detail, and every claim points
at the passage it came from so you can check it.

## Who it is for

A software or ML engineer who knows basic ML but finds research papers slow and
hard to extract value from. Wants mechanisms, trade-offs, and limitations —
not a simplified summary that removes them.

## The promise

- Explain difficult research clearly **without removing technical depth**.
- Ground every factual claim in the indexed corpus.
- Put the citation next to the claim it supports.
- Say what it does not know, instead of filling the gap from model knowledge.
- Publish exactly which papers it has read.

## What it does

1. Explain a language-centric GenAI concept.
2. Answer questions about a specific indexed paper.
3. Compare indexed papers, methods, or findings.
4. Synthesize findings across indexed papers.
5. Find relevant indexed papers for a topic.

## What it refuses

- General-purpose Q&A.
- Anything outside language-centric GenAI (vision, speech, audio, robotics).
- User-uploaded documents; general web search.
- Serving or redistributing arXiv PDFs.

## Answer statuses

Every request ends in exactly one of these:

| Status | Meaning |
|--------|---------|
| `answered` | the corpus supports the answer |
| `partial` | part is supported; the gap is named |
| `insufficient_evidence` | in scope, but the corpus cannot support an answer |
| `out_of_scope` | outside language-centric GenAI |
| `needs_clarification` | ambiguity would change retrieval or the conclusion |
| `error` | a dependency or validation step failed |

## The citation contract

- Every factual claim carries at least one citation.
- A citation resolves to an indexed paper **and a link to the exact paragraph**
  it came from — not just a title, and not just a page number:

  ```
  https://arxiv.org/html/2005.11401v4#S2.p3
  ```

  The reader clicks the claim and lands on the sentence, on arXiv's own site.
  Each passage also carries its section path, so a citation can be shown as
  "Methods › Retriever: DPR" without following the link.
- Where a paper has no such anchor, the citation falls back to a coarser
  location and **says so**. It never pretends to a precision it does not have.
- The cited passage must actually support the claim; this is checked before the
  answer is shown.
- `answered` and `partial` require at least one valid citation.
- A failure path may never produce an ungrounded answer.

## Success

We can show, with recorded evidence, that it answers supported questions
faithfully, abstains when it should, retrieves the right passages, and produces
citations that survive checking.

---

## Corpus scope — the transparency promise

The product is meaningless unless a user can see **exactly what it has read**.
So the corpus is bounded and published, not "everything on arXiv".

| Constraint | Value | Why |
|---|---|---|
| **Ceiling** | 8,000–10,000 documents | large enough to be useful, small enough to curate honestly and to publish in full |
| **Domain** | language-centric GenAI only | a stated boundary lets the product refuse credibly instead of guessing |
| **Source** | arXiv, categories `cs.CL`, `cs.LG`, `cs.AI` | one source with stable IDs, versions, and licensing we can point at |
| **Date cutoff** | a fixed date, published with the release | "why doesn't it know about X?" gets a precise answer |
| **Published list** | every indexed paper, with its arXiv link | the user can verify the boundary themselves |

Two tracks inside the ceiling:

- **Foundational** — a small set of pre-cutoff papers that established the
  mechanisms everything else builds on. Without these, recent papers are
  unreadable.
- **Recent** — everything in-domain from the cutoff forward, up to the ceiling.

Each release publishes: the cutoff date, the inclusion and exclusion rules, the
counts, and the complete paper list. A paper counts as *covered* only when its
full text is successfully indexed — not when it was merely downloaded.

**Undecided and deliberately so:** the exact cutoff date, the topic filter, and
how the two tracks are sized. Those are decided in the corpus sprint, with data.
What is decided now is the *shape*: bounded, single-domain, dated, published.

---

*Still undecided:* storage, providers, and interface. They are outcomes of the
components we build, not inputs. See `ROADMAP.md`.
