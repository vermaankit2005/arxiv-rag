# Principles

Seven principles. Each one is a claim we must be able to *demonstrate*, not a
component we must install. Each has a stated landing point and an honest "not
yet" until then.

| # | Principle | What it means here | Lands when |
|---|-----------|--------------------|-----------|
| 1 | **Model routing** | cheap model by default, strong model only when the request's features justify it | we have cost + quality data per request type. Routing before data is guesswork. |
| 2 | **Evals** | a score, on a frozen dataset, that moves when we change something | the first change whose effect we cannot see by eye |
| 3 | **Guardrails** | scope refusal, evidence gate, citation validation, output schema | with the first grounded answer. This *is* the product. |
| 4 | **Cost control** | a measured USD figure per request and a hard ceiling | the first component that calls a paid model |
| 5 | **Tool contract** | every external thing sits behind a typed port with a fake | the second caller of that thing, or the first test that needs it faked |
| 6 | **Observability** | for any answer: which passages, which model, how long, how much | when we must explain an answer to someone who is not us |
| 7 | **Security** | paper text is untrusted input, never instruction; no write tools | with the first prompt that concatenates paper text |

## How a principle earns its implementation

A principle is satisfied by the smallest thing that demonstrates it.
"Cost control" starts as a printed dollar figure, not a budget service.
"Observability" starts as a JSON run record, not LangSmith.
We upgrade only when the small version demonstrably fails.

## The three we will probably never need

Saying so, with reasons, is stronger than building them.

- **Human approval flows** — the system has no write tools. It reads papers.
- **Autonomous agent loops** — control flow is fixed by design; see `GRAPH.md`.
- **PII masking** — public papers and our own questions. Revisit if accounts arrive.

## The rule that outranks all seven

**No component enters this project without a demonstrated need.**

Before a database, queue, cache, vector store, framework, abstraction layer, or
config system: point at something broken, slow, or wrong *in the running
program*. Not predicted by a document. Actually hit.

Prefer a file to a database, a function to a class, a script to a framework, and
a constant to a setting — until the simple version measurably hurts. When it
does, write down the number that convinced us.
