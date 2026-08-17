# arxiv-rag

An evidence-grounded reading assistant for language-centric GenAI research
papers. Every claim it makes points at the passage it came from.

**Status: sprint 1 — document loading.** Nothing works yet.

## Documents

| File | What it holds |
|------|---------------|
| [docs/PRODUCT.md](docs/PRODUCT.md) | what it does, what it refuses, the citation contract |
| [docs/GRAPH.md](docs/GRAPH.md) | the runtime map — nodes and edges, nothing more |
| [docs/PRINCIPLES.md](docs/PRINCIPLES.md) | the seven principles and where each one lands |
| [docs/PROCESS.md](docs/PROCESS.md) | how a component gets built, decided, and tested |
| [docs/ROADMAP.md](docs/ROADMAP.md) | the likely order of components |
| [docs/decisions/](docs/decisions/) | one record per decided component, with the numbers |

## How this is built

Bottom-up. One component at a time, each one decided with data and tested before
the next begins. No component enters without a demonstrated need — see
`docs/PRINCIPLES.md`.
