# arxiv-rag — working agreement

Read `docs/PROCESS.md` and `docs/PRINCIPLES.md` before proposing work.

## The rule you must hold me to

**No component enters this project without a demonstrated need.**

If I ask for a database, queue, cache, vector store, framework, abstraction
layer, or config system before something in the *running program* is broken,
slow, or wrong — say so, and make me justify it. Predicted need is not need.

## How we build

- One component at a time. Finish it, decide it with data, test it, document it,
  then choose the next one.
- Every judgement call between tools gets a throwaway benchmark in
  `experiments/` and a record in `docs/decisions/`.
- Tests encode invariants a user would care about, or a bug that actually
  happened. Nothing else. No coverage targets.
- Prefer a file to a database, a function to a class, a script to a framework,
  and a constant to a setting.

## Reference repo

`D:\Ankit\Projects\arxiv-rag` is an earlier top-down attempt. It has good
thinking in `docs/PRODUCT.md`, `docs/ARCHITECTURE.md`, and `CORPUS_STRATEGY.md`
— read it for the *idea* and for smart decisions already reasoned through.

**Do not copy its plan or its code.** It specifies a finished system and built
790 lines of infrastructure before answering a single question about a paper.
That is the mistake this repo exists to avoid.

## Documents

Short and scannable. If a document is long enough to be overwhelming, it has
failed. `docs/` holds five files and they should stay that way.
