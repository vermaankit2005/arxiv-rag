# Project guidance

Read the latest relevant code and documentation before proposing work.

## How we build

- Keep Python code simple and direct. Avoid clever or unnecessary abstractions.
- Keep Python function signatures and imports on one line when they fit reasonably.
- Never run Ruff in this repository, including Ruff lint, fix, or format commands.
  Preserve the existing code formatting and do not run any formatter unless the
  user explicitly requests it.
- Add or update relevant tests for code changes and run them unless the user says
  otherwise.
- Do not create automated tests for evaluation datasets, evaluators, or eval
  scripts unless explicitly requested. Validate evals through real runs and
  manual review; continue testing production code normally.
- Follow a clear requested scope directly. Do not reopen settled decisions or
  introduce alternative designs unless a real blocker makes the agreed approach
  impossible.

## Evaluation contracts

- Keep application-level evals under `evals/application/`, with each metric's
  evaluator in its own file. Put shared application-safety targets and helpers in
  `evals/application/safety.py`.
- Keep one local, versioned application-safety source dataset, then publish one
  filtered, versioned LangSmith dataset per metric. Every safety case must use
  controlled context and belong to exactly one metric.
- Every primary score uses `1 = pass`. Evidence abstention never counts as a
  safety refusal. Evaluators run their uploaded metric dataset directly without
  fetching or filtering examples.
- Every eval script keeps the same shape: module docstring,
  `LANGSMITH_DATASET_NAME`, `EXPERIMENT_PREFIX`, `EXPERIMENT_METADATA`, target,
  evaluator, then a `run_*` that passes `metadata`, `experiment_prefix`, and a
  `description`.
- Name local eval datasets `<level>_<metric>_dataset.json` and give the published
  LangSmith dataset the same basename.
- Never change production behavior merely to make an evaluator pass. Evals must
  measure the unchanged shipping target and expose missing capabilities honestly.
- Claims about papers require valid citations. Generator citation markers are
  `[P1]` or `[P1] [P2]`; prompt only for that format, but also parse grouped
  `[P1, P2]` so a correct answer is not rejected.

## Runtime and application architecture

- Ollama is the generator and semantic-judge provider. Read `GENERATOR_MODEL` and
  `JUDGE_MODEL` from `.env`; never hard-code chat model names in production or
  eval metadata.
- Every Ollama chat, embedding, and evaluation-judge request must include the
  Cloudflare Access service-token headers loaded from `.env`. Never hard-code or
  expose those credentials.
- `answer_question` in `src/arxiv_rag/answering/__main__.py` is the single backend
  entry point for one question. The CLI, UI, and future API layers must use it.
- LangSmith tracing belongs only in `src/arxiv_rag/`. Keep tracing off by default,
  keep tracing code out of the Streamlit UI, and never trace credentials or
  Cloudflare Access headers.

## Documentation

- Keep documentation short, practical, and easy to scan.
- Keep `CLAUDE.md` limited to durable, project-wide decisions and directions. Do
  not put sprint scope, task lists, status, run results, temporary thresholds, or
  implementation history here.
- Keep sprint-specific information in the active file under `docs/sprints/` and
  update it as work happens. If a sprint document disagrees with the code, correct
  the document.
- `docs/production-readiness.md` is the canonical record for production-readiness,
  portfolio, safety, evaluation, and operational ideas.
- Keep `docs/EVALS.md` lean and organized into component, pipeline, and
  application-level evals. Keep detailed history in sprint, decision, experiment,
  or LangSmith records.
- Never edit or regenerate `docs/production-readiness.html` unless explicitly
  requested.
- This project is partly for learning. For a new feature, default to explanation,
  design, and small guided steps. Do not implement or create files until the user
  explicitly requests implementation and the scope is agreed.
