# Project guidance

Read the latest relevant code and documentation before proposing work.

## How we build

- All Python code must be simple and easy to understand. Prefer clear, direct
  code over clever or unnecessarily abstract code.
- Keep Python function signatures and imports on one line when they fit
  reasonably. Do not wrap them into parenthesized multiline formatting by default.
- Any requested implementation or code change includes adding or updating the
  relevant tests and running them, unless the user explicitly says otherwise.
- Do not create automated tests for evaluation datasets, evaluators, or eval
  scripts unless the user explicitly requests them. Validate evals through real
  runs and manual review. Continue testing production code normally.
- When the user's requested scope is clear, follow it directly. Do not introduce
  alternative designs, reinterpret settled decisions, or reopen them unless a
  real blocker makes the requested approach impossible.
- Keep application-level evals under `evals/application/`, with each metric's
  evaluator in its own file. Put shared application-safety targets and helpers in
  `evals/application/safety.py`.
- Keep one local, versioned application-safety source dataset, then publish one
  filtered, versioned LangSmith dataset per metric. Every safety case must use
  controlled context and belong to exactly one metric. Every primary score uses
  `1 = pass`; evidence abstention never counts as a safety refusal. Evaluators run
  their uploaded metric dataset directly without fetching or filtering examples.
- Every eval script keeps the same shape: module docstring, `LANGSMITH_DATASET_NAME`,
  `EXPERIMENT_PREFIX`, `EXPERIMENT_METADATA`, target, evaluator, then a `run_*`
  that passes `metadata`, `experiment_prefix`, and a `description`.
- Name local eval datasets `<level>_<metric>_dataset.json` and give the published
  LangSmith dataset the same basename.
- Never change production behavior merely to make an evaluator pass. Evals must
  measure the unchanged shipping target and expose missing capabilities honestly.
- Generator citation markers are `[P1]` or `[P1] [P2]`. Prompt for that format
  only. Still parse grouped `[P1, P2]` so a correct answer is not rejected.
- Every Ollama chat, embedding, and evaluation-judge request must include the
  Cloudflare Access service-token headers loaded from `.env`; never hard-code
  those credentials. Ollama is the generator and semantic-judge provider.
- Read `GENERATOR_MODEL` and `JUDGE_MODEL` from `.env`. Do not hard-code chat
  model names in generator, judge, or eval metadata.

## Documents

Keep documentation short, practical, and easy to scan.

### Persistent project guidance

- Whenever the user makes or confirms a durable project decision, proactively
  record it in this file during the same turn without waiting to be asked. Do not
  rely only on the conversation to remember important instructions.
- `docs/production-readiness.md` is the canonical record for production-readiness,
  portfolio, safety, evaluation, and operational ideas discussed for this project.
- Keep `docs/EVALS.md` lean, practical, and organized into component, pipeline,
  and application-level evals. Preserve important dataset-selection principles,
  metric contracts, accepted results, and current status; keep detailed history
  in sprint, decision, experiment, or LangSmith records instead of the main guide.
- Never edit or regenerate `docs/production-readiness.html` unless the user
  explicitly asks for that file to be changed.
- The user is building this project partly to learn. When discussing a new
  feature, default to explanation, design, and small guided steps; do not
  implement the feature or create files unless the user explicitly asks for
  implementation. Agree on the exact scope before editing.
- Execute Sprint 07 in this order: build the smallest useful regression
  pipeline, understand and add privacy-safe LangSmith tracing, then improve the
  measured eval failures.
- Keep the regression pipeline deliberately small: plain dictionaries and direct
  functions, no report/config dataclass layers. Print only safe example IDs,
  scores, completion, average, and status. For now, every regression metric uses
  a temporary minimum score of `0.75`. The priority suite uses one fixed 12-of-24
  generation subset; the full suite uses all frozen examples. Local runs do not
  upload to LangSmith unless `--upload` is explicit.
- When eval-driven product improvement resumes after Sprint 07, prioritize
  naturalness, sensitive-data/PII protection, and policy-response accuracy;
  address citation support next.

### Sprint documents

- Update the active file in `docs/sprints/` alongside code and decisions, not at
  the end of the sprint.
- Keep the established structure: status, goal, decisions, steps, tests, evals,
  done conditions, log, and next question.
- Record measured numbers and clearly distinguish completed work, manual checks,
  open decisions, and deferred work. If the document disagrees with the code,
  the code wins and the document must be corrected.
- Sprint 07 combines LangSmith application tracing with the evaluation regression
  pipeline; do not build a custom tracing system. Tracing stays off by default
  and must not export raw user, prompt, passage, answer, credential, or header content.
