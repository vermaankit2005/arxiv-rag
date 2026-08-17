# Decisions

One file per decided component. Named `NN-component.md`.

Every file answers five questions and nothing else:

```markdown
# NN — <component>

**Decided:** YYYY-MM-DD

## The question
What we had to choose between, and why it mattered.

## Candidates
The real options. Two or three. Named, with versions.

## How we measured
The experiment, the data it ran on, and where the script lives
(`experiments/...`). Anyone must be able to re-run it.

## Numbers
The table. Actual results.

## Decision
What we picked, in one sentence, and the margin that decided it.

## What would change our mind
The observation that would make us revisit this.
```

If a decision has no numbers, say so explicitly and say why measuring was not
worth its cost. An honest "we picked the obvious one, here is why measuring
would not have helped" is a valid entry. A silent choice is not.
