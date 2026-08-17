# How we work

## The loop, per component

1. **Take one small component.** One. Not a layer, not a sprint of five things.
2. **Find the candidate approaches.** Two or three real options, named.
3. **Decide with data, not taste.** Build a throwaway benchmark in
   `experiments/`, run it on real papers, record the numbers.
4. **Write the decision down** in `docs/decisions/` — what we tried, what we
   measured, what we picked, what would change our mind.
5. **Implement it**, keeping the final graph in mind but not building for it.
6. **Stub whatever it depends on** that does not exist yet.
7. **Test it.** Build the golden dataset the tests need, and commit it.
8. **Run the tests, improve the score**, record before/after.
9. **Document it**, then ask: what is the next best thing this product needs?

Step 9's answer is the next sprint. It is chosen from what is now broken or
missing, not from a list written in advance.

## One sprint at a time

We think about **one sprint completely** before starting it: approach, tests,
tools, and what data will decide the judgement calls. Then we build it. Then we
stop and choose the next one.

Nothing is planned three sprints out. `ROADMAP.md` holds a *likely* order, and
it is expected to change.

## When to write a test

Write one when either is true:

1. **It encodes an invariant a user would care about.**
   "Every citation resolves to a passage that was actually retrieved" — yes.
   "The chunker returns a list" — no.
2. **You just fixed a bug.** One test, named after the bug.

If neither is true, do not write it. No coverage targets. Mock the network, the
clock, and the model — nothing else.

## When to add a component

Never on prediction. Only when something in the *running program* is broken,
slow, or wrong. Write down the number that convinced us. See `PRINCIPLES.md`.

## What "done" looks like for a sprint

- The component works on real data.
- Its decision is recorded with the numbers behind it.
- Its invariants have tests, and they pass.
- The next question is written down.

## Stuck?

Ask one question: **what is the smallest thing that is currently wrong?** Fix
that. If the answer is "nothing is wrong, I just think I should add X" — the
sprint is finished and X does not belong yet.
