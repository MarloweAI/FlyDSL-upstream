---
name: code-review
description: >
  FlyDSL pull request review for Copilot code review on GitHub. Applies the review-pr skill in
  .claude/skills/review-pr, with rules derived per diff by its triage.py, to catch
  silent-correctness bugs, JIT cache-key mistakes, architecture-capability holes, index-width
  overflow, unbounded buffer descriptors, layout-algebra errors, dead flags, and tests or
  benchmarks that cannot fail.
---

# FlyDSL code review on GitHub

`.claude/skills/review-pr/SKILL.md` is the standard; read it in full first. This file changes only
what a Copilot code review cannot do as written there. Where the two disagree, this file wins.

## Evidence, instead of fetch.sh

Do not run `fetch.sh` or `run_one.sh`; they need a logged-in `gh`. Use `/tmp/review` as `$WORK`.
The checkout is the PR head and stands in for `head/`. From the repository root:

```bash
W=/tmp/review; T=.claude/skills/review-pr/triage.py; mkdir -p $W
git fetch -q origin <PR base branch> && git diff --no-color --no-ext-diff FETCH_HEAD...HEAD > $W/pr.diff
git worktree add -q --detach $W/merge-target FETCH_HEAD
python3 $T rules $W/pr.diff "<PR title>" . > $W/rules.txt
python3 $T expand $W/rules.txt .claude/skills/review-pr/rules.md > $W/rules_expanded.txt
python3 $T scan $W/pr.diff > $W/scan.txt
python3 $T core $W/pr.diff . > $W/core_expected.txt
python3 $T bounds $W/pr.diff . > $W/bounds.txt
python3 $T citest $W/pr.diff . . > $W/ci_coverage.txt
python3 $T benchcov $W/pr.diff . > $W/bench_compare.txt
```

If git finds no merge base, run `git fetch --unshallow origin` and retry. Take CI checks, human
review comments and the `ratio=` rows of the `test (linux-flydsl-*)` job logs from the GitHub MCP
server; if a log is unavailable, the Perf line is `NOT MEASURED` as Step 9 allows.

## Steps 2 to 9

Run Steps 2 to 8.6 as written. No independent reader exists for Step 8.7: write
`NONE AVAILABLE -- Copilot code review has no independent reader` to `$WORK/independent.txt`.
Write the card to `$WORK/card.md` with the review line reading
`Review (advisory, not independently refuted): ...`, then run
`python3 .claude/skills/review-pr/triage.py gates /tmp/review .` and go back to the failing step
until every gate passes.

## Output

Unlike a `review-pr` card, this review is posted. Post each finding of the card as a review comment
on its `file:line`, keeping its three parts, severity and `[verified]` / `[inferred]` tag. Put the
card's one-sentence summary and its Review, CI and Perf lines in the overview. No rule ids.
