---
name: refresh-results
description: Regenerate reports/results.json and the README results block, then have quant-reviewer check the diff. Run only when the user asks to refresh results.
disable-model-invocation: true
allowed-tools: Bash(make report) Bash(uv run *) Bash(git diff *)
---

# Refresh results

1. Run `make report` (through the experiment-runner agent if it takes long).
   This rebuilds `reports/results.json`, `reports/tearsheet.html`, and rewrites
   the README block between `<!-- RESULTS:START -->` and `<!-- RESULTS:END -->`.
2. Check the block was regenerated, not hand-edited:
   `uv run python ${CLAUDE_SKILL_DIR}/scripts/check_readme_block.py`.
   It exits 1 if any generated block (README, docs/RESULTS.md, and the marked blocks in
   other docs) differs from what `results.json` renders to.
3. Show `git diff --stat reports README.md`.
4. Run the quant-reviewer agent on the diff. If it returns FAIL, fix the cause
   (usually a stale artifact or a claim in prose that no longer matches) and
   repeat from step 1.
