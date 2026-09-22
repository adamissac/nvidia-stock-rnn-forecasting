#!/usr/bin/env python3
"""Exit 1 if any generated block in README.md or docs/ differs from what results.json renders to."""

from __future__ import annotations

import sys
from pathlib import Path

from nvquant.reporting.docs_gen import generated_blocks, render_results_md
from nvquant.reporting.readme import extract_block, has_block
from nvquant.reporting.results import load_results


def main() -> int:
    """Compare every committed generated block with a fresh render."""
    res = load_results(Path("reports/results.json"))
    blocks = generated_blocks(res)
    stale = []
    readme = Path("README.md").read_text(encoding="utf-8")
    for required in ("RESULTS", "WHAT_DIDNT_WORK"):
        if not has_block(readme, required):
            stale.append(f"README.md is missing the {required} block")
    for path in [Path("README.md"), *sorted(Path("docs").glob("*.md"))]:
        text = path.read_text(encoding="utf-8")
        if path.name == "RESULTS.md":
            # the generated timestamp line aside, the file must match a fresh render
            if text.split("\n", 4)[4:] != render_results_md(res).split("\n", 4)[4:]:
                stale.append(str(path))
            continue
        for name, block in blocks.items():
            if has_block(text, name) and extract_block(text, name).strip() != block.strip():
                stale.append(f"{path} [{name}]")
    if stale:
        print("stale or hand-edited generated blocks (run `make report`):\n  " + "\n  ".join(stale))
        return 1
    print("every generated block matches reports/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
