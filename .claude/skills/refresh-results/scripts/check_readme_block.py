#!/usr/bin/env python3
"""Exit 1 if the README results block doesn't match what results.json renders to."""

from __future__ import annotations

import sys
from pathlib import Path

from nvquant.reporting.readme import extract_block, render_results_block
from nvquant.reporting.results import load_results


def main() -> int:
    """Compare the committed README block with a fresh render."""
    readme = Path("README.md").read_text(encoding="utf-8")
    expected = render_results_block(load_results(Path("reports/results.json")))
    if extract_block(readme).strip() != expected.strip():
        print("README results block is stale or hand-edited; run `make report`.")
        return 1
    print("README results block matches reports/results.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
