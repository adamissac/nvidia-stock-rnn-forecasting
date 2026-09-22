"""Writing rules from docs/CONVENTIONS.md: no em dashes and no hype words in the docs."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HYPE = re.compile(
    r"\b(cutting-edge|robust|robustly|robustness|leverage[sd]?|leveraging|seamless(ly)?|state-of-the-art|powerful)\b",
    re.I,
)
# CONVENTIONS.md lists the banned words themselves, so it can't be checked against them.
SKIP = {"CONVENTIONS.md"}
DOCS = [
    p
    for p in [
        ROOT / "README.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "CHANGELOG.md",
        ROOT / "legacy" / "README.md",
        *sorted((ROOT / "docs").rglob("*.md")),
    ]
    if p.exists() and p.name not in SKIP
]


@pytest.mark.parametrize("path", DOCS, ids=[str(p.relative_to(ROOT)) for p in DOCS])
def test_no_em_dashes_or_hype(path):
    text = path.read_text(encoding="utf-8")
    assert "—" not in text, f"em dash in {path}"
    bad = sorted({m.group(0) for m in HYPE.finditer(text)})
    assert not bad, f"hype words in {path}: {bad}"
