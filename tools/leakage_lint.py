#!/usr/bin/env python3
"""Fast static check for common lookahead patterns in nvquant source files.

Usage
-----
    python leakage_lint.py [PATH ...]

With no paths it scans ``src/nvquant``. Exit code 0 means clean, 1 means at
least one finding. Findings print as ``path:line: CODE message``.

A line can opt out with a trailing ``# leakage-ok: <reason>`` comment. The
reason is required so every exception is explained in the code itself.

Rules
-----
LK001  negative ``shift`` outside a ``labels`` package (reads the future)
LK002  ``center=True`` in a rolling window (uses future rows)
LK003  backfill (``bfill``/``backfill``) copies future values into the past
LK004  full-sample mean/std/min/max used to normalize inside ``features``
LK005  scaler or feature selector fit inside ``features`` or ``labels``
LK006  HMM ``predict``/``predict_proba``/``decode``/``score_samples``
       (smoothed posteriors or Viterbi paths) in a module that uses hmmlearn
LK007  shuffled or unpurged sklearn splitters outside the ``cv`` package
LK008  ``shift(0)`` or unshifted position times return in ``backtest``

The lint is deliberately simple and errs toward false positives. The property
tests in ``tests/property`` are the real guarantee; this catches the obvious
mistakes within a second of the edit.
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

OK_MARK = re.compile(r"#\s*leakage-ok:\s*\S+")
NEG_SHIFT = re.compile(r"\.shift\(\s*(?:periods\s*=\s*)?-\s*[\w(]")
CENTER = re.compile(r"center\s*=\s*True")
BACKFILL = re.compile(r"\.bfill\(|\.backfill\(|method\s*=\s*['\"](?:bfill|backfill)['\"]")
SHIFT_ZERO = re.compile(r"\.shift\(\s*(?:periods\s*=\s*)?0\s*\)")
SCALERS = re.compile(
    r"\b(?:StandardScaler|MinMaxScaler|RobustScaler|MaxAbsScaler|QuantileTransformer|"
    r"PowerTransformer|SelectKBest|SelectFromModel|RFE|VarianceThreshold|PCA)\s*\("
)
SPLITTERS = re.compile(
    r"\b(?:train_test_split|KFold|StratifiedKFold|ShuffleSplit|cross_val_score|"
    r"cross_validate|cross_val_predict|GridSearchCV|RandomizedSearchCV)\s*\("
)
HMM_METHODS = {"predict", "predict_proba", "decode", "score_samples"}
STAT_METHODS = {"mean", "std", "min", "max", "var"}
WINDOWED = {"rolling", "expanding", "ewm", "groupby", "resample"}


@dataclass(frozen=True)
class Finding:
    """One lint finding."""

    path: Path
    line: int
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.code} {self.message}"


def _in_package(path: Path, name: str) -> bool:
    return name in path.parts


def _chain_has_window(node: ast.AST) -> bool:
    """Return True if an attribute/call chain contains rolling/expanding/etc."""
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, ast.Call):
            current = current.func
        elif isinstance(current, ast.Attribute):
            if current.attr in WINDOWED:
                return True
            current = current.value
        elif isinstance(current, ast.Subscript):
            current = current.value
        else:
            return False
    return False


def _full_sample_stats(tree: ast.AST) -> Iterator[int]:
    """Yield line numbers where a no-arg .mean()/.std() feeds a subtraction or division."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Sub | ast.Div):
            continue
        for side in (node.left, node.right):
            if (
                isinstance(side, ast.Call)
                and isinstance(side.func, ast.Attribute)
                and side.func.attr in STAT_METHODS
                and not side.args
                and not _chain_has_window(side.func.value)
            ):
                yield side.lineno


def _hmm_calls(tree: ast.AST) -> Iterator[int]:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in HMM_METHODS
        ):
            yield node.lineno


def lint_source(path: Path, source: str) -> list[Finding]:
    """Lint one file's source text.

    Parameters
    ----------
    path : Path
        Path used for package-scoping rules and for reporting.
    source : str
        File contents.

    Returns
    -------
    list of Finding
        Findings sorted by line, excluding lines marked ``# leakage-ok:``.
    """
    lines = source.splitlines()
    findings: list[Finding] = []
    in_labels = _in_package(path, "labels")
    in_features = _in_package(path, "features")
    in_cv = _in_package(path, "cv")
    in_backtest = _in_package(path, "backtest")

    for lineno, text in enumerate(lines, start=1):
        code = text.split("#", 1)[0]
        if NEG_SHIFT.search(code) and not in_labels:
            findings.append(Finding(path, lineno, "LK001", "negative shift reads future rows"))
        if CENTER.search(code):
            findings.append(Finding(path, lineno, "LK002", "centered rolling window"))
        if BACKFILL.search(code):
            findings.append(Finding(path, lineno, "LK003", "backfill copies future values back"))
        if (in_features or in_labels) and SCALERS.search(code):
            findings.append(
                Finding(path, lineno, "LK005", "scaler/selector fit inside features or labels")
            )
        if SPLITTERS.search(code) and not in_cv:
            findings.append(
                Finding(path, lineno, "LK007", "unpurged or shuffled splitter outside cv/")
            )
        if in_backtest and SHIFT_ZERO.search(code):
            findings.append(Finding(path, lineno, "LK008", "shift(0) means same-bar execution"))

    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    if tree is not None:
        if in_features:
            for lineno in _full_sample_stats(tree):
                findings.append(
                    Finding(path, lineno, "LK004", "full-sample statistic used to normalize")
                )
        if "hmmlearn" in source:
            for lineno in _hmm_calls(tree):
                findings.append(
                    Finding(
                        path,
                        lineno,
                        "LK006",
                        "HMM smoothed posterior or Viterbi path; use the forward filter",
                    )
                )

    kept = [
        f for f in findings if not (0 < f.line <= len(lines) and OK_MARK.search(lines[f.line - 1]))
    ]
    return sorted(set(kept), key=lambda f: (f.line, f.code))


def iter_python_files(paths: Iterable[Path]) -> Iterator[Path]:
    """Expand directories into the .py files they contain."""
    for p in paths:
        if p.is_dir():
            yield from sorted(q for q in p.rglob("*.py") if ".venv" not in q.parts)
        elif p.suffix == ".py" and p.exists():
            yield p


def main(argv: list[str] | None = None) -> int:
    """Run the lint and print findings. Returns the process exit code."""
    args = sys.argv[1:] if argv is None else argv
    targets = [Path(a) for a in args] or [Path("src/nvquant")]
    findings: list[Finding] = []
    for file in iter_python_files(targets):
        findings.extend(lint_source(file, file.read_text(encoding="utf-8")))
    for f in findings:
        print(f)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
