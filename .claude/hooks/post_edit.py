#!/usr/bin/env python3
"""PostToolUse hook for Edit/Write: format, autofix, and leakage-lint Python files.

Reads the hook payload from stdin (``tool_input.file_path``). For ``.py`` files
inside the project it runs ``ruff format`` and ``ruff check --fix`` through the
project environment, then the leakage lint. Lint findings go to stderr with
exit code 2 so Claude sees them. If uv or the project environment is missing
(for example before ``make setup``), the formatting step is skipped quietly.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _project_dir() -> Path:
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[2]


def _run_ruff(project: Path, file: Path) -> None:
    uv = shutil.which("uv")
    if uv is None or not (project / ".venv").exists():
        return
    for args in (["ruff", "format", "--quiet"], ["ruff", "check", "--fix", "--quiet"]):
        subprocess.run(
            [uv, "run", "--no-sync", *args, str(file)],
            cwd=project,
            capture_output=True,
            timeout=60,
            check=False,
        )


def main() -> int:
    """Hook entry point."""
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    file_path = (payload.get("tool_input") or {}).get("file_path")
    if not file_path or not str(file_path).endswith(".py"):
        return 0
    project = _project_dir()
    file = Path(file_path).resolve()
    try:
        file.relative_to(project.resolve())
    except ValueError:
        return 0
    if not file.exists():
        return 0

    _run_ruff(project, file)

    lint_dir = project / ".claude" / "skills" / "leakage-guard" / "scripts"
    sys.path.insert(0, str(lint_dir))
    try:
        import leakage_lint  # type: ignore[import-not-found]
    except ImportError:
        return 0
    rel = file.relative_to(project.resolve())
    if rel.parts[:2] != ("src", "nvquant"):
        return 0
    findings = leakage_lint.lint_source(rel, file.read_text(encoding="utf-8"))
    if findings:
        print("leakage lint found possible lookahead:", file=sys.stderr)
        for f in findings:
            print(f"  {f}", file=sys.stderr)
        print(
            "Fix it, or mark a deliberate case with '# leakage-ok: <reason>'.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
