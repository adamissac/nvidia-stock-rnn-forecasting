#!/usr/bin/env python3
"""Make LightGBM importable on macOS without Homebrew's libomp.

LightGBM's macOS wheel links ``@rpath/libomp.dylib`` and only searches Homebrew
and MacPorts paths. PyTorch's wheel ships the same OpenMP runtime, so when
``import lightgbm`` fails this adds torch's ``lib`` directory as an rpath and
re-signs the dylib ad hoc. Using torch's copy also means only one OpenMP
runtime is ever loaded. It does nothing on other platforms or when LightGBM
already imports. Run by ``make setup``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def _imports() -> bool:
    return (
        subprocess.run([sys.executable, "-c", "import lightgbm"], capture_output=True).returncode
        == 0
    )


def main() -> int:
    """Patch the rpath if needed. Returns 0 on success or when nothing is needed."""
    if sys.platform != "darwin" or _imports():
        return 0
    lgb = importlib.util.find_spec("lightgbm")
    torch = importlib.util.find_spec("torch")
    if lgb is None or torch is None or lgb.origin is None or torch.origin is None:
        print("lightgbm or torch not installed; run uv sync first", file=sys.stderr)
        return 1
    dylib = Path(lgb.origin).parent / "lib" / "lib_lightgbm.dylib"
    torch_lib = Path(torch.origin).parent / "lib"
    if not (torch_lib / "libomp.dylib").exists():
        print("torch has no bundled libomp; run `brew install libomp`", file=sys.stderr)
        return 1
    subprocess.run(["install_name_tool", "-add_rpath", str(torch_lib), str(dylib)], check=False)
    subprocess.run(["codesign", "--force", "-s", "-", str(dylib)], check=False)
    if not _imports():
        print("could not make lightgbm importable; run `brew install libomp`", file=sys.stderr)
        return 1
    print(f"patched {dylib} to use {torch_lib / 'libomp.dylib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
