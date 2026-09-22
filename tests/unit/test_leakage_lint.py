import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("leakage_lint", ROOT / "tools/leakage_lint.py")
lint = importlib.util.module_from_spec(spec)
sys.modules["leakage_lint"] = lint
spec.loader.exec_module(lint)


def codes(src: str, path: str = "src/nvquant/features/x.py") -> set[str]:
    return {f.code for f in lint.lint_source(Path(path), src)}


def test_each_rule_fires():
    assert "LK001" in codes("y = x.shift(-1)\n")
    assert "LK002" in codes("y = x.rolling(5, center=True).mean()\n")
    assert "LK003" in codes("y = x.bfill()\n")
    assert "LK004" in codes("z = (x - x.mean()) / x.std()\n")
    assert "LK005" in codes("s = StandardScaler()\n")
    assert "LK006" in codes(
        "import hmmlearn\np = model.predict_proba(X)\n", "src/nvquant/models/regime.py"
    )
    assert "LK007" in codes("a, b = train_test_split(X)\n", "src/nvquant/models/m.py")
    assert "LK008" in codes("p = pos.shift(0)\n", "src/nvquant/backtest/e.py")


def test_allowed_cases():
    assert codes("y = x.shift(-1)\n", "src/nvquant/labels/forward.py") == set()
    assert codes("y = x.shift(-1)  # leakage-ok: target construction\n") == set()
    assert codes("y = x.rolling(5).mean()\nz = (x - x.rolling(5).mean())\n") == set()
    assert codes("kf = KFold(5)\n", "src/nvquant/cv/splits.py") == set()


def test_repo_source_is_clean():
    assert lint.main([str(ROOT / "src/nvquant")]) == 0


def test_main_reports_findings(tmp_path, capsys):
    bad = tmp_path / "features" / "bad.py"
    bad.parent.mkdir()
    bad.write_text("y = x.bfill()\n")
    assert lint.main([str(tmp_path)]) == 1
    assert "LK003" in capsys.readouterr().out
