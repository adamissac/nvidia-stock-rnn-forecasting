import pytest

from nvquant.reporting.readme import END, START, extract_block, fmt, inject, label


def test_fmt_and_label():
    assert fmt(None) == "n/a" and fmt(float("nan")) == "n/a"
    assert fmt(0.1234, ".1f", pct=True) == "12.3%"
    assert label("bh_target", "NVDA") == "Buy and hold NVDA"
    assert label("ridge__voltarget", "NVDA") == "`ridge` + voltarget"


def test_inject_and_extract(tmp_path):
    p = tmp_path / "README.md"
    p.write_text(f"# t\n\n{START}\nold\n{END}\n\nafter\n")
    inject(p, "new | table")
    text = p.read_text()
    assert extract_block(text) == "new | table" and text.endswith("after\n")
    p.write_text("no markers")
    with pytest.raises(ValueError):
        inject(p, "x")


def test_build_results_refuses_missing_inputs(tmp_path, fast_cfg):
    from nvquant.reporting.results import build_results

    with pytest.raises(FileNotFoundError, match=r"data_quality\.json"):
        build_results(fast_cfg, tmp_path)


def test_generated_blocks_never_touch_unlisted_docs(tmp_path):
    from nvquant.reporting.docs_gen import generated_targets
    from nvquant.reporting.readme import END, START

    spec = tmp_path / "SPEC.md"
    spec.write_text(f"the README block between {START} and {END}\n")
    targets = generated_targets(tmp_path, tmp_path / "README.md")
    assert spec not in targets
