from pathlib import Path

import pytest
from pydantic import ValidationError

from nvquant.config import config_hash, deep_merge, load_config

ROOT = Path(__file__).resolve().parents[2]


def test_profiles_load_and_differ():
    full = load_config(ROOT / "configs/full.yaml")
    fast = load_config(ROOT / "configs/fast.yaml")
    assert full.profile == "full" and fast.profile == "fast"
    assert fast.data.source == "synthetic" and full.data.source == "yahoo"
    assert config_hash(full) != config_hash(fast)
    # deep merge keeps base params the profile doesn't override
    assert fast.models["lgbm"].params["num_leaves"] == full.models["lgbm"].params["num_leaves"]
    assert fast.models["lgbm"].params["n_estimators"] == 60


def test_hash_is_stable():
    a = load_config(ROOT / "configs/full.yaml")
    b = load_config(ROOT / "configs/full.yaml")
    assert config_hash(a) == config_hash(b)
    assert len(config_hash({"x": 1})) == 12


def test_deep_merge():
    assert deep_merge({"a": {"b": 1, "c": 2}, "d": 1}, {"a": {"b": 3}}) == {
        "a": {"b": 3, "c": 2},
        "d": 1,
    }


def test_unknown_keys_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(f"extends: {ROOT / 'configs/base.yaml'}\ncv: {{initail_train: 3}}\n")
    with pytest.raises(ValidationError):
        load_config(p)


def test_exogenous_lag_must_be_positive():
    with pytest.raises(ValidationError):
        load_config(ROOT / "configs/fast.yaml", overrides={"features": {"exogenous_lag": 0}})


def test_top_level_must_be_mapping(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_universe_tickers_unique_target_first():
    cfg = load_config(ROOT / "configs/full.yaml")
    t = cfg.universe.all_tickers()
    assert t[0] == "NVDA" and len(t) == len(set(t))
