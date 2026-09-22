"""Small pipeline helpers that decide what gets reported."""

import pytest

from nvquant.experiments.pipeline import best_strategy, peer_config


def test_best_strategy_ignores_nan_sharpes():
    assert best_strategy({"flat": float("nan"), "a": 0.4, "b": 0.9}) == "b"
    assert best_strategy({"a": 0.4, "flat": float("nan")}) == "a"
    with pytest.raises(ValueError):
        best_strategy({"flat": float("nan")})


def test_peer_config_swaps_target_and_keeps_original(fast_cfg):
    pc = peer_config(fast_cfg, "AMD")
    assert pc.universe.target == "AMD"
    assert "NVDA" in pc.universe.peers and "AMD" not in pc.universe.peers
    assert "AMD" in pc.universe.required and "NVDA" not in pc.universe.required
    assert fast_cfg.universe.target == "NVDA" and "AMD" in fast_cfg.universe.peers
