import numpy as np
import torch

from nvquant.experiments.repro import (
    available_accelerators,
    detect_device,
    git_sha,
    seed_everything,
    sha256_file,
    sha256_text,
)


def test_seed_everything_is_deterministic():
    # seed_everything has to seed numpy's legacy global state, so that's what this reads
    seed_everything(7)
    a = (np.random.rand(3), torch.rand(3))  # noqa: NPY002
    seed_everything(7)
    b = (np.random.rand(3), torch.rand(3))  # noqa: NPY002
    assert np.allclose(a[0], b[0]) and torch.equal(a[1], b[1])


def test_device_defaults_to_cpu():
    assert detect_device("cpu") == "cpu"
    assert detect_device("auto") in {"cpu", "cuda", "mps"}
    assert set(available_accelerators()) == {"cuda", "mps"}


def test_hashing(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("abc")
    assert sha256_file(p) == sha256_text("abc")
    assert git_sha()
