"""One training loop for every sequence model, wrapped as a :class:`Forecaster`.

Per fit: standardize features with training-row statistics, hold out the last
``val_frac`` of training samples (after a 2-sample gap so no validation label
overlaps a training label), train with AdamW, gradient clipping, and early
stopping on validation loss, then keep the best epoch. Repeat for each seed and
average the seeds' outputs.
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd
import torch
from torch import nn

from nvquant.models.base import Forecaster
from nvquant.models.deep.nets import QUANTILES, PatchTSTNet, RNNNet, TCNNet, head_loss

VAL_GAP = 2


def windows(Z: np.ndarray, positions: np.ndarray, seq_len: int) -> np.ndarray:
    """Stack ``Z[p - seq_len + 1 : p + 1]`` for each position p (earlier rows zero-padded)."""
    pad = np.zeros((seq_len - 1, Z.shape[1]), dtype=Z.dtype)
    Zp = np.concatenate([pad, Z])
    view = np.lib.stride_tricks.sliding_window_view(Zp, seq_len, axis=0)  # (n, f, seq)
    return np.ascontiguousarray(view[positions].transpose(0, 2, 1))


class DeepForecaster(Forecaster):
    """LSTM/GRU, TCN, or PatchTST with a Gaussian or quantile head."""

    def __init__(
        self,
        arch: str,
        head: str = "gaussian",
        seq_len: int = 20,
        seeds: int = 3,
        max_epochs: int = 40,
        patience: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        val_frac: float = 0.15,
        grad_clip: float = 1.0,
        device: str = "cpu",
        base_seed: int = 0,
        **arch_params: Any,
    ) -> None:
        self.arch = arch
        self.head = head
        self.seq_len = seq_len
        self.seeds = seeds
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.grad_clip = grad_clip
        self.device = device
        self.base_seed = base_seed
        self.arch_params = arch_params
        self.probabilistic = True
        self.quantiles = QUANTILES if head == "quantile" else ()
        self.name = f"{arch}_{head}"

    def _net(self, n_features: int) -> nn.Module:
        p = self.arch_params
        if self.arch == "rnn":
            return RNNNet(n_features, p.get("hidden", 16), p.get("cell", "lstm"), self.head)
        if self.arch == "tcn":
            return TCNNet(
                n_features, p.get("channels", 16), p.get("levels", 3), p.get("kernel", 3), self.head
            )
        if self.arch == "patchtst":
            return PatchTSTNet(
                n_features, self.seq_len, p.get("patch_len", 8), p.get("stride", 4),
                p.get("d_model", 16), p.get("n_heads", 2), p.get("n_layers", 1), self.head,
            )  # fmt: skip
        raise ValueError(f"unknown arch {self.arch}")

    def _standardize(self, X: pd.DataFrame) -> np.ndarray:
        z = (X.to_numpy(dtype=np.float32) - self.mu_) / self.sd_
        return np.clip(np.nan_to_num(z, nan=0.0), -10, 10).astype(np.float32)

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None) -> Self:
        """Train ``seeds`` networks with early stopping."""
        rows = X.loc[: y.index.max()]
        vals = rows.to_numpy(dtype=np.float32)
        self.mu_ = np.nanmean(vals, axis=0)
        sd = np.nanstd(vals, axis=0)
        self.sd_ = np.where(sd > 0, sd, 1.0).astype(np.float32)
        Z = self._standardize(rows)
        pos = rows.index.get_indexer(y.index)
        Xw = torch.from_numpy(windows(Z, pos, self.seq_len))
        yt = torch.from_numpy(y.to_numpy(dtype=np.float32))
        wt = torch.from_numpy(
            np.ones(len(y), np.float32)
            if sample_weight is None
            else sample_weight.to_numpy(np.float32)
        )
        n_val = max(1, int(len(y) * self.val_frac))
        n_tr = len(y) - n_val - VAL_GAP
        if n_tr < self.batch_size // 4:
            raise ValueError("not enough training samples for the deep model")
        tr = slice(0, n_tr)
        va = slice(n_tr + VAL_GAP, len(y))
        self.nets_: list[nn.Module] = []
        self.epochs_: list[int] = []
        for s in range(self.seeds):
            torch.manual_seed(self.base_seed + s)
            gen = torch.Generator().manual_seed(self.base_seed + s)
            net = self._net(Z.shape[1]).to(self.device)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            best, best_state, best_epoch, bad = np.inf, None, 0, 0
            for epoch in range(self.max_epochs):
                net.train()
                perm = torch.randperm(n_tr, generator=gen)
                for i in range(0, n_tr, self.batch_size):
                    b = perm[i : i + self.batch_size]
                    opt.zero_grad()
                    loss = head_loss(self.head, net(Xw[tr][b]), yt[tr][b], wt[tr][b])
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
                    opt.step()
                net.eval()
                with torch.no_grad():
                    vloss = float(head_loss(self.head, net(Xw[va]), yt[va], wt[va]))
                if vloss < best - 1e-6:
                    best, best_epoch, bad = vloss, epoch, 0
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            if best_state is not None:
                net.load_state_dict(best_state)
            net.eval()
            self.nets_.append(net)
            self.epochs_.append(best_epoch + 1)
        return self

    def _raw(self, X: pd.DataFrame, index: pd.DatetimeIndex) -> np.ndarray:
        rows = X.loc[: index.max()]
        Z = self._standardize(rows)
        Xw = torch.from_numpy(windows(Z, rows.index.get_indexer(index), self.seq_len))
        with torch.no_grad():
            return np.stack([net(Xw).numpy() for net in self.nets_])  # (seeds, n, k)

    def predict_dist(self, X: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Seed-averaged distribution. Gaussian: mixture mean and std. Quantile: averaged quantiles."""
        out = self._raw(X, index)
        if self.head == "gaussian":
            mu = out[:, :, 0]
            var = np.exp(out[:, :, 1])
            m = mu.mean(axis=0)
            v = (var + mu**2).mean(axis=0) - m**2
            return pd.DataFrame({"mean": m, "std": np.sqrt(np.maximum(v, 1e-12))}, index=index)
        q = out.mean(axis=0)
        return pd.DataFrame(
            {"mean": q[:, 1], **{f"q{qq:g}": q[:, i] for i, qq in enumerate(QUANTILES)}},
            index=index,
        )

    def predict(self, X: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
        """Point forecast: the Gaussian mean or the median."""
        return self.predict_dist(X, index)["mean"]
