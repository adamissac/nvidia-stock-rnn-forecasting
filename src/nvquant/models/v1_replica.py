"""A PyTorch replica of the v1 notebook's model, run through proper walk-forward.

v1: four stacked ``LSTM(100, activation='relu')`` layers with ``Dropout(0.2)``
between them and a ``Dense(1)`` head, trained with Adam and MSE for 100 epochs
(batch size 32) on MinMax-scaled windows of 60 opening prices, predicting the
next open. Keras's ``activation='relu'`` replaces tanh in the candidate and
output activations, so :class:`ReluLSTMCell` does the same (PyTorch's built-in
LSTM is tanh-only). Recurrent activations stay sigmoid, as in Keras.

What changes versus v1 is only the evaluation: the data is in ascending time
order, the scaler is fit on each training window, every out-of-sample day is
predicted, and the forecast is compared with persistence (tomorrow's open
equals today's open).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

WINDOW = 60
UNITS = 100
LAYERS = 4
DROPOUT = 0.2
EPOCHS = 100
BATCH = 32


class ReluLSTMCell(nn.Module):
    """LSTM cell with ReLU in place of tanh for the candidate and the cell output."""

    def __init__(self, n_in: int, n_hidden: int) -> None:
        super().__init__()
        self.lin = nn.Linear(n_in + n_hidden, 4 * n_hidden)
        self.n_hidden = n_hidden

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One step. Returns (h, c)."""
        h, c = state
        i, f, g, o = self.lin(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.relu(g)
        h = torch.sigmoid(o) * torch.relu(c)
        return h, c


class V1Net(nn.Module):
    """Four stacked ReLU-LSTM layers with dropout and a linear head (282,101 parameters, as in v1)."""

    def __init__(self, units: int = UNITS, layers: int = LAYERS, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.cells = nn.ModuleList(
            [ReluLSTMCell(1 if i == 0 else units, units) for i in range(layers)]
        )
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(units, 1)
        self.units = units

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window, 1)."""
        seq = x
        for cell in self.cells:
            h = torch.zeros(x.shape[0], self.units)
            c = torch.zeros(x.shape[0], self.units)
            outs = []
            for t in range(seq.shape[1]):
                h, c = cell(seq[:, t], (h, c))
                outs.append(h)
            seq = self.drop(torch.stack(outs, dim=1))
        return self.out(seq[:, -1]).squeeze(1)


def _minmax(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """v1's MinMax scaling with bounds fit on the training window."""
    return (v - lo) / (hi - lo)


def n_parameters(net: nn.Module) -> int:
    """Trainable parameter count."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


@dataclass(frozen=True)
class V1Result:
    """Out-of-sample v1 replica forecasts of the next open, plus persistence."""

    frame: pd.DataFrame
    fits: list[dict[str, object]]


def run_v1_replica(
    opens: pd.Series,
    first_test: pd.Timestamp,
    last_test: pd.Timestamp,
    train_len: int = 252,
    refit_every: int = 252,
    epochs: int = EPOCHS,
    seed: int = 0,
) -> V1Result:
    """Walk-forward v1: refit on the trailing ``train_len`` sessions every ``refit_every`` sessions.

    At decision date t (after the close) the input is the 60 opens up to and
    including t's open, and the target is the open of t+1, as in v1.
    """
    o = opens.dropna()
    dates = o.index[(o.index >= first_test) & (o.index <= last_test)]
    target = o.shift(
        -1
    )  # leakage-ok: next open is the v1 target; only past targets are used for training
    rows = []
    fits = []
    for k in range(0, len(dates), refit_every):
        block = dates[k : k + refit_every]
        r = block[0]
        pos = o.index.get_loc(r)
        # training windows end at t-1 so their target (open at t) is known at the refit
        train_end = pos - 1
        train_start = max(WINDOW, train_end - train_len + 1)
        lo, hi = (
            float(o.iloc[train_start - WINDOW : train_end + 2].min()),
            float(o.iloc[train_start - WINDOW : train_end + 2].max()),
        )
        vals = o.to_numpy(dtype=np.float32)
        X = np.stack([vals[i - WINDOW + 1 : i + 1] for i in range(train_start, train_end + 1)])
        y = vals[train_start + 1 : train_end + 2]
        torch.manual_seed(seed + k)
        net = V1Net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        Xt = torch.from_numpy(_minmax(X, lo, hi)[:, :, None].astype(np.float32))
        yt = torch.from_numpy(_minmax(y, lo, hi).astype(np.float32))
        gen = torch.Generator().manual_seed(seed + k)
        final_loss = float("nan")
        for _ in range(epochs):
            net.train()
            perm = torch.randperm(len(yt), generator=gen)
            for i in range(0, len(yt), BATCH):
                b = perm[i : i + BATCH]
                opt.zero_grad()
                loss = nn.functional.mse_loss(net(Xt[b]), yt[b])
                loss.backward()
                opt.step()
                final_loss = loss.item()
        net.eval()
        idx = [o.index.get_loc(d) for d in block]
        Xb = np.stack([vals[i - WINDOW + 1 : i + 1] for i in idx])
        with torch.no_grad():
            pred_scaled = net(
                torch.from_numpy(_minmax(Xb, lo, hi)[:, :, None].astype(np.float32))
            ).numpy()
        pred = pred_scaled * (hi - lo) + lo
        fits.append({"refit_date": str(r.date()), "n_train": len(y), "final_train_mse_scaled": final_loss,
                     "train_min": lo, "train_max": hi})  # fmt: skip
        for d, p in zip(block, pred, strict=True):
            rows.append({"date": d, "pred": float(p), "open_t": float(o.loc[d]), "open_next": float(target.loc[d]),
                         "scale_lo": lo, "scale_hi": hi})  # fmt: skip
    frame = pd.DataFrame(rows).set_index("date")
    frame["persistence"] = frame["open_t"]
    return V1Result(frame, fits)


def v1_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """RMSE (dollars and v1's scaled units), direction accuracy, and implied-return IC, v1 versus persistence."""
    from nvquant.evaluation.forecast import diebold_mariano, information_coefficient

    f = frame.dropna(subset=["open_next"])
    e_model = f["open_next"] - f["pred"]
    e_pers = f["open_next"] - f["persistence"]
    width = f["scale_hi"] - f["scale_lo"]
    actual_up = f["open_next"] > f["open_t"]
    pred_up = f["pred"] > f["open_t"]
    with np.errstate(invalid="ignore", divide="ignore"):  # blown-up predictions can be <= 0
        implied = np.log(f["pred"].to_numpy() / f["open_t"].to_numpy())
    realized = np.log(f["open_next"].to_numpy() / f["open_t"].to_numpy())
    dm = diebold_mariano(e_model.to_numpy(), e_pers.to_numpy())
    return {
        "n": float(len(f)),
        "rmse_model": float(np.sqrt(np.mean(e_model**2))),
        "rmse_persistence": float(np.sqrt(np.mean(e_pers**2))),
        "rmse_model_scaled": float(np.sqrt(np.mean((e_model / width) ** 2))),
        "rmse_persistence_scaled": float(np.sqrt(np.mean((e_pers / width) ** 2))),
        "rmse_ratio": float(np.sqrt(np.mean(e_model**2)) / np.sqrt(np.mean(e_pers**2))),
        "direction_accuracy": float(np.mean(actual_up == pred_up)),
        "share_days_up": float(np.mean(actual_up)),
        "implied_return_ic": information_coefficient(implied, realized).ic,
        "dm_stat": dm.stat,
        "dm_pvalue": dm.pvalue,
        "share_test_above_train_max": float(np.mean(f["open_t"] > f["scale_hi"])),
        "median_abs_error_model": float(np.median(np.abs(e_model))),
        "median_abs_error_persistence": float(np.median(np.abs(e_pers))),
        "share_days_worse_than_persistence": float(np.mean(np.abs(e_model) > np.abs(e_pers))),
        "share_days_blown_up": float(np.mean(np.abs(f["pred"]) > 10 * f["scale_hi"])),
        "max_abs_prediction": float(np.max(np.abs(f["pred"]))),
        "max_open": float(f["open_next"].max()),
    }
