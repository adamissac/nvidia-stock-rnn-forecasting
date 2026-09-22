"""Small sequence networks with two output heads.

Heads:

- ``gaussian``: outputs mean and log-variance, trained with the Gaussian
  negative log-likelihood (a heteroscedastic model: it learns when it is less
  sure).
- ``quantile``: outputs the 10th, 50th, and 90th percentiles, trained with the
  pinball loss. The three outputs are built as a median plus cumulative
  softplus offsets so the quantiles can't cross.
"""

from __future__ import annotations

import math

import torch
from torch import nn

QUANTILES = (0.1, 0.5, 0.9)


class Head(nn.Module):
    """Maps a hidden vector to distribution parameters."""

    def __init__(self, hidden: int, kind: str) -> None:
        super().__init__()
        if kind not in ("gaussian", "quantile"):
            raise ValueError(f"unknown head {kind}")
        self.kind = kind
        self.out = nn.Linear(hidden, 2 if kind == "gaussian" else 3)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Gaussian: (mu, log_var). Quantile: (q10, q50, q90), monotone by construction."""
        z = self.out(h)
        if self.kind == "gaussian":
            return torch.stack([z[:, 0], z[:, 1].clamp(-10, 6)], dim=1)
        mid = z[:, 1]
        lo = mid - nn.functional.softplus(z[:, 0])
        hi = mid + nn.functional.softplus(z[:, 2])
        return torch.stack([lo, mid, hi], dim=1)


def head_loss(kind: str, out: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted Gaussian NLL or pinball loss."""
    if kind == "gaussian":
        mu, log_var = out[:, 0], out[:, 1]
        nll = 0.5 * (log_var + (y - mu) ** 2 / log_var.exp() + math.log(2 * math.pi))
        return (w * nll).sum() / w.sum()
    losses = []
    for i, q in enumerate(QUANTILES):
        e = y - out[:, i]
        losses.append(torch.maximum(q * e, (q - 1) * e))
    return (w * torch.stack(losses, dim=1).sum(dim=1)).sum() / w.sum()


class RNNNet(nn.Module):
    """One-layer LSTM or GRU over the feature sequence; the last hidden state feeds the head."""

    def __init__(
        self, n_features: int, hidden: int, cell: str, head: str, dropout: float = 0.1
    ) -> None:
        super().__init__()
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = Head(hidden, head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, features)."""
        out, _ = self.rnn(x)
        return self.head(self.drop(out[:, -1]))


class _CausalConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class TCNNet(nn.Module):
    """Temporal convolutional network (Bai, Kolter, Koltun 2018): dilated causal convs with residuals."""

    def __init__(
        self,
        n_features: int,
        channels: int,
        levels: int,
        kernel: int,
        head: str,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        c_in = n_features
        for i in range(levels):
            layers.append(_TCNBlock(c_in, channels, kernel, 2**i, dropout))
            c_in = channels
        self.blocks = nn.Sequential(*layers)
        self.head = Head(channels, head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, features)."""
        h = self.blocks(x.transpose(1, 2))
        return self.head(h[:, :, -1])


class _TCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = _CausalConv(c_in, c_out, kernel, dilation)
        self.conv2 = _CausalConv(c_out, c_out, kernel, dilation)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(torch.relu(self.conv1(x)))
        h = self.drop(torch.relu(self.conv2(h)))
        return torch.relu(h + self.skip(x))


class PatchTSTNet(nn.Module):
    """A small PatchTST-style encoder (Nie et al. 2023).

    The sequence is cut into overlapping patches over time; each patch (all
    features in it, flattened) is linearly embedded, a learned positional
    embedding is added, and a Transformer encoder mixes patches. The mean of
    the encoded patches feeds the head. Unlike the paper this isn't
    channel-independent, which keeps it small for a single-asset problem.
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        patch_len: int,
        stride: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        head: str,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        n_patches = (seq_len - patch_len) // stride + 1
        self.embed = nn.Linear(patch_len * n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = Head(d_model, head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, features)."""
        patches = x.unfold(1, self.patch_len, self.stride)  # (b, n_patches, features, patch_len)
        patches = patches.flatten(2)
        h = self.encoder(self.embed(patches) + self.pos)
        return self.head(h.mean(dim=1))
