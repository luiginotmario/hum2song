"""Contour encoder: conv front end, transformer, masked attention pool, 256-d embedding.

One tower embeds both hummed contours (RMVPE) and reference contours (MIDI), since both
are the same kind of input after key normalization.
"""

import math

import torch
from torch import nn

from hum2song.contour.features import FEATURE_DIM

MAX_POSITIONS = 4096


def sinusoidal_positions(length: int, dim: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32)[:, None]
    rate = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    table = torch.zeros(length, dim)
    table[:, 0::2] = torch.sin(position * rate)
    table[:, 1::2] = torch.cos(position * rate)
    return table


class ConvFront(nn.Module):
    """Two 1-D convolutions; the second halves the frame rate. Padded frames stay zero."""

    def __init__(self, in_dim: int, dim: int) -> None:
        super().__init__()
        self.first = nn.Conv1d(in_dim, dim // 2, kernel_size=5, padding=2)
        self.second = nn.Conv1d(dim // 2, dim, kernel_size=5, stride=2, padding=2)
        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        mask = valid[:, None, :].to(features.dtype)
        hidden = self.activation(self.first(features.transpose(1, 2))) * mask
        return self.activation(self.second(hidden)).transpose(1, 2)


class MaskedAttnPool(nn.Module):
    """Attention pool over time that ignores padded frames."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        logits = self.score(hidden).squeeze(-1).masked_fill(~valid, -1.0e4)
        weights = torch.softmax(logits.float(), dim=-1).to(hidden.dtype)
        return torch.einsum("bt,btd->bd", weights, hidden)


class ContourEncoder(nn.Module):
    """(batch, frames, FEATURE_DIM) contour features + validity mask -> L2-normalized embedding."""

    def __init__(
        self,
        dim: int = 256,
        layers: int = 6,
        heads: int = 4,
        dropout: float = 0.1,
        out_dim: int = 256,
        temperature_init: float = 0.07,
        in_dim: int = FEATURE_DIM,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.front = ConvFront(in_dim, dim)
        self.register_buffer(
            "positions", sinusoidal_positions(MAX_POSITIONS, dim), persistent=False
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(dim)
        self.pool = MaskedAttnPool(dim)
        self.proj = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, out_dim))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature_init)))

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        hidden = self.front(features, valid)
        token_valid = valid[:, ::2][:, : hidden.shape[1]]
        hidden = hidden + self.positions[: hidden.shape[1]]
        hidden = self.transformer(hidden, src_key_padding_mask=~token_valid)
        pooled = self.pool(self.norm(hidden), token_valid)
        return nn.functional.normalize(self.proj(pooled), dim=-1)

    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=1.0e-3, max=1.0)
