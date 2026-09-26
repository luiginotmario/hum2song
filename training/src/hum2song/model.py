"""Retrieval tower: layer mix, attentive pool, 256-d projection, query-type head."""

import math
from dataclasses import dataclass

import torch
from torch import nn

QTYPE_TO_INDEX = {"hum": 0, "whistle": 1, "sing": 2}
INDEX_TO_QTYPE = {index: name for name, index in QTYPE_TO_INDEX.items()}


@dataclass
class ModelOutput:
    """Embeddings are L2-normalized. qtype_logits cover hum, whistle, sing."""

    embedding: torch.Tensor
    qtype_logits: torch.Tensor
    temperature: torch.Tensor


class LayerMix(nn.Module):
    """Learned softmax weights over encoder layers."""

    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_layers))

    def forward(self, layers: list[torch.Tensor]) -> torch.Tensor:
        weights = torch.softmax(self.logits, dim=0)
        stacked = torch.stack(layers, dim=0)
        return torch.einsum("l,lbtd->btd", weights, stacked)


class AttnPool(nn.Module):
    """Single-query attention pool over time."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(hidden).squeeze(-1), dim=-1)
        return torch.einsum("bt,btd->bd", weights, hidden)


class ProjectionHead(nn.Module):
    """MLP projection followed by L2 normalization."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.net(pooled), dim=-1)


class TinyEncoder(nn.Module):
    """Small conv + transformer stack used by the CPU smoke test. Not MERT."""

    def __init__(self, n_layers: int = 2, hidden: int = 32, stride: int = 160) -> None:
        super().__init__()
        self.front = nn.Conv1d(1, hidden, kernel_size=stride, stride=stride)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=4,
                    dim_feedforward=hidden * 2,
                    dropout=0.0,
                    batch_first=True,
                )
                for _index in range(n_layers)
            ]
        )

    def forward(self, wave: torch.Tensor) -> list[torch.Tensor]:
        hidden = self.front(wave.unsqueeze(1)).transpose(1, 2)
        states: list[torch.Tensor] = []
        for layer in self.layers:
            hidden = layer(hidden)
            states.append(hidden)
        return states

    def set_trainable_top(self, n_top: int) -> None:
        """Freeze the stack, then unfreeze the last n_top transformer layers."""
        for parameter in self.parameters():
            parameter.requires_grad = False
        if n_top <= 0:
            return
        for layer in self.layers[-n_top:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True


class RetrievalModel(nn.Module):
    """Shared tower for query audio and song audio."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int,
        n_layers: int,
        proj_hidden: int = 512,
        proj_dim: int = 256,
        n_qtypes: int = 3,
        temperature_init: float = 0.07,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.layer_mix = LayerMix(n_layers)
        self.pool = AttnPool(hidden_dim)
        self.proj = ProjectionHead(hidden_dim, proj_hidden, proj_dim)
        self.qtype_head = nn.Linear(hidden_dim, n_qtypes)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature_init)))

    def forward(self, wave: torch.Tensor) -> ModelOutput:
        mixed = self.layer_mix(self.encoder(wave))
        pooled = self.pool(mixed)
        return ModelOutput(
            embedding=self.proj(pooled),
            qtype_logits=self.qtype_head(pooled),
            temperature=self.temperature(),
        )

    def temperature(self) -> torch.Tensor:
        """Positive temperature, initialized at 0.07."""
        return self.log_temperature.exp().clamp(min=1.0e-3, max=1.0)

    def set_trainable_top(self, n_top: int) -> None:
        """Freeze or partially unfreeze the encoder. The head always trains."""
        self.encoder.set_trainable_top(n_top)
        for module in (self.layer_mix, self.pool, self.proj, self.qtype_head):
            for parameter in module.parameters():
                parameter.requires_grad = True
        self.log_temperature.requires_grad = True

    def train(self, mode: bool = True):
        """Keep a frozen encoder in eval so its dropout does not move features."""
        super().train(mode)
        feature = getattr(getattr(self.encoder, "mert", None), "feature_extractor", None)
        if feature is not None:
            feature.eval()
        if not any(parameter.requires_grad for parameter in self.encoder.parameters()):
            self.encoder.eval()
        return self
