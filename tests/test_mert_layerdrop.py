"""MERT layer hooks stay complete after the encoder is unfrozen and training."""

import torch
from torch import nn

from hum2song.mert import MERT_LAYERS, MertEncoder
from hum2song.model import RetrievalModel


class _DropConfig:
    """Stand-in for the published MERT config, which sets layerdrop to 0.05."""

    def __init__(self) -> None:
        self.layerdrop = 0.05
        self.mask_time_prob = 0.05
        self.mask_feature_prob = 0.0


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.bias


class _DroppingEncoder(nn.Module):
    """Skip a layer in train mode the way HubertEncoder does."""

    def __init__(self, config: _DropConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(_Layer() for _index in range(MERT_LAYERS))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            skip = self.training and float(torch.rand(())) < self.config.layerdrop
            if skip:
                continue
            hidden = layer(hidden)
        return hidden


class _FakeMert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _DropConfig()
        self.encoder = _DroppingEncoder(self.config)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(input_values.shape[0], 4, 8)
        return self.encoder(hidden)


def test_unfrozen_encoder_captures_every_layer_in_train_mode() -> None:
    fake = _FakeMert()
    encoder = MertEncoder(fake)
    assert fake.config.layerdrop == 0.0
    assert fake.config.mask_time_prob == 0.0
    model = RetrievalModel(encoder, hidden_dim=8, n_layers=MERT_LAYERS, proj_hidden=8, proj_dim=4)
    model.set_trainable_top(6)
    model.train()
    assert model.encoder.training
    wave = torch.zeros(2, 320)
    for _step in range(30):
        states = model.encoder(wave)
        assert len(states) == MERT_LAYERS
    output = model(wave)
    assert output.embedding.shape == (2, 4)
