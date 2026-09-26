"""MERT-v1-95M encoder. Layer states are captured with forward hooks.

MERT's remote code on transformers 5.x returns hidden_states=None, so the
12 transformer blocks are hooked directly instead of reading that field.
"""

import torch
from torch import nn
from transformers import AutoModel

MERT_HIDDEN = 768
MERT_LAYERS = 12
MERT_NAME = "m-a-p/MERT-v1-95M"


class MertEncoder(nn.Module):
    """Wrap a Hugging Face MERT model and return one state per transformer layer."""

    def __init__(self, mert: nn.Module) -> None:
        super().__init__()
        self.mert = mert
        self._captured: list[torch.Tensor] = []
        layers = self.mert.encoder.layers
        if len(layers) != MERT_LAYERS:
            raise RuntimeError(f"expected {MERT_LAYERS} MERT layers, found {len(layers)}")
        for layer in layers:
            layer.register_forward_hook(self._hook)
        _disable_spec_augment(self.mert)

    def forward(self, wave: torch.Tensor) -> list[torch.Tensor]:
        self._captured = []
        self.mert(input_values=wave)
        if len(self._captured) != MERT_LAYERS:
            raise RuntimeError(f"hooks captured {len(self._captured)} layers")
        return list(self._captured)

    def set_trainable_top(self, n_top: int) -> None:
        """Freeze MERT, then unfreeze the top n transformer layers. The CNN stays frozen."""
        for parameter in self.mert.parameters():
            parameter.requires_grad = False
        if n_top <= 0:
            return
        for layer in self.mert.encoder.layers[-n_top:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    def _hook(self, _module: nn.Module, _inputs: tuple, output: object) -> None:
        self._captured.append(_layer_hidden(output))


def load_mert_encoder(model_name: str = MERT_NAME) -> MertEncoder:
    """Download m-a-p/MERT-v1-95M and wrap it. Requires trust_remote_code."""
    mert = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    return MertEncoder(mert)


def _disable_spec_augment(mert: nn.Module) -> None:
    config = mert.config
    if hasattr(config, "mask_time_prob"):
        config.mask_time_prob = 0.0
    if hasattr(config, "mask_feature_prob"):
        config.mask_feature_prob = 0.0


def _layer_hidden(output: object) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    hidden = getattr(output, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):
        return hidden
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError(f"unrecognized MERT layer output: {type(output)!r}")
