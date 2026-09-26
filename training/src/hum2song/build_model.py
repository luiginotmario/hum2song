"""Construct the retrieval model named by a train config."""

from hum2song.config import TrainConfig
from hum2song.mert import MERT_HIDDEN, MERT_LAYERS, load_mert_encoder
from hum2song.model import RetrievalModel, TinyEncoder


def build_model(config: TrainConfig) -> RetrievalModel:
    """Build a tiny tower or MERT-v1-95M. Unknown encoder names raise ValueError."""
    builders = {"tiny": _build_tiny, "mert": _build_mert}
    builder = builders.get(config.encoder)
    if builder is None:
        known = ", ".join(sorted(builders))
        raise ValueError(f"unknown encoder {config.encoder}. known: {known}")
    return builder(config)


def _build_tiny(config: TrainConfig) -> RetrievalModel:
    encoder = TinyEncoder(n_layers=config.tiny_layers, hidden=config.tiny_hidden)
    return RetrievalModel(
        encoder,
        hidden_dim=config.tiny_hidden,
        n_layers=config.tiny_layers,
        proj_hidden=config.proj_hidden,
        proj_dim=config.proj_dim,
        temperature_init=config.temperature_init,
    )


def _build_mert(config: TrainConfig) -> RetrievalModel:
    encoder = load_mert_encoder(config.mert_name)
    return RetrievalModel(
        encoder,
        hidden_dim=MERT_HIDDEN,
        n_layers=MERT_LAYERS,
        proj_hidden=config.proj_hidden,
        proj_dim=config.proj_dim,
        temperature_init=config.temperature_init,
    )
