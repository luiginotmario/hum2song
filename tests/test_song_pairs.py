import numpy as np
import torch
import torch.nn.functional as F

from hum2song.contour.augment import ContourAugment
from hum2song.contour.song_pairs import (
    SongWindowDataset,
    collate_song_windows,
    keep_fma,
    title_key,
)
from hum2song.losses import clews_loss


def song(seconds: float, offset: float) -> np.ndarray:
    t = np.arange(int(seconds / 0.02)) * 0.02
    return (60 + offset + 4 * np.sin(t * (1 + offset / 10))).astype(np.float32)


def test_title_key_and_fma_parity():
    assert title_key("Die with a Smile (Official Audio)") == "diewithasmile"
    assert title_key("Hello - Remastered") == "hello"
    assert keep_fma("fma:000002", "even") and not keep_fma("fma:000003", "even")
    assert keep_fma("fma:000003", "odd") and keep_fma("fma:000003", "all")


def test_song_window_items_and_collate():
    routes = {
        "a": {"ref": song(60, 0), "alt": song(60, 0.5)},
        "b": {"ref": song(40, 3), "alt": None},
    }
    data = SongWindowDataset(routes, ContourAugment(), seed=0, refs=3)
    batch = collate_song_windows([data[0], data[1]])
    assert batch["query"].shape[0] == 2 and batch["refs"].shape[0] == 6
    assert batch["refs_per_song"] == 3 and batch["song_id"] == ["a", "b"]


def test_clews_loss_prefers_matching_segments():
    torch.manual_seed(0)
    query = F.normalize(torch.randn(4, 16), dim=-1)
    noise = F.normalize(torch.randn(4, 3, 16), dim=-1)
    matched = noise.clone()
    matched[:, 1] = query
    ids = ["a", "b", "c", "d"]
    assert clews_loss(query, matched, ids) < clews_loss(query, noise, ids)


def test_song_window_survives_nearly_unvoiced_song():
    ref = np.full(3000, np.nan, dtype=np.float32)
    ref[::400] = 60.0
    data = SongWindowDataset({"a": {"ref": ref, "alt": None}}, ContourAugment(), seed=0)
    for epoch in range(3):
        data.epoch.set(epoch)
        assert data[0]["query"].shape[0] > 0
