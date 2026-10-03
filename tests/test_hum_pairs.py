"""D-039 hum-to-song pairs: synthetic hum or sung query vs vocal melody, same checkpoint."""

from pathlib import Path

import numpy as np
import pytest
import torch

from hum2song.contour.augment import ContourAugment
from hum2song.contour.config import ContourConfig, load_contour_config
from hum2song.contour.hum_pairs import HUM_STYLE_PROB, HumSongDataset, hum_query, sung_query
from hum2song.contour.song_pairs import collate_song_windows
from hum2song.contour.train import (
    build_model,
    hum_song_dataset,
    load_checkpoint,
    save_checkpoint,
    song_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
FLAT = np.full(400, 64.0, np.float32)


def quiet_augment(**overrides) -> ContourAugment:
    fields = {
        "stretch_min": 1.0,
        "stretch_max": 1.0,
        "warp_depth": 0.0,
        "interval_scale": 0.0,
        "drift_semitones": 0.0,
        "jitter_semitones": 0.0,
        "dropout_prob": 0.0,
        "octave_error_prob": 0.0,
    }
    return ContourAugment(**{**fields, **overrides})


def melody_route() -> dict:
    t = np.arange(3000) * 0.02
    contour = (60 + 5 * np.sin(t)).astype(np.float32)
    return {"a": {"ref": contour, "alt": contour + 1}, "b": {"ref": contour + 3, "alt": None}}


def test_hum_style_humanizes_and_sung_style_does_not(monkeypatch):
    seen = {"calls": 0}

    def spy(contour, rng, frame_s, **kwargs):
        seen["calls"] += 1
        return contour

    monkeypatch.setattr("hum2song.contour.hum_pairs.humanize", spy)
    hummed = hum_query(FLAT, np.random.default_rng(0), quiet_augment())
    assert seen["calls"] == 1 and len(hummed) == len(FLAT)
    sung_query(FLAT, np.random.default_rng(0), quiet_augment())
    assert seen["calls"] == 1


def test_off_pitch_drift_remains_after_median_centering():
    query = sung_query(FLAT, np.random.default_rng(0), quiet_augment(drift_semitones=2.0))
    voiced = query[~np.isnan(query)]
    assert voiced.max() - voiced.min() > 0.2


def test_off_tempo_stretch_changes_query_length():
    spec = quiet_augment(stretch_min=0.5, stretch_max=0.5)
    query = sung_query(FLAT, np.random.default_rng(0), spec)
    assert len(query) == 200


def test_dataset_collate_is_a_clews_batch():
    data = HumSongDataset(melody_route(), ContourAugment(), seed=0, refs=3, hum_prob=HUM_STYLE_PROB)
    batch = collate_song_windows([data[0], data[1]])
    assert batch["query"].shape[0] == 2 and batch["query"].shape[-1] == 2
    assert batch["refs"].shape[0] == 6 and batch["refs_per_song"] == 3
    assert batch["song_id"] == ["a", "b"]
    assert torch.isfinite(batch["query"]).all()


def test_dataset_survives_a_nearly_unvoiced_melody():
    ref = np.full(3000, np.nan, dtype=np.float32)
    ref[::400] = 60.0
    data = HumSongDataset({"a": {"ref": ref, "alt": None}}, ContourAugment(), seed=0, hum_prob=1.0)
    for epoch in range(3):
        data.epoch.set(epoch)
        assert data[0]["query"].shape[0] > 0 and len(data[0]["refs"]) == 4


def test_hum_config_is_a_hum_finetune_of_e2b():
    config = load_contour_config(ROOT / "configs" / "train_contour_hum.yaml")
    assert config.run_name == "contour_hum_s0"
    assert config.song_source == "hums" and config.song_pairs and config.song_fma_parity == "all"
    assert config.init_ckpt == "ckpt/contour_e2b_s0/best.pt"
    assert config.input_kind == "contour"
    assert config.chad_pairs and config.chad_val
    assert config.whistle_aug_prob == 0.0 and not config.whistle_synth
    assert not config.mlend_pairs and not config.mlend_val
    assert config.select_metric == "val/chad_val_top1"


def test_unknown_song_source_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown song_source"):
        song_dataset(tmp_path, ContourConfig(song_source="coversongs"))


def test_hum_source_uses_the_hum_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr("hum2song.contour.train.training_songs", lambda *_args: [])
    monkeypatch.setattr(
        "hum2song.contour.train.load_song_routes",
        lambda _rows: melody_route(),
    )
    data = hum_song_dataset(tmp_path, ContourConfig(song_source="hums", song_refs=2))
    assert isinstance(data, HumSongDataset) and len(data) == 2
    assert song_dataset(tmp_path, ContourConfig(song_source="hums")).ids == data.ids


def test_hum_checkpoint_loads_with_the_live_loader(tmp_path):
    config = ContourConfig(
        run_name="contour_hum_s0",
        song_source="hums",
        data_root=str(tmp_path),
        dim=32,
        layers=1,
        heads=2,
        out_dim=16,
    )
    model = build_model(config)
    path = tmp_path / "ckpt" / "contour_hum_s0" / "best.pt"
    save_checkpoint(path, model, config, 0, {"val/chad_val_top1": 0.0})
    loaded, loaded_config, step = load_checkpoint(path, torch.device("cpu"))
    assert step == 0 and loaded_config.song_source == "hums"
    for left, right in zip(model.state_dict().values(), loaded.state_dict().values(), strict=True):
        assert torch.equal(left, right.cpu())
