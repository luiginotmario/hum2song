import importlib.util
from pathlib import Path

import numpy as np
import torch

from hum2song.contour.trackers import (
    RMVPE_HOP,
    RMVPE_SAMPLE_RATE,
    rmvpe_track,
    segment_bounds,
    segmented_salience,
)

SCRIPT = Path(__file__).resolve().parents[1] / "training" / "scripts" / "extract_previews.py"
BINS = 360


class FakeRmvpe:
    """Salience of frame i depends only on the sample it is centred on, like a local model."""

    def extract_mel(self, audio: torch.Tensor, center: bool = True) -> torch.Tensor:
        frames = len(audio) // RMVPE_HOP + 1
        index = torch.clamp(torch.arange(frames) * RMVPE_HOP, max=len(audio) - 1)
        return audio[index][None, None, :]

    def mel2hidden(self, mel: torch.Tensor) -> torch.Tensor:
        return mel[0, 0][None, :, None].repeat(1, 1, BINS)


class BrokenModels(dict):
    def __getitem__(self, key):
        raise RuntimeError("cuDNN error")


def load_script():
    spec = importlib.util.spec_from_file_location("extract_previews", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_segment_bounds_cover_audio():
    assert segment_bounds(10, 4) == [(0, 4), (4, 8), (8, 10)]


def test_segmented_salience_matches_one_pass():
    audio = np.random.default_rng(0).random(RMVPE_SAMPLE_RATE * 7 + 123).astype(np.float32)
    whole = segmented_salience(FakeRmvpe(), audio, max_s=100)
    parts = segmented_salience(FakeRmvpe(), audio, max_s=2)
    assert parts.shape == whole.shape == (len(audio) // RMVPE_HOP + 1, BINS)
    np.testing.assert_allclose(parts, whole)


def test_rmvpe_track_layout_with_segments():
    audio = np.zeros(RMVPE_SAMPLE_RATE * 3, dtype=np.float32)
    assert rmvpe_track(FakeRmvpe(), audio, max_s=1).shape == (2, len(audio) // RMVPE_HOP + 1)


def test_save_tracks_records_model_failure(tmp_path):
    module = load_script()
    item = {"row": {"song_id": "youtube:abc", "role": "target"}, "status": "ok"}
    item["audio"] = np.zeros(44100 * 6, dtype=np.float32)
    saved = module.save_tracks(item, BrokenModels(), torch.device("cpu"), tmp_path)
    assert saved["status"] == "extract_failed" and "cuDNN" in saved["error"]
    assert not list(tmp_path.iterdir())
