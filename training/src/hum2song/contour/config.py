"""YAML config for contour-branch training (configs/train_contour.yaml)."""

import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from hum2song.config import DEFAULT_DATA_ROOT


@dataclass
class ContourConfig:
    """Every contour-branch hyperparameter. CLI overrides win over the YAML file."""

    run_name: str = "contour"
    data_root: str | None = None
    ckpt_dir: str | None = None
    steps: int = 6000
    batch_size: int = 256
    lr: float = 3.0e-4
    weight_decay: float = 0.05
    warmup_steps: int = 300
    grad_clip: float = 1.0
    dim: int = 256
    layers: int = 6
    heads: int = 4
    dropout: float = 0.1
    out_dim: int = 256
    temperature_init: float = 0.07
    num_workers: int = 16
    seed: int = 0
    log_every: int = 50
    val_every: int = 500
    select_metric: str = "val/humtrans_val_shift+7_top1"
    stretch_min: float = 0.6
    stretch_max: float = 1.7
    warp_depth: float = 0.2
    interval_scale: float = 0.15
    drift_semitones: float = 0.7
    jitter_semitones: float = 0.15
    dropout_prob: float = 0.5
    octave_error_prob: float = 0.1
    query_min_s: float = 3.0
    query_max_s: float = 12.0
    ref_start_jitter_s: float = 1.0
    ref_extra_min_s: float = -1.0
    ref_extra_max_s: float = 3.0
    shift_semitones: float = 7.0
    synthetic_midi_dirs: str = ""
    holdout_midi_dir: str = "raw/essen_midi/deutschl"
    holdout_count: int = 2000
    synthetic_exclude: str = ""
    whistle_aug_prob: float = 0.0
    whistle_compress_min: float = 0.55
    whistle_compress_max: float = 0.85
    whistle_gap_prob: float = 0.9
    whistle_gap_max_s: float = 0.6
    whistle_fold_prob: float = 0.0
    whistle_fold_range_st: float = 6.0
    whistle_synth: bool = False
    whistle_synth_songs: bool = True
    mlend_pairs: bool = False
    mlend_repeat: int = 2
    mlend_val: bool = False
    song_holdout: bool = False
    chad_pairs: bool = False
    chad_repeat: int = 1
    chad_val: bool = False
    # Fine-tuning (D-021): start from this checkpoint (relative to the data root) and stop
    # after `early_stop_patience` validations without a better val/select (0 = never).
    init_ckpt: str = ""
    early_stop_patience: int = 0
    # E2a (D-026): self-supervised song-window pairs with the CLEWS loss, added to the
    # InfoNCE loss of the other pairs with weight `song_loss_weight`.
    song_pairs: bool = False
    song_libraries: str = "youtube_charts_v1"
    song_fma_parity: str = "even"
    song_batch_size: int = 64
    song_refs: int = 4
    song_offset_s: float = 3.0
    song_mix_prob: float = 0.5
    song_loss_weight: float = 1.0
    clews_gamma: float = 5.0
    clews_eps: float = 1.0e-6
    # E2b (D-028): "covers" swaps the self-supervised song windows for CHAD cover -> original
    # pairs (contour/cover_pairs.py); loss and batch settings are the song_* ones above.
    song_source: str = "windows"
    # E3 (D-030): "salience" feeds soft/real RMVPE salience (±18 st crop + voicing)
    # instead of the hard-F0 (pitch, voicing) features.
    input_kind: str = "contour"

    def resolved_data_root(self) -> Path:
        return Path(self.data_root or os.environ.get("H2S_DATA", DEFAULT_DATA_ROOT))

    def resolved_ckpt_dir(self) -> Path:
        if self.ckpt_dir:
            return Path(self.ckpt_dir)
        return self.resolved_data_root() / "ckpt" / self.run_name


def load_contour_config(path: Path | None, overrides: dict | None = None) -> ContourConfig:
    """YAML mapping plus non-None overrides. Unknown keys raise ValueError."""
    payload: dict = {}
    if path is not None:
        payload.update(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    payload.update({key: value for key, value in (overrides or {}).items() if value is not None})
    known = {item.name for item in fields(ContourConfig)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    return ContourConfig(**payload)
