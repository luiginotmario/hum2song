"""YAML configs for stage-A training and retrieval eval."""

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

DEFAULT_DATA_ROOT = "/lambda/nfs/hum2song-data"


@dataclass
class TrainConfig:
    """Every stage-A hyperparameter. CLI flags override the YAML file."""

    stage: str = "A"
    encoder: str = "mert"
    mert_name: str = "m-a-p/MERT-v1-95M"
    tiny_layers: int = 2
    tiny_hidden: int = 32
    sample_rate: int = 24000
    crop_seconds: float = 10.0
    query_min_seconds: float = 3.0
    query_max_seconds: float = 12.0
    song_jitter_s: float = 1.0
    batch_size: int = 32
    grad_accum_steps: int = 1
    steps: int = 20000
    warmup_steps: int = 500
    lr_head: float = 1.0e-3
    lr_backbone: float = 1.0e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    precision: str = "bf16"
    freeze_steps: int = 1000
    unfreeze_top_layers: int = 6
    temperature_init: float = 0.07
    qtype_loss_weight: float = 0.1
    conf_loss_weight: float = 0.0
    conf_gamma: float = 2.0
    proj_dim: int = 256
    proj_hidden: int = 512
    seed: int = 0
    num_workers: int = 4
    log_every: int = 20
    save_every: int = 500
    manifest: str | None = None
    data_root: str | None = None
    ckpt_dir: str | None = None
    run_name: str = "stage_a"
    augment: bool = True
    aug_pitch: bool = True
    aug_time: bool = True
    aug_noise: bool = True
    aug_gain: bool = True
    aug_eq: bool = True
    aug_rir: bool = True
    aug_codec: bool = False
    pitch_semitones: float = 4.0
    time_stretch_min: float = 0.8
    time_stretch_max: float = 1.25
    snr_min_db: float = 3.0
    snr_max_db: float = 30.0
    gain_db: float = 6.0
    limit: int | None = None
    resume: str | None = None
    dry_run: bool = False

    def resolved_manifest(self) -> Path:
        """pairs_real.jsonl under the data root when manifest is omitted."""
        if self.manifest:
            return Path(self.manifest)
        return self.resolved_data_root() / "pairs_real.jsonl"

    def resolved_data_root(self) -> Path:
        """$H2S_DATA, then /lambda/nfs/hum2song-data, unless data_root is set."""
        if self.data_root:
            return Path(self.data_root)
        if self.manifest:
            return Path(self.manifest).parent
        return Path(os.environ.get("H2S_DATA", DEFAULT_DATA_ROOT))

    def resolved_ckpt_dir(self) -> Path:
        """Checkpoint directory on the persistent disk."""
        if self.ckpt_dir:
            return Path(self.ckpt_dir)
        return self.resolved_data_root() / "ckpt" / self.run_name


@dataclass
class EvalConfig:
    """Retrieval eval over one or more manifest groups."""

    manifest: str | None = None
    data_root: str | None = None
    ckpt: str | None = None
    sets: str = "mirqbsh"
    distractors: int = 0
    out: str | None = None
    batch_size: int = 16
    crop_seconds: float = 10.0
    sample_rate: int = 24000
    songs: str | None = None
    dry_run: bool = False

    def resolved_manifest(self) -> Path:
        if self.manifest:
            return Path(self.manifest)
        return Path(os.environ.get("H2S_DATA", DEFAULT_DATA_ROOT)) / "pairs_real.jsonl"

    def resolved_data_root(self) -> Path:
        if self.data_root:
            return Path(self.data_root)
        return self.resolved_manifest().parent

    def resolved_out(self) -> Path:
        if self.out:
            return Path(self.out)
        return self.resolved_data_root() / "runs" / "eval" / "report.json"

    def resolved_songs(self) -> Path:
        if self.songs:
            return Path(self.songs)
        return self.resolved_data_root() / "songs.jsonl"

    def set_names(self) -> list[str]:
        if self.sets.strip() == "all":
            return ["all"]
        return [part.strip() for part in self.sets.split(",") if part.strip()]


def train_config_from_mapping(payload: dict) -> TrainConfig:
    """Rebuild a train config from a checkpoint dict, ignoring unknown keys."""
    known = {item.name for item in fields(TrainConfig)}
    filtered = {key: value for key, value in payload.items() if key in known}
    return TrainConfig(**filtered)


def load_train_config(path: Path, overrides: dict | None = None) -> TrainConfig:
    """Load train YAML and apply CLI overrides that are not None."""
    return _load(TrainConfig, path, overrides)


def load_eval_config(path: Path | None, overrides: dict | None = None) -> EvalConfig:
    """Load eval YAML. A missing path yields defaults plus overrides."""
    return _load(EvalConfig, path, overrides)


def dump_config(config: TrainConfig | EvalConfig, path: Path) -> None:
    """Write the resolved config next to a run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(asdict(config), sort_keys=True), encoding="utf-8")


def _load(cls, path: Path | None, overrides: dict | None):
    payload: dict = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must be a YAML mapping")
        payload.update(loaded)
    for key, value in (overrides or {}).items():
        if value is not None:
            payload[key] = value
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    return cls(**payload)
