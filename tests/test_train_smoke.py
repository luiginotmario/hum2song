"""CPU training loop on a handful of fake clips, including resume."""

from pathlib import Path

import pytest
import torch

from hum2song.config import TrainConfig
from hum2song.manifest import PairRecord, write_pairs
from hum2song.model import RetrievalModel, TinyEncoder
from hum2song.train_loop import run_training
from tests.support import write_tone


def _config(root: Path, manifest: Path, steps: int, resume: str | None = None) -> TrainConfig:
    return TrainConfig(
        stage="A",
        encoder="tiny",
        sample_rate=24000,
        crop_seconds=0.25,
        query_min_seconds=0.25,
        query_max_seconds=0.25,
        song_jitter_s=0.0,
        batch_size=2,
        grad_accum_steps=1,
        steps=steps,
        warmup_steps=0,
        precision="fp32",
        freeze_steps=0,
        unfreeze_top_layers=1,
        proj_dim=32,
        proj_hidden=32,
        num_workers=0,
        log_every=1,
        save_every=1,
        manifest=str(manifest),
        data_root=str(root),
        ckpt_dir=str(root / "ckpt"),
        run_name="smoke",
        augment=False,
        seed=0,
        resume=resume,
        dry_run=False,
    )


def _model() -> RetrievalModel:
    encoder = TinyEncoder(n_layers=2, hidden=32)
    return RetrievalModel(encoder, hidden_dim=32, n_layers=2, proj_hidden=32, proj_dim=32)


def _manifest(root: Path) -> Path:
    records = []
    for index, song in enumerate(("alpha", "alpha", "beta", "beta")):
        query = f"queries/{song}_{index}.wav"
        song_path = f"catalog/{song}.wav"
        write_tone(root / query, 180.0 + index * 40)
        write_tone(root / song_path, 180.0 if song == "alpha" else 360.0)
        records.append(
            PairRecord(
                pair_id=f"fake:{index}",
                query_path=query,
                qtype="hum",
                qsource="real",
                song_id=f"fake:{song}",
                song_start_s=0.0,
                song_dur_s=0.3,
                split="train",
                group="humtrans",
                song_path=song_path,
                title=song,
            )
        )
    path = root / "pairs_real.jsonl"
    write_pairs(path, records)
    return path


def test_training_loop_learns_a_step_and_resumes(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    model = _model()
    last = run_training(_config(tmp_path, manifest, steps=1), model)
    payload = torch.load(last, map_location="cpu", weights_only=False)
    assert payload["step"] == 1
    assert payload["manifest_sha256"]
    assert "loss" not in payload
    resumed = run_training(_config(tmp_path, manifest, steps=2, resume=str(last)), model)
    payload = torch.load(resumed, map_location="cpu", weights_only=False)
    assert payload["step"] == 2
    assert torch.isfinite(torch.tensor(payload["step"]))


def test_stage_b_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    config = _config(tmp_path, manifest, steps=1)
    config.stage = "B"
    with pytest.raises(NotImplementedError):
        run_training(config, _model())
