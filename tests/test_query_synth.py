"""Synthetic whistle and wrong-lyrics queries (D-040)."""

import inspect
import json
from pathlib import Path

import numpy as np
import torch

from hum2song.contour.augment import ContourAugment
from hum2song.contour.config import ContourConfig, load_contour_config
from hum2song.contour.model import ContourEncoder
from hum2song.contour.song_pairs import collate_song_windows
from hum2song.contour.train import song_dataset
from hum2song.query_synth.errors import drop_one_note, shift_one_interval
from hum2song.query_synth.gates import mlend_gate, mtg_gate
from hum2song.query_synth.generate import render_song, write_set
from hum2song.query_synth.manifest import require_synthetic, split_name, train_rows
from hum2song.query_synth.render import (
    ORIGIN_SYNTHETIC,
    PRESENTATION,
    STYLE_WHISTLE,
    STYLE_WRONG_LYRICS,
    render_query,
    shift_to_whistle,
)
from hum2song.query_synth.report import synthetic_section
from hum2song.query_synth.signal import semitone_to_hz
from hum2song.query_synth.singing import SYLLABLE_NAMES, singing_waveform
from hum2song.query_synth.smoke import run_smoke

ROOT = Path(__file__).resolve().parents[1]


def quiet() -> ContourAugment:
    return ContourAugment(
        stretch_min=1.0,
        stretch_max=1.0,
        warp_depth=0.0,
        interval_scale=0.0,
        drift_semitones=0.0,
        jitter_semitones=0.0,
        dropout_prob=0.0,
        octave_error_prob=0.0,
    )


def train_song() -> str:
    for number in range(200):
        song_id = f"fma:{number}"
        if split_name(song_id) == "train":
            return song_id
    raise AssertionError("no train bucket in the first 200 ids")


def tune() -> np.ndarray:
    return np.concatenate(
        [np.full(30, pitch, dtype=np.float32) for pitch in (60.0, 64.0, 67.0, 64.0)]
    )


def test_wrong_interval_and_dropped_note_edit_one_run():
    shifted = shift_one_interval(tune(), np.random.default_rng(0), probability=1.0)
    assert not np.allclose(shifted, tune(), equal_nan=True)
    dropped = drop_one_note(tune(), np.random.default_rng(1), probability=1.0)
    assert np.isnan(dropped).sum() >= 30


def test_whistle_register_is_higher_and_inside_the_band():
    flat = np.full(80, 60.0, dtype=np.float32)
    shifted = shift_to_whistle(flat, np.random.default_rng(0))
    median_hz = float(semitone_to_hz(np.median(shifted)))
    assert np.median(shifted) >= 72.0
    assert 500.0 <= median_hz <= 4000.0


def test_render_takes_no_lyric_text():
    assert "lyric" not in inspect.signature(render_query).parameters
    assert "lyric" not in inspect.signature(singing_waveform).parameters


def test_smoke_writes_two_files_and_labels_them_synthetic(tmp_path):
    report = run_smoke(tmp_path)
    assert (tmp_path / "whistle.wav").stat().st_size > 44
    assert (tmp_path / "wrong_lyrics.wav").stat().st_size > 44
    assert report["origin"] == ORIGIN_SYNTHETIC
    assert report["presentation"] == PRESENTATION
    assert report["whistle"]["probe_h2_db"] < -25.0
    assert report["wrong_lyrics"]["probe_h2_db"] > report["whistle"]["probe_h2_db"] + 10.0
    assert set(report["wrong_lyrics"]["syllables"]) <= set(SYLLABLE_NAMES)
    assert report["wrong_lyrics"]["syllables"]
    assert report["whistle"]["median_hz"] >= 500.0
    assert report["whistle"]["tracker_median_abs_st"] < 1.5
    assert report["wrong_lyrics"]["tracker_median_abs_st"] < 1.5


def test_generated_manifest_is_synthetic(tmp_path):
    contour = np.full(400, 62.0, dtype=np.float32)
    contour[100:140] = 65.0
    contour[200:240] = 69.0
    rows = render_song(_payload(tmp_path, train_song(), contour, seed=0))
    path = write_set(tmp_path, rows)
    loaded = train_rows(path)
    assert len(loaded) == 2
    assert {row["style"] for row in loaded} == {STYLE_WHISTLE, STYLE_WRONG_LYRICS}
    for row in loaded:
        require_synthetic(row)
        assert (tmp_path / row["wav"]).is_file()
        assert (tmp_path / row["contour"]).is_file()
        names = set(row["syllables"])
        assert names <= set(SYLLABLE_NAMES)


def test_dataset_collate(tmp_path):
    contour = np.tile(np.array([60.0, 62.0, 64.0, 67.0], dtype=np.float32), 80)
    write_set(tmp_path, render_song(_payload(tmp_path, train_song(), contour, seed=1)))
    config = ContourConfig(
        song_source="query_synth",
        song_pairs=True,
        song_refs=2,
        data_root=str(tmp_path),
        query_synth_manifest="query_synth/manifest.jsonl",
    )
    data = song_dataset(tmp_path, config)
    batch = collate_song_windows([data[0], data[1]])
    assert batch["refs_per_song"] == 2
    assert batch["query"].shape[-1] == 2
    assert torch.isfinite(batch["query"]).all()


def test_gates_block_without_owned_audio(tmp_path):
    mlend = mlend_gate(tmp_path)
    mtg = mtg_gate(tmp_path, ["mtgqbh:1"])
    assert mlend["status"] == "blocked" and "owned/mlend_targets" in mlend["reason"]
    assert "Potter" in mlend["reason"]
    assert mtg["status"] == "blocked" and "owned/mtgqbh" in mtg["reason"]
    assert "top1" not in mlend


def test_synthetic_report_cannot_be_read_as_real(tmp_path):
    song = np.sin(np.linspace(0, 8, 500)).astype(np.float32) + 64.0
    lines = []
    for split, song_id in (("train", "fma:2"), ("test", "fma:3")):
        safe = song_id.replace(":", "_")
        ref = tmp_path / "query_synth" / "refs" / f"{safe}.npy"
        qry = tmp_path / "query_synth" / "f0" / f"{safe}.npy"
        ref.parent.mkdir(parents=True, exist_ok=True)
        qry.parent.mkdir(parents=True, exist_ok=True)
        np.save(ref, song)
        np.save(qry, song[:200])
        lines.append(
            {
                "query_id": f"{song_id}:whistle:0",
                "song_id": song_id,
                "style": "whistle",
                "split": split,
                "origin": ORIGIN_SYNTHETIC,
                "presentation": PRESENTATION,
                "contour": qry.relative_to(tmp_path).as_posix(),
                "ref_contour": ref.relative_to(tmp_path).as_posix(),
            }
        )
    manifest = tmp_path / "query_synth" / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in lines), encoding="utf-8")
    model = ContourEncoder(dim=32, layers=1, heads=2, out_dim=16)
    report = synthetic_section(
        tmp_path, model, torch.device("cpu"), "query_synth/manifest.jsonl", 2, 0
    )
    assert report["origin"] == "synthetic"
    assert report["presentation"] == PRESENTATION
    assert report["status"] == "ok"
    assert isinstance(report["top1"], float)
    assert isinstance(report["top10"], float)
    assert isinstance(report["mrr"], float)


def test_qsynth_config_keeps_e2b_and_uses_the_contour_tower():
    config = load_contour_config(ROOT / "configs" / "train_contour_qsynth.yaml")
    assert config.run_name == "contour_qsynth_s0"
    assert config.init_ckpt == "ckpt/contour_e2b_s0/best.pt"
    assert config.song_source == "query_synth"
    assert config.hum_loss_weight == 0.15
    assert config.input_kind == "contour"
    assert config.whistle_synth is False
    assert ContourConfig().hum_loss_weight == 1.0


def _payload(root: Path, song_id: str, contour: np.ndarray, seed: int) -> dict:
    return {
        "song_id": song_id,
        "contour": contour,
        "root": str(root),
        "augment": quiet().__dict__,
        "query_min_s": 3.0,
        "query_max_s": 4.0,
        "per_song": 1,
        "seed": seed,
        "source": "vocal_f0",
        "sing_tracker": "cepstrum",
    }
