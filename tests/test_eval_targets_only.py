"""Eval always reports the targets-only line and never uses HumTrans as distractors."""

import json
from pathlib import Path

from hum2song.config import EvalConfig
from hum2song.eval.run_eval import _distractors, evaluate_pairs
from hum2song.manifest import PairRecord
from hum2song.model import RetrievalModel, TinyEncoder
from tests.support import write_tone


def _pairs(root: Path) -> list[PairRecord]:
    pairs = []
    for index, song in enumerate(("00014", "00020")):
        write_tone(root / f"queries/{song}.wav", 200.0 + index * 80)
        write_tone(root / f"catalog/{song}.wav", 200.0 + index * 80)
        pairs.append(
            PairRecord(
                pair_id=f"mirqbsh:{song}",
                query_path=f"queries/{song}.wav",
                qtype="hum",
                qsource="real",
                song_id=f"mirqbsh:{song}",
                song_start_s=0.0,
                song_dur_s=0.3,
                split="test",
                group="mirqbsh",
                song_path=f"catalog/{song}.wav",
                title=song,
            )
        )
    return pairs


def _songs(root: Path) -> Path:
    write_tone(root / "catalog/ht.wav", 500.0)
    write_tone(root / "catalog/other.wav", 600.0)
    rows = [
        {"song_id": "humtrans:0001", "source": "humtrans", "audio_path": "catalog/ht.wav"},
        {"song_id": "fma:0001", "source": "fma", "audio_path": "catalog/other.wav"},
    ]
    path = root / "songs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _config(root: Path, distractors: int) -> EvalConfig:
    return EvalConfig(
        data_root=str(root),
        songs=str(root / "songs.jsonl"),
        sets="mirqbsh",
        distractors=distractors,
        num_workers=0,
        batch_size=2,
        crop_seconds=0.25,
    )


def _model() -> RetrievalModel:
    encoder = TinyEncoder(n_layers=2, hidden=32)
    return RetrievalModel(encoder, hidden_dim=32, n_layers=2, proj_hidden=32, proj_dim=32)


def test_humtrans_songs_are_never_distractors(tmp_path: Path) -> None:
    _songs(tmp_path)
    extras = _distractors(_config(tmp_path, 10), {"mirqbsh:00014"})
    assert extras == [("fma:0001", "catalog/other.wav")]


def test_targets_only_is_the_headline_without_distractors(tmp_path: Path) -> None:
    pairs = _pairs(tmp_path)
    result = evaluate_pairs(_model(), pairs, _config(tmp_path, 0))["results"][0]
    assert result["headline"] == "targets_only"
    assert result["targets_only"]["count"] == 2
    assert result["targets_only"] == result["metrics"]
    assert "not chunked" in result["targets_only_note"]


def test_targets_only_ignores_distractors(tmp_path: Path) -> None:
    pairs = _pairs(tmp_path)
    _songs(tmp_path)
    result = evaluate_pairs(_model(), pairs, _config(tmp_path, 10))["results"][0]
    assert result["distractor_count"] == 1
    assert result["headline"] == "metrics"
    assert result["targets_only"]["count"] == 2
    assert result["targets_only"]["top1"] >= result["metrics"]["top1"]
