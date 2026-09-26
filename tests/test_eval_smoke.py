"""Eval writes MIR-QBSH top-10 next to the CHAD 0.921 baseline."""

from pathlib import Path

from hum2song.config import EvalConfig
from hum2song.eval.run_eval import evaluate_pairs
from hum2song.manifest import PairRecord
from hum2song.model import RetrievalModel, TinyEncoder
from tests.support import write_tone


def test_report_places_our_top10_next_to_chad(tmp_path: Path) -> None:
    root = tmp_path / "data"
    pairs = []
    for index, song in enumerate(("00014", "00020")):
        query = f"queries/{song}.wav"
        song_path = f"catalog/{song}.wav"
        write_tone(root / query, 200.0 + index * 80)
        write_tone(root / song_path, 200.0 + index * 80)
        pairs.append(
            PairRecord(
                pair_id=f"mirqbsh:{song}",
                query_path=query,
                qtype="hum",
                qsource="real",
                song_id=f"mirqbsh:{song}",
                song_start_s=0.0,
                song_dur_s=0.3,
                split="test",
                group="mirqbsh",
                song_path=song_path,
                title=song,
            )
        )
    encoder = TinyEncoder(n_layers=2, hidden=32)
    model = RetrievalModel(encoder, hidden_dim=32, n_layers=2, proj_hidden=32, proj_dim=32)
    config = EvalConfig(
        manifest=str(root / "pairs_real.jsonl"),
        data_root=str(root),
        sets="mirqbsh",
        distractors=0,
        batch_size=2,
        crop_seconds=0.25,
        sample_rate=24000,
    )
    report = evaluate_pairs(model, pairs, config)
    result = report["results"][0]
    assert result["set"] == "mirqbsh"
    assert result["chad_top10"] == 0.921
    assert result["baselines"]["chad_top10_jang_midi"] == 0.921
    assert result["metrics"]["top10"] is not None
    assert 0.0 <= result["metrics"]["top10"] <= 1.0
    assert result["metrics"]["mrr"] is not None
    assert result["per_query_type"]["hum"]["count"] == 2
    assert result["per_query_type"]["whistle"]["count"] == 0
    assert result["per_query_type"]["sing"]["count"] == 0
    assert result["per_qtype"] == result["per_query_type"]
    assert "melody" in report["matching"]
    assert result["comparable_to_chad"] is False
    assert report["baselines"]["chad_top10_jang_midi"] == 0.921
