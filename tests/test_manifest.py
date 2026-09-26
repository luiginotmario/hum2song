"""The JSONL manifest round-trips and rejects a bad query type."""

import json
from pathlib import Path

import pytest

from hum2song.manifest import PairRecord, read_pairs, write_pairs


def test_pair_roundtrip(tmp_path: Path) -> None:
    record = PairRecord(
        pair_id="mirqbsh:year2003:person00001:00014",
        query_path="queries/mirqbsh/mirqbsh_year2003_person00001_00014.wav",
        qtype="hum",
        qsource="real",
        song_id="mirqbsh:00014",
        song_start_s=0.0,
        song_dur_s=8.0,
        split="test",
        group="mirqbsh",
        song_path="catalog/mirqbsh/00014.wav",
        title="Twinkle, Twinkle, Little Star",
    )
    path = tmp_path / "pairs_real.jsonl"
    write_pairs(path, [record])
    loaded = read_pairs(path)
    assert loaded == [record]
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["query_type"] == "hum"
    assert payload["qtype"] == "hum"
    assert loaded[0].query_type == "hum"


def test_query_type_alone_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps(
            {
                "pair_id": "mtgqbh:1",
                "query_path": "queries/mtgqbh/1.wav",
                "query_type": "sing",
                "qsource": "real",
                "song_id": "mtgqbh:song",
                "song_start_s": None,
                "song_dur_s": None,
                "split": "test",
                "group": "mtgqbh",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = read_pairs(path)
    assert loaded[0].query_type == "sing"
    assert loaded[0].qtype == "sing"


def test_disagreeing_query_type_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps(
            {
                "pair_id": "x",
                "query_path": "q.wav",
                "query_type": "sing",
                "qtype": "hum",
                "qsource": "real",
                "song_id": "s",
                "split": "train",
                "group": "humtrans",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        read_pairs(path)


def test_bad_qtype_is_rejected() -> None:
    record = PairRecord(
        pair_id="x",
        query_path="q.wav",
        qtype="speech",
        qsource="real",
        song_id="s",
        song_start_s=None,
        song_dur_s=None,
        split="train",
        group="humtrans",
    )
    with pytest.raises(ValueError):
        write_pairs(Path("/tmp/unused-pairs.jsonl"), [record])
