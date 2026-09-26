"""The JSONL manifest round-trips and rejects a bad query type."""

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
