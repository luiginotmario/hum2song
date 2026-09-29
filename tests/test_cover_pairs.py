import json

import numpy as np

from hum2song.contour.augment import ContourAugment
from hum2song.contour.cover_pairs import (
    CoverPairDataset,
    cover_batch_rows,
    fragment_pairs,
    mapped_start,
    read_fragments,
    tag_version,
    title_blocked,
    tokens,
)
from hum2song.contour.song_pairs import collate_song_windows

TARGETS = [(tokens("Someone Like You"), tokens("Adele")), (tokens("One"), tokens("U2"))]


def melody(seconds: float, offset: float) -> np.ndarray:
    t = np.arange(int(seconds / 0.02)) * 0.02
    return (60 + offset + 4 * np.sin(t)).astype(np.float32)


def test_tokens_and_title_blocking():
    assert tokens("Can't Stop – Mediterráneo") == ["cant", "stop", "mediterraneo"]
    assert title_blocked("Someone Like You - Adele (cover)", TARGETS)
    assert title_blocked("U2 - One (acoustic)", TARGETS)
    assert not title_blocked("One Direction - Night Changes", TARGETS)
    assert not title_blocked("Someone You Loved (piano)", TARGETS)


def test_fragments_and_mapping(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text(
        "group_id,fragment_id,id,audio_type,youtube_id,interval,correlation,"
        "check_by_crowdsource,is_available,duration\n"
        'g,1,a,original,O,"(10, 26)",1.0,False,True,16\n'
        'g,1,b,cover,C,"(20, 36)",0.8,False,True,16\n'
        'g,2,c,original,O,"(90, 105)",1.0,False,True,15\n',
        encoding="utf-8",
    )
    fragments = read_fragments(csv)
    pairs = fragment_pairs(fragments[("g", "O")], fragments[("g", "C")])
    assert pairs == [((10.0, 26.0), (20.0, 36.0))]
    start, ratio = mapped_start(24.0, *pairs[0])
    assert start == 14.0 and ratio == 1.0
    assert tag_version("c1_g") == ("cover", "g") and tag_version("x_g") is None


def test_cover_pair_items_and_collate():
    items = [
        {
            "group": group,
            "ref": melody(120, shift),
            "cover": {"ref": melody(120, shift + 2), "alt": melody(120, shift + 2)},
            "pairs": [((10.0, 26.0), (20.0, 36.0))],
        }
        for group, shift in (("a", 0), ("b", 5))
    ]
    data = CoverPairDataset(items, ContourAugment(), seed=0, refs=3)
    batch = collate_song_windows([data[0], data[1]])
    assert batch["query"].shape[0] == 2 and batch["refs"].shape[0] == 6
    assert batch["song_id"] == ["a", "b"]


def test_cover_pair_survives_short_cover():
    item = {
        "group": "a",
        "ref": melody(60, 0),
        "cover": {"ref": melody(5, 0), "alt": melody(5, 0)},
        "pairs": [((10.0, 26.0), (20.0, 36.0))],
    }
    assert CoverPairDataset([item], ContourAugment(), seed=0)[0]["query"].shape[0] > 0


def test_cover_batch_rows_drop_blocked_and_known(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    for tag in ("o_g1", "c1_g1", "c1_g2", "zz_g3"):
        (audio / f"{tag}.webm").write_bytes(b"x")
    (tmp_path / "e2b_meta.tsv").write_text(
        "o_g1\tO1\t200\tArtist - Song\nc1_g1\tC1\t180\tSong (cover)\n"
        "c1_g2\tC2\t190\tAdele - Someone Like You cover\n",
        encoding="utf-8",
    )
    batch = cover_batch_rows(tmp_path, TARGETS, {"youtube:C1"})
    assert [r["song_id"] for r in batch["rows"]] == ["youtube:O1"]
    assert batch["dropped"] == {"blocked": 1, "known": 1, "no_meta": 1}
    assert sorted(p.name for p in audio.iterdir()) == ["o_g1.webm"]
    assert json.loads(json.dumps(batch["rows"][0]))["version"] == "original"


def test_stitch_sections_places_segments(tmp_path):
    from hum2song.contour.cover_pairs import stitch_sections

    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "c1_g.s2.m4a").write_bytes(b"a")
    (audio / "c1_g.s5.m4a").write_bytes(b"b")
    (audio / "o_g.webm").write_bytes(b"c")

    def decode(path, rate):
        return np.full(rate, 0.5 if path.name.endswith("s2.m4a") else -0.5, dtype=np.float32)

    assert stitch_sections(audio, decode, rate=100) == 1
    assert sorted(p.name for p in audio.iterdir()) == ["c1_g.wav", "o_g.webm"]
    import wave

    with wave.open(str(audio / "c1_g.wav")) as handle:
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    assert len(samples) == 600 and samples[250] > 0 and samples[550] < 0 and samples[50] == 0


def test_rename_id_files(tmp_path):
    from hum2song.contour.cover_pairs import rename_id_files

    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "v3_AbC-1.m4a").write_bytes(b"x")
    (tmp_path / "v3_meta.tsv").write_text("AbC-1\t200\tSong\n", encoding="utf-8")
    lst = tmp_path / "list.tsv"
    lst.write_text("o_g\thttps://www.youtube.com/watch?v=AbC-1\t1-2\n", encoding="utf-8")
    assert rename_id_files(tmp_path, lst) == 1
    assert (tmp_path / "audio" / "o_g.m4a").exists()
    assert (tmp_path / "e2b_meta.tsv").read_text(encoding="utf-8") == "o_g\tAbC-1\t200\tSong\n"
