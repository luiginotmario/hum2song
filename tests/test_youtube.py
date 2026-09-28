import importlib.util
from pathlib import Path

from hum2song.catalog.youtube import (
    SOURCE_TIER,
    batch_rows,
    db_song,
    is_target,
    read_batch_meta,
)

EVAL_SCRIPT = Path(__file__).resolve().parents[1] / "training" / "scripts" / "eval_previews.py"
TARGETS = [
    {
        "target_id": "chad:a",
        "youtube_id": "hLQl3WQQoQ0",
        "artist": "adele",
        "title": "someone like you",
    },
    {"target_id": "mtgqbh:wave", "artist": "Antonio Carlos Jobim", "title": "Wave"},
]


def write_batch(tmp_path: Path, lines: list[str], files: list[str]) -> Path:
    (tmp_path / "audio").mkdir()
    for name in files:
        (tmp_path / "audio" / name).write_bytes(b"x")
    (tmp_path / "batch_meta.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


def load_eval():
    spec = importlib.util.spec_from_file_location("eval_previews", EVAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_read_batch_meta_skips_malformed(tmp_path):
    path = tmp_path / "m.tsv"
    path.write_text("bb1\tabc\t200\tSong\nbroken line\n", encoding="utf-8")
    assert read_batch_meta(path) == [
        {"tag": "bb1", "video_id": "abc", "duration_s": "200", "video_title": "Song"}
    ]


def test_is_target_by_video_song_or_video_title():
    assert is_target(TARGETS, "hLQl3WQQoQ0", {}, "anything")
    assert is_target(TARGETS, "x", {"artist": "Adele", "title": "Someone Like You"}, "")
    assert is_target(TARGETS, "x", {}, "Adele - Someone Like You (Live)")
    assert not is_target(TARGETS, "x", {"artist": "Adele", "title": "Hello"}, "Adele - Hello")


def test_batch_rows_drops_missing_duplicate_and_target_like(tmp_path):
    batch = write_batch(
        tmp_path,
        [
            "bb1\tv1\t200\tDua Lipa - Levitating",
            "bb2\tv2\t200\tNo audio song",
            "bb3\tv1\t200\tDua Lipa - Levitating again",
            "bb4\tv4\t200\tAdele - Someone Like You",
        ],
        ["bb1.webm", "bb3.webm", "bb4.m4a"],
    )
    meta = {"bb1": {"artist": "Dua Lipa", "title": "Levitating", "genre": "pop_chart"}}
    out = batch_rows(batch, meta, TARGETS, set())
    assert [r["song_id"] for r in out["rows"]] == ["youtube:v1"]
    assert out["rows"][0]["role"] == "distractor" and out["rows"][0]["genre"] == "pop_chart"
    assert out["dropped"] == {"no_audio": 1, "duplicate": 1, "target_like": 1}


def test_db_song_is_tier_y_full():
    song = db_song({"song_id": "youtube:v1", "title": "T", "artist": "A", "genre": ""}, 180.0)
    assert song["source_tier"] == SOURCE_TIER == "Y"
    assert song["coverage"] == "full" and song["genre"] is None


def test_eval_full_song_distractor_settings():
    module = load_eval()
    full = {"role": "distractor", "source": "youtube_full", "target_id": None}
    preview = {"role": "distractor", "source": "deezer_preview", "target_id": None}
    fma = {"role": "fma", "source": "fma_full", "target_id": None}
    assert module.keep_song(full, ("youtube_full",), "full_fma", "chad")
    assert not module.keep_song(preview, ("youtube_full",), "full_fma", "chad")
    assert not module.keep_song(full, ("youtube_full",), "charts_fma", "chad")
    assert module.keep_song(preview, ("youtube_full",), "all_fma", "chad")
    assert module.keep_song(fma, ("youtube_full",), "full_fma", "chad")
