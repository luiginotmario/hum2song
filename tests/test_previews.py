import numpy as np
import torch

from hum2song.catalog.previews import (
    RateLimiter,
    best_match,
    clean_video_title,
    deezer_row,
    itunes_row,
    match_score,
    plain,
    song_key,
    split_video_title,
)
from hum2song.catalog.real_eval import (
    covered_targets,
    pick_track,
    positive_negative,
    song_scores,
    target_ranks,
)
from hum2song.catalog.targets import MLEND_TARGETS, mlend_targets, parse_interval, read_csv_rows


def row(title: str, artist: str, preview: str | None = "u") -> dict:
    return {"title": title, "artist": artist, "preview_url": preview}


def test_plain_strips_accents_brackets_and_featuring():
    assert plain("Beyoncé (Live) feat. Jay-Z") == "beyonce"
    assert plain("Simon & Garfunkel") == "simon and garfunkel"


def test_video_title_parsing():
    parsed = split_video_title("Queen – Bohemian Rhapsody [Official Video Remastered]")
    assert parsed == {"artist": "queen", "title": "bohemian rhapsody"}
    assert split_video_title("Some Song (Official Audio)") == {"artist": "", "title": "some song"}
    assert clean_video_title("Adele - Hello (Official Music Video) HD") == "adele hello"


def test_match_score_needs_the_artist():
    target = {"title": "Let It Go", "artist": "Idina Menzel"}
    assert match_score(target, row("Let It Go (From Frozen)", "Idina Menzel")) >= 0.9
    assert match_score(target, row("Let It Go", "Kids Karaoke Band")) < 0.8


def test_best_match_prefers_real_artist_and_skips_missing_previews():
    target = {"title": "Hakuna Matata", "artist": "Nathan Lane"}
    rows = [
        row("Hakuna Matata", "Nathan Lane, Ernie Sabella", preview=None),
        row("hakuna matata", "Gunna"),
    ]
    assert best_match(rows, target) is None
    rows.append(row("Hakuna Matata", "Nathan Lane, Ernie Sabella"))
    assert best_match(rows, target)["artist"] == "Nathan Lane, Ernie Sabella"


def test_best_match_title_only_target_uses_title_threshold():
    target = {"title": "some song", "artist": ""}
    assert best_match([row("Some Song", "X")], target)["match_score"] == 1.0
    assert best_match([row("Another Tune", "X")], target) is None


def test_service_rows():
    itunes = itunes_row(
        {
            "trackId": 5,
            "trackName": "T",
            "artistName": "A",
            "previewUrl": "p",
            "trackTimeMillis": 200000,
        }
    )
    assert itunes["song_id"] == "itunes:5" and itunes["full_duration_s"] == 200.0
    deezer = deezer_row(
        {
            "id": 7,
            "title": "T",
            "artist": {"name": "A"},
            "preview": "",
            "isrc": "X1",
            "duration": 180,
        }
    )
    assert deezer["song_id"] == "deezer:7" and deezer["preview_url"] is None
    assert song_key("The Beatles", "Help!") == "the beatles|help"


def test_rate_limiter_spaces_calls():
    limiter = RateLimiter(0.05)
    limiter.wait()
    first = limiter.last
    limiter.wait()
    assert limiter.last - first >= 0.049


def test_mlend_targets_and_interval():
    targets = mlend_targets()
    assert len(targets) == len(MLEND_TARGETS) == 8
    assert {
        "target_id": "mlend:Frozen",
        "dataset": "mlend",
        "title": "Let It Go",
        "artist": "Idina Menzel",
        "fragment_start_s": 126,
    } in targets
    assert parse_interval("(0.4, 13.11)") == (0.4, 13.11)


def test_read_csv_rows_handles_cr_line_endings(tmp_path):
    path = tmp_path / "q.csv"
    path.write_bytes(b"Filename,Title\rq1.wav, Help\rq2.wav, Yesterday\r")
    assert read_csv_rows(path) == [
        {"Filename": "q1.wav", "Title": "Help"},
        {"Filename": "q2.wav", "Title": "Yesterday"},
    ]


def test_song_scores_best_chunk_per_song():
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    chunks = torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
    owner = torch.tensor([0, 0, 1])
    scores = song_scores(queries, chunks, owner, 2)
    np.testing.assert_allclose(scores, [[1.0, 0.0], [0.8, 1.0]])


def test_target_ranks_any_version_and_missing_target():
    scores = np.array([[0.9, 0.5, 0.7], [0.1, 0.2, 0.3]])
    songs = ["a", None, "a"]
    assert target_ranks(scores, songs, ["a", "b"]) == [1, None]
    assert target_ranks(np.array([[0.2, 0.5, 0.1]]), songs, ["a"]) == [2]


def test_positive_negative_and_coverage():
    scores = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.3]])
    positives, negatives = positive_negative(scores, ["a", "b"], ["a", "a", "b"])
    np.testing.assert_allclose(positives, [0.9, 0.8, 0.3])
    np.testing.assert_allclose(negatives, [0.1, 0.2, 0.1])
    coverage = covered_targets(positives, negatives, ["a", "a", "b"])
    assert coverage["covered"] == {"a": True, "b": True}


def test_pick_track_falls_back_to_mix():
    silent = np.zeros((2, 3000), dtype=np.float32)
    sung = np.stack([np.full(3000, 220.0), np.ones(3000)]).astype(np.float32)
    kind, chunks = pick_track({"vocals": silent, "mix": sung}, "vocals_or_mix")
    assert kind == "mix" and len(chunks) > 0
    assert pick_track({"vocals": sung, "mix": silent}, "vocals_or_mix")[0] == "vocals"
    assert pick_track({"vocals": silent, "mix": sung}, "vocals")[1] == []
