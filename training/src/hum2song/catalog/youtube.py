"""Full-length YouTube songs as library rows (D-019, D-020).

Audio is fetched on Luigi's Mac with yt-dlp in batches, copied to Lambda, decoded once by
extract_previews.py and deleted; only melody tracks and metadata are kept (Tier Y: not
published, not redistributed). A batch arrives as audio/<tag>.<ext> plus batch_meta.tsv
lines "tag, video id, duration, video title"; tag metadata says which chart listed it.
A distractor that could be one of the evaluation targets is dropped, so a target is never
competing with a second copy of itself.
"""

from pathlib import Path

from hum2song.catalog.previews import artist_score, plain, title_score

SOURCE = "youtube_full"
SOURCE_TIER = "Y"
SAME_TITLE = 0.9
SAME_ARTIST = 0.5
INSTRUMENTAL_GENRES = ("dance", "house_electronic")


def read_batch_meta(path: Path) -> list[dict]:
    """batch_meta.tsv -> rows; malformed lines are skipped."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 4 and parts[1]:
            rows.append(
                {
                    "tag": parts[0],
                    "video_id": parts[1],
                    "duration_s": parts[2],
                    "video_title": parts[3],
                }
            )
    return rows


def audio_file(audio_dir: Path, tag: str) -> Path | None:
    found = sorted(audio_dir.glob(f"{tag}.*"))
    return found[0] if found else None


def same_song(target: dict, artist: str, title: str) -> bool:
    return (
        title_score(target.get("title", ""), title) >= SAME_TITLE
        and artist_score(target.get("artist", ""), artist) >= SAME_ARTIST
    )


def title_in_video(target: dict, video_title: str) -> bool:
    words, video = plain(target.get("title", "")), plain(video_title)
    artist = plain(target.get("artist", ""))
    return bool(words) and bool(artist) and words in video and artist.split()[0] in video


def is_target(targets: list[dict], video_id: str, meta: dict, video_title: str) -> bool:
    """Could this chart song be an evaluation target (same video or same song)?"""
    return any(
        t.get("youtube_id") == video_id
        or same_song(t, meta.get("artist", ""), meta.get("title", ""))
        or title_in_video(t, video_title)
        for t in targets
    )


def distractor_row(item: dict, meta: dict, path: Path) -> dict:
    return {
        "song_id": f"youtube:{item['video_id']}",
        "source": SOURCE,
        "role": "distractor",
        "target_id": None,
        "title": meta.get("title") or item["video_title"],
        "artist": meta.get("artist", ""),
        "video_title": item["video_title"],
        "youtube_id": item["video_id"],
        "chart": meta.get("src", ""),
        "genre": meta.get("genre", ""),
        "audio_path": str(path),
    }


def drop_reason(item, meta, path, targets, known) -> str | None:
    if path is None:
        return "no_audio"
    if item["video_id"] in known:
        return "duplicate"
    if is_target(targets, item["video_id"], meta, item["video_title"]):
        return "target_like"
    return None


def batch_rows(batch_dir: Path, tag_meta: dict, targets: list[dict], known: set[str]) -> dict:
    """New distractor rows of one batch, plus counts of what was dropped and why."""
    rows, dropped = [], {"no_audio": 0, "duplicate": 0, "target_like": 0}
    for item in read_batch_meta(batch_dir / "batch_meta.tsv"):
        meta = tag_meta.get(item["tag"], {})
        path = audio_file(batch_dir / "audio", item["tag"])
        reason = drop_reason(item, meta, path, targets, known)
        if reason:
            dropped[reason] += 1
            continue
        known.add(item["video_id"])
        rows.append(distractor_row(item, meta, path))
    return {"rows": rows, "dropped": dropped}


def db_song(row: dict, duration_s: float | None) -> dict:
    """songs-table fields for pgvector (catalog/db.py SONG_COLUMNS)."""
    return {
        "song_id": row["song_id"],
        "source": SOURCE,
        "source_tier": SOURCE_TIER,
        "coverage": "full",
        "title": row.get("title"),
        "artist": row.get("artist"),
        "genre": row.get("genre") or None,
        "license": "youtube (features only, D-019)",
        "duration_s": duration_s,
    }
