"""E2b (D-028): CHAD cover -> original melody pairs for CLEWS-style training.

CHAD (Amatov et al., ISMIR 2023) lists, per song group, the original YouTube recording and
cover versions, with aligned fragment intervals (same `fragment_id` = same passage). The
original and the best-correlated cover of each training group were downloaded (full audio,
deleted after extraction) and cached as melody tracks in library/chad_covers_v1.

One item = one (group, cover):
  query  a 3-12 s window of the cover's melody inside one aligned fragment (vocal stem, or
         with probability `mix_prob` the full-mix track), humanized and augmented like a hum
  refs   `refs` windows of the original's melody track around the aligned position (the
         fragment offset mapped linearly), each a little shifted, longer or shorter, stretched
The CLEWS loss then takes the best reference window as the positive, so fragment alignment
errors of a few seconds are tolerated.
"""

import ast
import csv
import re
import unicodedata
import wave
from pathlib import Path

import numpy as np

from hum2song.catalog.fma import read_jsonl
from hum2song.catalog.real_eval import pick_track
from hum2song.catalog.real_pool import library_rows, load_tracks
from hum2song.contour.augment import ContourAugment, augment_contour, humanize
from hum2song.contour.data import features_tensor, seconds_to_frames
from hum2song.contour.features import FRAME_S, rmvpe_contour, trim_unvoiced
from hum2song.contour.song_pairs import MIN_QUERY_S, SongWindowDataset

COVER_LIBRARY = "chad_covers_v1"
CHAD_CSV = "raw/chad/repo/metadata/dataset.csv"
METHOD = "vocals_or_mix"
MIN_TITLE_LETTERS = 5
TARGET_LIBRARY = "previews_v1"
QUERY_TRIES = 8


def tokens(text: str | None) -> list[str]:
    """Lower-case ASCII words; accents dropped and apostrophes joined ("Can't" -> "cant")."""
    plain = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    return re.findall(r"[a-z0-9]+", re.sub(r"['’`]", "", plain.lower()))


def contains_run(words: list[str], run: list[str]) -> bool:
    size = len(run)
    return size > 0 and any(words[i : i + size] == run for i in range(len(words) - size + 1))


def eval_targets(root: Path) -> list[tuple[list[str], list[str]]]:
    """(title words, artist words) of every evaluation target (CHAD, MTG-QBH, MLEnd)."""
    rows = read_jsonl(root / "library" / TARGET_LIBRARY / "targets.jsonl")
    targets = [(tokens(r.get("title")), tokens(r.get("artist"))) for r in rows]
    return [target for target in targets if target[0]]


def target_match(words: list[str], target: tuple[list[str], list[str]]) -> bool:
    """Title words appear in order; short titles ("One", "Help") also need the artist."""
    title, artist = target
    specific = len("".join(title)) >= MIN_TITLE_LETTERS
    return contains_run(words, title) and (specific or contains_run(words, artist))


def title_blocked(video_title: str | None, targets: list) -> bool:
    """True when the video title names an evaluation target's song (YouTube titles carry
    artist, title and extras in any order)."""
    words = tokens(video_title)
    return any(target_match(words, target) for target in targets)


def read_fragments(path: Path) -> dict[tuple[str, str], dict[str, tuple[float, float]]]:
    """(group, youtube id) -> {fragment_id: (start_s, end_s)} for originals and covers."""
    fragments: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["audio_type"] == "humming":
                continue
            key = (row["group_id"], row["youtube_id"])
            fragments.setdefault(key, {})[row["fragment_id"]] = tuple(
                float(v) for v in ast.literal_eval(row["interval"])
            )
    return fragments


def fragment_pairs(original: dict, cover: dict) -> list[tuple[tuple, tuple]]:
    """Aligned ((orig start, end), (cover start, end)) of the fragments both versions have."""
    return [(original[f], cover[f]) for f in sorted(set(original) & set(cover))]


def reference_contour(path: str) -> np.ndarray | None:
    tracks = load_tracks(path)
    if tracks is None:
        return None
    kind, chunks = pick_track(tracks, METHOD)
    return rmvpe_contour(tracks[kind]) if chunks else None


def cover_routes(path: str) -> dict | None:
    """{"ref": vocal contour, "alt": full-mix contour} of a cover recording."""
    tracks = load_tracks(path)
    if tracks is None:
        return None
    return {"ref": rmvpe_contour(tracks["vocals"]), "alt": rmvpe_contour(tracks["mix"])}


def library_versions(root: Path) -> dict[str, dict]:
    """group -> {"original": row, "covers": [rows]} of extracted, unblocked recordings."""
    blocked = eval_targets(root)
    versions: dict[str, dict] = {}
    for row in library_rows(root, [COVER_LIBRARY]):
        if not Path(row["_track"]).exists() or title_blocked(row.get("video_title"), blocked):
            continue
        entry = versions.setdefault(row["group"], {"original": None, "covers": []})
        slot = "original" if row["version"] == "original" else "covers"
        entry[slot] = row if slot == "original" else [*entry["covers"], row]
    return {g: v for g, v in versions.items() if v["original"] and v["covers"]}


def cover_items(root: Path) -> list[dict]:
    """One training item per (group, cover) with at least one aligned fragment."""
    fragments = read_fragments(root / CHAD_CSV)
    items = []
    for group, entry in sorted(library_versions(root).items()):
        original = entry["original"]
        reference = reference_contour(original["_track"])
        if reference is None:
            continue
        for cover in entry["covers"]:
            pairs = fragment_pairs(
                fragments.get((group, original["youtube_id"]), {}),
                fragments.get((group, cover["youtube_id"]), {}),
            )
            routes = cover_routes(cover["_track"]) if pairs else None
            if routes is None:
                continue
            items.append({"group": group, "ref": reference, "cover": routes, "pairs": pairs})
    return items


def mapped_start(start_s: float, original: tuple, cover: tuple) -> tuple[float, float]:
    """Original-side start (s) and time ratio for a cover-side start inside a fragment."""
    ratio = (original[1] - original[0]) / max(cover[1] - cover[0], 1e-6)
    return original[0] + (start_s - cover[0]) * ratio, ratio


class CoverPairDataset(SongWindowDataset):
    """Cover query window + original reference windows around the aligned position."""

    def __init__(self, items: list[dict], augment: ContourAugment, seed: int, **kwargs) -> None:
        super().__init__({}, augment, seed, **kwargs)
        self.items = items
        self.ids = [item["group"] for item in items]

    def cover_window(self, item: dict, rng: np.random.Generator):
        """(query crop, original start frame, original length frames)."""
        for _attempt in range(QUERY_TRIES):
            original, cover = item["pairs"][int(rng.integers(len(item["pairs"])))]
            length_s = float(rng.uniform(*self.query_s))
            latest = max(cover[1] - length_s, cover[0])
            start_s = float(rng.uniform(cover[0], latest))
            source = self.query_source(item["cover"], rng)
            first = seconds_to_frames(start_s)
            crop = source[first : first + seconds_to_frames(length_s)]
            ref_start, ratio = mapped_start(start_s, original, cover)
            placed = (seconds_to_frames(max(ref_start, 0.0)), seconds_to_frames(length_s * ratio))
            if len(trim_unvoiced(crop)) >= seconds_to_frames(MIN_QUERY_S):
                return trim_unvoiced(crop), *placed
        return (crop if len(crop) else source[:1]), *placed

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng([self.seed, self.epoch.get(), index, 3])
        item = self.items[index]
        query, start, length = self.cover_window(item, rng)
        query = augment_contour(humanize(query, rng, FRAME_S), rng, self.augment, FRAME_S)
        refs = [self.reference(item["ref"], start, length, rng) for _ in range(self.refs)]
        return {
            "query": features_tensor(query),
            "refs": [features_tensor(r) for r in refs],
            "song_id": item["group"],
        }


def tag_version(tag: str) -> tuple[str, str] | None:
    """Download tag "o_<group>" / "c1_<group>" / "c2_<group>" -> (version, group)."""
    prefix, _, group = tag.partition("_")
    versions = {"o": "original", "c1": "cover", "c2": "cover"}
    return (versions[prefix], group) if prefix in versions and group else None


def read_batch_meta(path: Path) -> dict[str, dict]:
    """tag -> {youtube_id, duration, title} of one batch's e2b_meta.tsv (last line wins)."""
    meta = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 4:
            meta[parts[0]] = {"youtube_id": parts[1], "duration": parts[2], "title": parts[3]}
    return meta


def batch_row(tag: str, audio: Path, meta: dict) -> dict | None:
    version = tag_version(tag)
    if version is None or tag not in meta:
        return None
    info = meta[tag]
    return {
        "song_id": f"youtube:{info['youtube_id']}",
        "source": "youtube_full",
        "role": "train",
        "group": version[1],
        "version": version[0],
        "youtube_id": info["youtube_id"],
        "video_title": info["title"],
        "audio_path": str(audio),
    }


def drop_reason(row: dict | None, known: set[str], blocked: list) -> str | None:
    if row is None:
        return "no_meta"
    if row["song_id"] in known:
        return "known"
    if title_blocked(row["video_title"], blocked):
        return "blocked"
    return None


def cover_batch_rows(batch_dir: Path, blocked: list, known: set[str]) -> dict:
    """Library rows for a downloaded E2b batch; blocked, known or unmatched files are deleted."""
    meta = read_batch_meta(batch_dir / "e2b_meta.tsv")
    rows, dropped = [], {"blocked": 0, "known": 0, "no_meta": 0}
    for audio in sorted((batch_dir / "audio").iterdir()):
        row = batch_row(audio.name.split(".")[0], audio, meta)
        reason = drop_reason(row, known, blocked)
        if reason:
            dropped[reason] += 1
            audio.unlink(missing_ok=True)
            continue
        known.add(row["song_id"])
        rows.append(row)
    return {"rows": rows, "dropped": dropped}


SECTION_FILE = re.compile(r"^(?P<tag>[^.]+)\.s(?P<start>\d+)\.\w+$")
STITCH_RATE = 44100


def section_files(audio_dir: Path) -> dict[str, list[tuple[float, Path]]]:
    """tag -> [(start_s, file)] of segment downloads named <tag>.s<start>.<ext>."""
    sections: dict[str, list[tuple[float, Path]]] = {}
    for path in sorted(audio_dir.iterdir()):
        match = SECTION_FILE.match(path.name)
        if match:
            sections.setdefault(match["tag"], []).append((float(match["start"]), path))
    return sections


def place_sections(parts: list[tuple[float, np.ndarray]], rate: int) -> np.ndarray:
    """One silent timeline with every segment at its start time (later segments win)."""
    end = max(int(start * rate) + len(audio) for start, audio in parts)
    timeline = np.zeros(end, dtype=np.float32)
    for start, audio in parts:
        first = int(start * rate)
        timeline[first : first + len(audio)] = audio
    return timeline


def write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes())


def stitch_sections(audio_dir: Path, decode, rate: int = STITCH_RATE) -> int:
    """Replace each tag's segment files by <tag>.wav with the segments at their CHAD times, so
    fragment intervals keep their absolute meaning; returns the number of stitched tags."""
    sections = section_files(audio_dir)
    for tag, files in sections.items():
        parts = [(start, decode(path, rate)) for start, path in files]
        write_wav(audio_dir / f"{tag}.wav", place_sections(parts, rate), rate)
        for _start, path in files:
            path.unlink()
    return len(sections)


def video_tags(list_path: Path) -> dict[str, str]:
    """youtube id -> download tag from a download list (tag, watch URL, ...)."""
    tags = {}
    for line in list_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and "v=" in parts[1]:
            tags[parts[1].split("v=", 1)[1]] = parts[0]
    return tags


def rename_id_files(batch_dir: Path, list_path: Path, prefix: str = "v3_") -> int:
    """Files named <prefix><youtube id>.<ext> (id-named batch downloads) -> <tag>.<ext>, and
    their meta lines (id, duration, title in <prefix>meta.tsv) appended to e2b_meta.tsv."""
    tags = video_tags(list_path)
    meta_path = batch_dir / f"{prefix}meta.tsv"
    lines = meta_path.read_text(encoding="utf-8").splitlines() if meta_path.exists() else []
    tagged = [
        f"{tags[line.split(chr(9))[0]]}\t{line}" for line in lines if line.split("\t")[0] in tags
    ]
    with open(batch_dir / "e2b_meta.tsv", "a", encoding="utf-8") as handle:
        handle.writelines(line + "\n" for line in tagged)
    renamed = 0
    for path in sorted((batch_dir / "audio").glob(f"{prefix}*")):
        video = path.name[len(prefix) :].rsplit(".", 1)[0]
        target = tags.get(video)
        if target:
            path.rename(path.with_name(f"{target}{path.suffix}"))
            renamed += 1
    return renamed
