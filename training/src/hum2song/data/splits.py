"""Song-level train/val/test assignment with no song leakage."""

import hashlib
from dataclasses import dataclass

HOLDOUT_GROUPS = frozenset({"mirqbsh", "mlend", "mtgqbh"})
# A song in one of these groups leaves the holdout when it has a sing clip,
# so sung queries can train. Every clip of that song still shares one split.
SUNG_TRAIN_GROUPS = frozenset({"mirqbsh"})
SPLIT_SEED = 0
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
SPLIT_RANK = {"train": 0, "val": 1, "test": 2}
VALID_SPLITS = frozenset(SPLIT_RANK)


class SplitLeakageError(ValueError):
    """Raised when one song_id appears in more than one split."""


@dataclass(frozen=True)
class SplitRow:
    """Fields the splitter needs. One row per recording."""

    song_id: str
    group: str
    official_split: str | None
    title: str | None
    query_type: str = "hum"


def normalize_title(title: str | None) -> str:
    """Lowercase a title and drop punctuation so cross-dataset names can match."""
    if not title:
        return ""
    cleaned = title.strip().lower()
    if cleaned in {"-", "none", "unknown"}:
        return ""
    characters = [char if char.isalnum() else " " for char in cleaned]
    return " ".join("".join(characters).split())


def hash_split(
    song_id: str,
    seed: int = SPLIT_SEED,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
) -> str:
    """Map a song id onto train, val, or test. The same id always gets the same split."""
    digest = hashlib.sha256(f"{seed}:{song_id}".encode()).hexdigest()
    unit = int(digest[:8], 16) / float(16**8)
    if unit < train_fraction:
        return "train"
    if unit < train_fraction + val_fraction:
        return "val"
    return "test"


def propose_split(row: SplitRow, sung_song_ids: frozenset[str] = frozenset()) -> str:
    """Pick the row's split before song-level reconciliation."""
    held_out = row.group in HOLDOUT_GROUPS and row.song_id not in sung_song_ids
    if held_out:
        return "test"
    if row.official_split in VALID_SPLITS:
        return row.official_split
    return hash_split(row.song_id)


def assign_song_splits(rows: list[SplitRow]) -> dict[str, str]:
    """Return song_id -> split. Holdouts are test. Official splits win over the hash.

    A MIR-QBSH song with a sing clip is hash-split instead of held out, and every
    clip of that song shares the split. A train or val song whose normalized title
    matches a test song is moved to test as well.
    """
    sung_song_ids = _sung_song_ids(rows)
    proposed: dict[str, str] = {}
    titles: dict[str, str] = {}
    for row in rows:
        choice = propose_split(row, sung_song_ids)
        current = proposed.get(row.song_id)
        if current is None or SPLIT_RANK[choice] > SPLIT_RANK[current]:
            proposed[row.song_id] = choice
        title = normalize_title(row.title)
        if title and row.song_id not in titles:
            titles[row.song_id] = title
    test_titles = {
        titles[song_id]
        for song_id, split in proposed.items()
        if split == "test" and song_id in titles
    }
    test_titles.discard("")
    resolved: dict[str, str] = {}
    for song_id, split in proposed.items():
        title = titles.get(song_id, "")
        blocked = bool(title) and title in test_titles and split != "test"
        resolved[song_id] = "test" if blocked else split
    return resolved


def _sung_song_ids(rows: list[SplitRow]) -> frozenset[str]:
    return frozenset(
        row.song_id
        for row in rows
        if row.group in SUNG_TRAIN_GROUPS and row.query_type == "sing"
    )


def assert_no_song_leakage(song_splits: list[tuple[str, str]]) -> None:
    """Raise SplitLeakageError if any song_id is listed under two splits."""
    seen: dict[str, str] = {}
    for song_id, split in song_splits:
        previous = seen.get(song_id)
        if previous is None:
            seen[song_id] = split
            continue
        if previous != split:
            raise SplitLeakageError(f"{song_id} is in both {previous} and {split}")
