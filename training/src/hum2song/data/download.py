"""Download hum datasets and write the shared pairs manifest."""

import argparse
import csv
import os
import shutil
from pathlib import Path

from hum2song.data.fetch import Checksum, download_file, extract_archive
from hum2song.data.sources import PARSERS, RawExample, parse_dataset
from hum2song.data.splits import SplitRow, assert_no_song_leakage, assign_song_splits
from hum2song.logutil import configure_logging, get_logger
from hum2song.manifest import PairRecord, write_jsonl, write_pairs
from hum2song.midi_render import render_midi_file

LOGGER = get_logger(__name__)

DEFAULT_DATA_ROOT = "/lambda/nfs/hum2song-data"
RENDER_SAMPLE_RATE = 24000
MLEND_RAW = "https://raw.githubusercontent.com/MLEndDatasets/HumsAndWhistles/main"
MLEND_CSV_NAME = "MLEndHWD_audio_attributes.csv"
MLEND_AUDIO_DIR = "MLEndHWD_audiofiles"
LICENSES = {
    "mirqbsh": "research-only",
    "humtrans": "cc-by-nc-4.0",
    "mtgqbh": "cc-by-4.0",
    "mlend": "research-only",
}


class RemoteFile:
    """One URL that belongs to a dataset."""

    def __init__(self, url: str, filename: str, algo: str, digest: str, archive: bool) -> None:
        self.url = url
        self.filename = filename
        self.algo = algo
        self.digest = digest
        self.archive = archive


DATASET_FILES: dict[str, tuple[RemoteFile, ...]] = {
    "mirqbsh": (
        RemoteFile(
            "https://music-ir.org/evaluation/MIREX/data/qbsh/MIR-QBSH-corpus.tar.gz",
            "MIR-QBSH-corpus.tar.gz",
            "sha256",
            "5eb978b61beac608bc2300a8bd32e1035229cffae68250d82bb41b94447ef181",
            True,
        ),
    ),
    "humtrans": (
        RemoteFile(
            "https://huggingface.co/datasets/dadinghh2/HumTrans/resolve/main/all_wav.zip",
            "all_wav.zip",
            "sha256",
            "ee2a9a4f24f8f988c55a17ee7fd55251be59d426d65b75ddd43a7571fcdab965",
            True,
        ),
        RemoteFile(
            "https://huggingface.co/datasets/dadinghh2/HumTrans/resolve/main/all_midi.zip",
            "all_midi.zip",
            "sha256",
            "70577e9cca108b315df0348f9195891517434ccdc521a47714aa58dd629f1f43",
            True,
        ),
        RemoteFile(
            "https://huggingface.co/datasets/dadinghh2/HumTrans/resolve/main/train_valid_test_keys.json",
            "train_valid_test_keys.json",
            "sha256",
            "dc3b3733fd5aeba16b935230036b9356c774d0ccb3fe12aa04ab91aa19855ab5",
            False,
        ),
    ),
    "mtgqbh": (
        RemoteFile(
            "https://zenodo.org/api/records/1290712/files/MTG-QBH.zip/content",
            "MTG-QBH.zip",
            "md5",
            "82a4e0a09832ebd8525e66c51d5111fc",
            True,
        ),
    ),
}


def run_download(argv: list[str] | None = None) -> None:
    """CLI entry used by training/scripts/download_datasets.py."""
    args = _parse_args(list(argv or []))
    configure_logging()
    names = _dataset_names(args.datasets)
    if args.dry_run:
        _log_plan(names)
        return
    root = Path(args.out)
    examples = _collect_examples(names, root)
    if not examples:
        LOGGER.warning("no examples were parsed; manifest was not written")
        return
    records, songs = build_pairs(examples, root, args.max_items)
    pairs_path = root / "pairs_real.jsonl"
    songs_path = root / "songs.jsonl"
    write_pairs(pairs_path, records)
    write_jsonl(songs_path, songs)
    LOGGER.info("wrote %s pairs to %s", len(records), pairs_path)
    LOGGER.info("split counts %s", _count_splits(records))


def build_pairs(
    examples: list[RawExample],
    data_root: Path,
    max_items: int | None,
    sample_rate: int = RENDER_SAMPLE_RATE,
) -> tuple[list[PairRecord], list[dict]]:
    """Link queries, render MIDI references, and assign song-level splits."""
    split_of = assign_song_splits(
        [
            SplitRow(item.song_id, item.group, item.official_split, item.title, item.qtype)
            for item in examples
        ]
    )
    tagged = [(item, split_of[item.song_id]) for item in examples]
    chosen = _take_subset(tagged, max_items)
    rendered: dict[Path, float] = {}
    records = [
        _materialize_one(item, split, data_root, sample_rate, rendered) for item, split in chosen
    ]
    records = sorted(records, key=_record_pair_id)
    assert_no_song_leakage([(record.song_id, record.split) for record in records])
    songs = _song_rows(chosen, records)
    return records, songs


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download hum datasets and write pairs_real.jsonl")
    parser.add_argument(
        "--datasets",
        default="mirqbsh,humtrans,mtgqbh,mlend",
        help="Comma-separated subset of mirqbsh,humtrans,mtgqbh,mlend",
    )
    parser.add_argument(
        "--out",
        default=os.environ.get("H2S_DATA", DEFAULT_DATA_ROOT),
        help="Data root. Defaults to $H2S_DATA or /lambda/nfs/hum2song-data",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print URLs and exit")
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Keep a deterministic subset of rows after parsing. Archives are still complete.",
    )
    return parser.parse_args(argv)


def _dataset_names(text: str) -> list[str]:
    names = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [name for name in names if name not in PARSERS]
    if unknown:
        known = ", ".join(sorted(PARSERS))
        raise SystemExit(f"unknown datasets: {', '.join(unknown)}. known: {known}")
    return names


def _log_plan(names: list[str]) -> None:
    for name in names:
        if name == "mlend":
            LOGGER.info("dry-run mlend %s/%s", MLEND_RAW, MLEND_CSV_NAME)
            LOGGER.info("dry-run mlend audio %s/%s/<file>.wav", MLEND_RAW, MLEND_AUDIO_DIR)
            continue
        for remote in DATASET_FILES[name]:
            LOGGER.info("dry-run %s %s -> raw/%s/%s", name, remote.url, name, remote.filename)


def _collect_examples(names: list[str], root: Path) -> list[RawExample]:
    examples: list[RawExample] = []
    for name in names:
        roots = fetch_dataset(name, root)
        if roots is None:
            LOGGER.warning("skipped %s", name)
            continue
        parsed = parse_dataset(name, roots)
        LOGGER.info("parsed %s rows from %s", len(parsed), name)
        examples.extend(parsed)
    return examples


def fetch_dataset(name: str, root: Path) -> list[Path] | None:
    """Download and extract one dataset. None means it was skipped on purpose."""
    if name == "mlend":
        return _fetch_mlend(root)
    raw_dir = root / "raw" / name
    roots = [raw_dir]
    for remote in DATASET_FILES[name]:
        dest = raw_dir / remote.filename
        checksum = Checksum(remote.algo, remote.digest)
        status = download_file(remote.url, dest, checksum)
        LOGGER.info("%s %s %s", name, status, dest)
        if not remote.archive:
            continue
        extracted = root / "extracted" / name / Path(remote.filename).stem
        extract_status = extract_archive(dest, extracted)
        LOGGER.info("%s extract %s %s", name, extract_status, extracted)
        roots.append(extracted)
    return roots


def _fetch_mlend(root: Path) -> list[Path]:
    """Download MLEnd from the public GitHub repo. No Kaggle credentials."""
    raw_dir = root / "raw" / "mlend"
    csv_path = raw_dir / MLEND_CSV_NAME
    status = download_file(f"{MLEND_RAW}/{MLEND_CSV_NAME}", csv_path)
    LOGGER.info("mlend csv %s %s", status, csv_path)
    audio_dir = raw_dir / MLEND_AUDIO_DIR
    missing = save_mlend_audio(_mlend_filenames(csv_path), audio_dir, download_file)
    if missing:
        LOGGER.warning("mlend skipped %s files that returned HTTP 404", missing)
    return [raw_dir]


def save_mlend_audio(filenames: list[str], audio_dir: Path, fetch) -> int:
    """Download each clip. A 404 is skipped and counted. Returns the skip count."""
    missing = 0
    for name in filenames:
        url = f"{MLEND_RAW}/{MLEND_AUDIO_DIR}/{name}"
        try:
            fetch(url, audio_dir / name)
        except RuntimeError as exc:
            if "HTTP 404" not in str(exc):
                raise
            missing += 1
            LOGGER.warning("mlend audio missing (404), skipping %s", name)
    return missing


def _mlend_filenames(path: Path) -> list[str]:
    names: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cleaned = {
                (key or "").strip().lower(): (value or "").strip() for key, value in row.items()
            }
            filename = cleaned.get("filename") or cleaned.get("public filename") or ""
            if filename:
                names.append(Path(filename).name)
    return names


def _materialize_one(
    example: RawExample,
    split: str,
    data_root: Path,
    sample_rate: int,
    rendered: dict[Path, float],
) -> PairRecord:
    suffix = example.source_query.suffix or ".wav"
    query_abs = data_root / "queries" / example.group / f"{_safe_id(example.pair_id)}{suffix}"
    _link_or_copy(example.source_query, query_abs)
    song_rel, start_s, duration_s = _place_song(example, data_root, sample_rate, rendered)
    return PairRecord(
        pair_id=example.pair_id,
        query_path=query_abs.relative_to(data_root).as_posix(),
        qtype=example.qtype,
        qsource="real",
        song_id=example.song_id,
        song_start_s=start_s,
        song_dur_s=duration_s,
        split=split,
        group=example.group,
        song_path=song_rel,
        title=example.title,
    )


def _place_song(
    example: RawExample,
    data_root: Path,
    sample_rate: int,
    rendered: dict[Path, float],
) -> tuple[str | None, float | None, float | None]:
    if example.song_kind != "midi" or example.source_song is None:
        return None, None, None
    dest = data_root / "catalog" / example.group / f"{example.source_song.stem}.wav"
    cached = rendered.get(example.source_song)
    if cached is None:
        cached = render_midi_file(example.source_song, dest, sample_rate)
        rendered[example.source_song] = cached
    return dest.relative_to(data_root).as_posix(), 0.0, cached


def _link_or_copy(source: Path, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size == source.stat().st_size:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


def _song_rows(chosen: list[tuple[RawExample, str]], records: list[PairRecord]) -> list[dict]:
    by_pair = {record.pair_id: record for record in records}
    songs: dict[str, dict] = {}
    for example, _split in sorted(chosen, key=_pair_sort_key):
        record = by_pair[example.pair_id]
        current = songs.get(example.song_id)
        if current is None:
            songs[example.song_id] = _new_song(example, record)
            continue
        _fill_song(current, example, record)
    return [songs[song_id] for song_id in sorted(songs)]


def _new_song(example: RawExample, record: PairRecord) -> dict:
    return {
        "song_id": example.song_id,
        "title": example.title,
        "artist": example.artist,
        "source": example.group,
        "audio_path": record.song_path,
        "duration_s": record.song_dur_s,
        "license": LICENSES.get(example.group, "unknown"),
        "lyrics": None,
        "meta": {},
    }


def _fill_song(current: dict, example: RawExample, record: PairRecord) -> None:
    if not current["title"] and example.title:
        current["title"] = example.title
    if not current["artist"] and example.artist:
        current["artist"] = example.artist
    if not current["audio_path"] and record.song_path:
        current["audio_path"] = record.song_path
        current["duration_s"] = record.song_dur_s


def _take_subset(
    tagged: list[tuple[RawExample, str]],
    max_items: int | None,
) -> list[tuple[RawExample, str]]:
    ordered = sorted(tagged, key=_pair_sort_key)
    if max_items is None or max_items >= len(ordered):
        return ordered
    buckets: dict[str, list[tuple[RawExample, str]]] = {"train": [], "val": [], "test": []}
    for item in ordered:
        buckets[item[1]].append(item)
    chosen: list[tuple[RawExample, str]] = []
    while len(chosen) < max_items:
        grew = False
        for split in ("train", "val", "test"):
            if not buckets[split] or len(chosen) >= max_items:
                continue
            chosen.append(buckets[split].pop(0))
            grew = True
        if not grew:
            break
    return sorted(chosen, key=_pair_sort_key)


def _pair_sort_key(item: tuple[RawExample, str]) -> str:
    return item[0].pair_id


def _record_pair_id(record: PairRecord) -> str:
    return record.pair_id


def _safe_id(value: str) -> str:
    return value.replace(":", "_").replace("/", "_")


def _count_splits(records: list[PairRecord]) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for record in records:
        counts[record.split] = counts.get(record.split, 0) + 1
    return counts
