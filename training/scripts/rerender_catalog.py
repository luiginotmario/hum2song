"""Re-render MIDI reference audio after the tempo-map fix (docs/DECISIONS.md D-010).

The old renderer reset tempo to 120 bpm in every track, so type-1 MIDI files (tempo in
track 0, notes in track 1) were rendered at the wrong speed. This script moves the old
catalog/<group> folders and the manifests to a timestamped backup, renders every
reference again from its source MIDI, and rewrites song_dur_s / duration_s.
"""

import argparse
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from hum2song.audio import audio_duration_s
from hum2song.config import DEFAULT_DATA_ROOT, default_num_workers
from hum2song.data.download import RENDER_SAMPLE_RATE
from hum2song.logutil import configure_logging, get_logger
from hum2song.manifest import read_jsonl, read_pairs, write_jsonl, write_pairs
from hum2song.midi_render import render_midi_file

LOGGER = get_logger(__name__)
DEFAULT_GROUPS = "humtrans,mirqbsh"
CHANGED_RATIO = 0.01


def main(argv: list[str] | None = None) -> None:
    """Back up, re-render, and update durations for the chosen groups."""
    configure_logging()
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    root = Path(args.data_root)
    groups = [name.strip() for name in args.groups.split(",") if name.strip()]
    backup = root / "backup" / f"pre_tempo_fix_{time.strftime('%Y%m%d_%H%M%S')}"
    jobs = [job for group in groups for job in _group_jobs(root, group)]
    LOGGER.info("%s references to render for %s", len(jobs), groups)
    old_durations = _durations(root, jobs)
    _back_up(root, groups, backup)
    durations = _render_all(jobs, args.workers)
    _rewrite_manifests(root, durations)
    _log_changes(old_durations, durations)
    LOGGER.info("done. backup at %s", backup)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-render MIDI references with tempo maps")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--groups", default=DEFAULT_GROUPS)
    parser.add_argument("--workers", type=int, default=default_num_workers())
    return parser.parse_args(argv)


def _group_jobs(root: Path, group: str) -> list[tuple[str, str, str]]:
    """(song_path, midi_path, dest_path) for every distinct reference of one group."""
    midis = {path.stem: path for path in (root / "raw" / group).rglob("*.mid")}
    song_paths = sorted(
        {
            str(record.song_path)
            for record in read_pairs(root / "pairs_real.jsonl")
            if record.group == group and record.song_path
        }
    )
    missing = [path for path in song_paths if Path(path).stem not in midis]
    if missing:
        raise FileNotFoundError(f"{len(missing)} {group} references have no MIDI: {missing[:3]}")
    return [(path, str(midis[Path(path).stem]), str(root / path)) for path in song_paths]


def _durations(root: Path, jobs: list[tuple[str, str, str]]) -> dict[str, float]:
    return {song: audio_duration_s(root / song) for song, _midi, _dest in jobs}


def _back_up(root: Path, groups: list[str], backup: Path) -> None:
    """Move catalog folders (same disk, instant) and copy both manifests."""
    backup.mkdir(parents=True, exist_ok=False)
    for group in groups:
        shutil.move(str(root / "catalog" / group), str(backup / "catalog" / group))
    for name in ("pairs_real.jsonl", "songs.jsonl"):
        shutil.copy2(root / name, backup / name)
    LOGGER.info("backed up %s and manifests to %s", groups, backup)


def _render_all(jobs: list[tuple[str, str, str]], workers: int) -> dict[str, float]:
    with ProcessPoolExecutor(max_workers=max(workers, 1)) as pool:
        durations = list(pool.map(_render_one, jobs, chunksize=64))
    return {song: duration for (song, _midi, _dest), duration in zip(jobs, durations, strict=True)}


def _render_one(job: tuple[str, str, str]) -> float:
    _song, midi, dest = job
    return render_midi_file(Path(midi), Path(dest), RENDER_SAMPLE_RATE)


def _rewrite_manifests(root: Path, durations: dict[str, float]) -> None:
    pairs_path = root / "pairs_real.jsonl"
    records = [_with_duration(record, durations) for record in read_pairs(pairs_path)]
    write_pairs(pairs_path, records)
    songs_path = root / "songs.jsonl"
    write_jsonl(songs_path, [_song_with_duration(row, durations) for row in read_jsonl(songs_path)])


def _with_duration(record, durations: dict[str, float]):
    if record.song_path not in durations:
        return record
    return replace(record, song_dur_s=durations[record.song_path])


def _song_with_duration(row: dict, durations: dict[str, float]) -> dict:
    if row.get("audio_path") not in durations:
        return row
    return {**row, "duration_s": durations[row["audio_path"]]}


def _log_changes(old: dict[str, float], new: dict[str, float]) -> None:
    changed = [song for song in new if abs(new[song] - old[song]) > CHANGED_RATIO * old[song]]
    by_group: dict[str, int] = {}
    for song in changed:
        group = Path(song).parent.name
        by_group[group] = by_group.get(group, 0) + 1
    LOGGER.info("duration changed by more than 1%%: %s of %s %s", len(changed), len(new), by_group)


if __name__ == "__main__":
    main()
