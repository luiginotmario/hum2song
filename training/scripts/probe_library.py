"""Search the library with rendered hums of indexed songs, and optionally real audio files.

    python scripts/probe_library.py --db postgresql://... --count 200 --audio hum.wav
    python scripts/probe_library.py --db ... --name youtube_charts_v1 --count 400  (D-020)

Only songs of library --name are probed (FMA .npy caches or YouTube .npz tracks, where the
track the index used is rendered: vocals, or the mix for instrumental songs). Results are
also broken down by genre tag and by that track kind.

Rendered hums: each song's cached melody track -> humanize + augment (as in song_eval)
-> random 8-12 s mostly voiced crop -> harmonic tone with noise -> the same search path as the API.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from hum2song.catalog.db import connect, library_counts, searchable_songs, song_hits
from hum2song.catalog.fma import read_jsonl
from hum2song.catalog.probe import hum_crop, rank_of, render_hum, summarize_ranks
from hum2song.catalog.real_eval import pick_track
from hum2song.catalog.real_pool import load_tracks
from hum2song.catalog.search import QueryEncoder, search_audio
from hum2song.config import DEFAULT_DATA_ROOT
from hum2song.contour.features import rmvpe_contour
from hum2song.contour.melody import TRACK_RATE
from hum2song.contour.song_eval import synthetic_hum
from hum2song.logutil import configure_logging, get_logger

LOGGER = get_logger(__name__)
DEFAULT_CKPT = "ckpt/contour_v2_s0/last.pt"
DEFAULT_RMVPE = "fig_contours/rmvpe.pt"
TOP_K = 50
PROBE_SEED = 20260928
METHOD = "vocals_or_mix"


def track_path(track_dir: Path, song_id: str) -> Path | None:
    stem = song_id.replace(":", "_")
    found = [track_dir / f"{stem}{ext}" for ext in (".npy", ".npz")]
    return next((path for path in found if path.exists()), None)


def indexed_track(path: Path) -> tuple[str, np.ndarray]:
    """(kind, contour) of the track the index used (vocals, or the mix as a fallback)."""
    tracks = load_tracks(str(path))
    kind, _chunks = pick_track(tracks, METHOD)
    return ("vocals" if path.suffix == ".npy" else kind), rmvpe_contour(tracks[kind])


def rendered_query(contour: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return render_hum(
        hum_crop(synthetic_hum(contour, rng), rng), TRACK_RATE, int(rng.integers(1 << 30))
    )


def probe_song(encoder, connection, song_id: str, path: Path, genre: str, rng) -> dict:
    kind, contour = indexed_track(path)
    embedding, voiced_s = encoder.embed(rendered_query(contour, rng))
    results = [asdict(hit) for hit in song_hits(connection, embedding, TOP_K)]
    return {
        "song_id": song_id,
        "genre": genre,
        "track": kind,
        "voiced_s": round(voiced_s, 2),
        "rank": rank_of(song_id, results),
    }


def library_genres(root: Path, name: str) -> dict[str, str]:
    path = root / "library" / name / "previews.jsonl"
    rows = read_jsonl(path) if path.exists() else []
    return {row["song_id"]: row.get("genre") or "unknown" for row in rows}


def breakdown(probes: list[dict], field: str) -> dict:
    groups = sorted({probe[field] for probe in probes})
    return {
        group: summarize_ranks([p["rank"] for p in probes if p[field] == group]) for group in groups
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rendered-hum probe of the song library")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--name", default="fma_full_3k")
    parser.add_argument("--db", required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--audio", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = QueryEncoder(args.data_root / DEFAULT_CKPT, args.data_root / DEFAULT_RMVPE, device)
    connection = connect(args.db)
    track_dir = args.data_root / "library" / args.name / "tracks"
    genres = library_genres(args.data_root, args.name)
    rng = np.random.default_rng(PROBE_SEED)
    paths = {song: track_path(track_dir, song) for song in searchable_songs(connection)}
    songs = [song for song, path in paths.items() if path is not None]
    chosen = [songs[i] for i in np.sort(rng.permutation(len(songs))[: args.count])]
    probes = [
        probe_song(encoder, connection, s, paths[s], genres.get(s, "unknown"), rng) for s in chosen
    ]
    report = {
        "library": library_counts(connection),
        "probed_library": args.name,
        "rendered": summarize_ranks([probe["rank"] for probe in probes]),
        "by_genre": breakdown(probes, "genre"),
        "by_track": breakdown(probes, "track"),
        "audio": {str(path): search_audio(encoder, connection, path, 10) for path in args.audio},
        "probes": probes,
    }
    LOGGER.info("rendered-hum probe: %s", report["rendered"])
    text = json.dumps(report, indent=1)
    print(text) if args.out is None else args.out.write_text(text)


if __name__ == "__main__":
    main()
