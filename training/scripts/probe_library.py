"""Search the library with rendered hums of indexed songs, and optionally real audio files.

    python scripts/probe_library.py --db postgresql://... --count 200 --audio hum.wav

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
from hum2song.catalog.probe import hum_crop, rank_of, render_hum, summarize_ranks
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


def rendered_query(track_file: Path, rng: np.random.Generator) -> np.ndarray:
    contour = rmvpe_contour(np.load(track_file))
    return render_hum(
        hum_crop(synthetic_hum(contour, rng), rng), TRACK_RATE, int(rng.integers(1 << 30))
    )


def probe_song(encoder, connection, song_id: str, track_dir: Path, rng) -> dict:
    audio = rendered_query(track_dir / f"{song_id.replace(':', '_')}.npy", rng)
    embedding, voiced_s = encoder.embed(audio)
    results = [asdict(hit) for hit in song_hits(connection, embedding, TOP_K)]
    return {"song_id": song_id, "voiced_s": round(voiced_s, 2), "rank": rank_of(song_id, results)}


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
    rng = np.random.default_rng(PROBE_SEED)
    songs = searchable_songs(connection)
    chosen = [songs[i] for i in np.sort(rng.permutation(len(songs))[: args.count])]
    probes = [probe_song(encoder, connection, song, track_dir, rng) for song in chosen]
    report = {
        "library": library_counts(connection),
        "rendered": summarize_ranks([probe["rank"] for probe in probes]),
        "audio": {str(path): search_audio(encoder, connection, path, 10) for path in args.audio},
        "probes": probes,
    }
    LOGGER.info("rendered-hum probe: %s", report["rendered"])
    text = json.dumps(report, indent=1)
    print(text) if args.out is None else args.out.write_text(text)


if __name__ == "__main__":
    main()
