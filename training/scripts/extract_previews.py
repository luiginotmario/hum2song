"""Previews -> cached melody tracks; the audio is deleted as soon as it is decoded (D-018).

    python scripts/extract_previews.py --name previews_v1

For each row of library/<name>/previews.jsonl: download the 30 s preview to a temporary
file (Deezer URLs are re-signed first), decode it with ffmpeg to 44.1 kHz mono, delete the
file, then cache two RMVPE tracks in tracks/<song_id>.npz: `vocals` (htdemucs vocal stem,
D-014) and `mix` (RMVPE on the whole mix, for instrumentals). status.jsonl records every
outcome, so failures are counted rather than silently dropped.
"""

import argparse
import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from hum2song.catalog.fma import read_jsonl
from hum2song.catalog.library import decode_audio, melody_track, to_track_rate
from hum2song.catalog.previews import PreviewClient
from hum2song.config import DEFAULT_DATA_ROOT
from hum2song.contour.melody import SEPARATOR_RATE, load_separator
from hum2song.contour.trackers import load_rmvpe, rmvpe_track
from hum2song.logutil import configure_logging, get_logger

LOGGER = get_logger(__name__)
DEFAULT_RMVPE = "fig_contours/rmvpe.pt"
DOWNLOAD_THREADS = 8
BLOCK = 64
MIN_AUDIO_S = 5.0
LOG_EVERY = 100


def track_file(track_dir: Path, song_id: str) -> Path:
    return track_dir / f"{song_id.replace(':', '_')}.npz"


def fresh_url(client: PreviewClient, row: dict) -> str | None:
    if row["source"] != "deezer_preview":
        return row["preview_url"]
    track = client.deezer_track(row["song_id"].split(":", 1)[1])
    return track["preview_url"] if track else None


def fetch_audio(client: PreviewClient, row: dict, scratch: Path) -> dict:
    """Download, decode and delete one preview; returns audio or the failure reason."""
    url = fresh_url(client, row)
    if not url:
        return {"row": row, "status": "no_preview_url"}
    path = scratch / f"{row['song_id'].replace(':', '_')}.bin"
    try:
        client.download(url, path)
        audio = decode_audio(path, SEPARATOR_RATE)
    except Exception as error:  # noqa: BLE001 - count every failure, keep going
        return {"row": row, "status": "download_or_decode_failed", "error": repr(error)[:200]}
    finally:
        path.unlink(missing_ok=True)
    if len(audio) < MIN_AUDIO_S * SEPARATOR_RATE:
        return {"row": row, "status": "too_short", "duration_s": len(audio) / SEPARATOR_RATE}
    return {"row": row, "status": "ok", "audio": audio}


def extract(item: dict, models: dict, device) -> dict:
    audio = item["audio"]
    return {
        "vocals": melody_track(models["separator"], models["rmvpe"], audio, device),
        "mix": rmvpe_track(models["rmvpe"], to_track_rate(audio, SEPARATOR_RATE)),
    }


def status_line(item: dict) -> dict:
    row = item["row"]
    fields = {k: item[k] for k in ("status", "error", "duration_s") if k in item}
    return {"song_id": row["song_id"], "role": row["role"], **fields}


def process_block(pool, client, rows, scratch, models, device, track_dir) -> list[dict]:
    lines = []
    for item in pool.map(lambda row: fetch_audio(client, row, scratch), rows):
        if item["status"] == "ok":
            item["duration_s"] = round(len(item["audio"]) / SEPARATOR_RATE, 2)
            np.savez(track_file(track_dir, item["row"]["song_id"]), **extract(item, models, device))
        lines.append(status_line(item))
    return lines


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract melody tracks from previews")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--name", default="previews_v1")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    library_dir = args.data_root / "library" / args.name
    track_dir = library_dir / "tracks"
    track_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_jsonl(library_dir / "previews.jsonl")]
    rows = [r for r in rows if not track_file(track_dir, r["song_id"]).exists()]
    models = {
        "separator": load_separator(device),
        "rmvpe": load_rmvpe(args.data_root / DEFAULT_RMVPE, device),
    }
    client = PreviewClient()
    LOGGER.info("%s previews to extract", len(rows))
    started = time.time()
    with (
        tempfile.TemporaryDirectory(prefix="h2s_previews_") as scratch,
        ThreadPoolExecutor(DOWNLOAD_THREADS) as pool,
        open(library_dir / "status.jsonl", "a", encoding="utf-8") as status,
    ):
        for first in range(0, len(rows), BLOCK):
            block = rows[first : first + BLOCK]
            lines = process_block(pool, client, block, Path(scratch), models, device, track_dir)
            status.writelines(json.dumps(line) + "\n" for line in lines)
            status.flush()
            done = first + len(block)
            LOGGER.info("%s / %s previews, %.2f/s", done, len(rows), done / (time.time() - started))


if __name__ == "__main__":
    main()
