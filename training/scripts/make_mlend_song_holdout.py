"""Draw the 2 held-out MLEnd songs and check HumTrans for the same melodies (D-016).

Songs are drawn with mlend.SONG_SEED from the sorted song ids. HumTrans has no titles, so
overlap is checked by melody: every MLEnd hum (all people) is embedded with an existing
checkpoint and matched to its nearest HumTrans reference segment. If one HumTrans song
wins at least OVERLAP_SHARE of a held-out song's hums, that HumTrans song is listed under
humtrans_exclude and dropped from training. Shares for the kept songs are printed as the
no-overlap baseline. Refuses to overwrite an existing manifest without --force.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from hum2song.contour.evaluate import embed_contours, query_contour, whole_reference
from hum2song.contour.mlend import (
    SONG_MANIFEST,
    TRACKERS,
    choose_heldout_songs,
    load_contours,
    mlend_clips,
    write_song_holdout,
)
from hum2song.contour.train import load_checkpoint
from hum2song.contour.train import load_contours as load_pair_contours
from hum2song.logutil import configure_logging, get_logger
from hum2song.manifest import read_pairs

LOGGER = get_logger(__name__)
OVERLAP_SHARE = 0.25


def humtrans_references(root: Path, config) -> tuple[list[np.ndarray], list[str]]:
    records = [r for r in read_pairs(root / "pairs_real.jsonl") if r.group == "humtrans"]
    references = load_pair_contours(root, records, config.shift_semitones)["references"]
    songs = {str(r.song_path): r.song_id for r in records if r.song_path}
    paths = sorted(path for path in songs if path in references)
    return [whole_reference(references[p])[0] for p in paths], [songs[p] for p in paths]


def nearest_shares(model, root: Path, config, device) -> dict[str, dict]:
    """Per MLEnd song: the HumTrans song nearest to most of its hums, and that share."""
    hums = [c for c in mlend_clips(root) if c.qtype == "hum"]
    contours = load_contours(root, TRACKERS["hum"], hums)
    hum_emb = embed_contours(model, [query_contour(contours[c.path]) for c in hums], device)
    refs, ref_songs = humtrans_references(root, config)
    nearest = np.argmax(hum_emb @ embed_contours(model, refs, device).T, axis=1)
    shares = {}
    for song in sorted({c.song for c in hums}):
        votes = Counter(ref_songs[i] for i, c in zip(nearest, hums, strict=True) if c.song == song)
        top, count = votes.most_common(1)[0]
        shares[song] = {"humtrans_song": top, "share": count / sum(votes.values())}
    return shares


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed held-out MLEnd songs plus overlap check")
    parser.add_argument("--ckpt", type=Path, required=True, help="Model used for the check")
    parser.add_argument("--out", type=Path, default=SONG_MANIFEST)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; the held-out songs are fixed (use --force)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, _step = load_checkpoint(args.ckpt, device)
    root = config.resolved_data_root()
    songs = choose_heldout_songs([c.song for c in mlend_clips(root)])
    shares = nearest_shares(model, root, config, device)
    exclude = [shares[s]["humtrans_song"] for s in songs if shares[s]["share"] >= OVERLAP_SHARE]
    write_song_holdout(args.out, songs, exclude)
    LOGGER.info("held out %s, HumTrans excluded %s", songs, exclude)
    print(json.dumps({"songs": songs, "exclude": exclude, "shares": shares}, indent=2))


if __name__ == "__main__":
    main()
