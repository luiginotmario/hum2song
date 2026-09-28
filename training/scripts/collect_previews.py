"""Collect preview metadata: hum-set targets (MLEnd, MTG-QBH, CHAD) plus chart distractors.

    python scripts/collect_previews.py --name previews_v1 --distractors 3000

Writes library/<name>/targets.jsonl (every target with its iTunes and Deezer match, or
none) and previews.jsonl (one row per preview to extract: role target or distractor).
Chart songs whose title matches a target title are dropped so a distractor is never
the right answer under another id.
"""

import argparse
import json
import sys
from pathlib import Path

from hum2song.catalog.fma import write_jsonl
from hum2song.catalog.previews import PreviewClient, best_match, plain, song_key, split_video_title
from hum2song.catalog.targets import chad_originals, mlend_targets, mtgqbh_targets
from hum2song.config import DEFAULT_DATA_ROOT
from hum2song.logutil import configure_logging, get_logger

LOGGER = get_logger(__name__)
CHART_COUNTRIES = (
    "us",
    "gb",
    "ca",
    "au",
    "ie",
    "nz",
    "de",
    "fr",
    "es",
    "it",
    "nl",
    "be",
    "se",
    "no",
    "dk",
    "fi",
    "at",
    "ch",
    "pt",
    "pl",
    "br",
    "mx",
    "ar",
    "cl",
    "co",
    "jp",
    "kr",
    "in",
    "za",
    "ph",
    "id",
    "sg",
    "my",
    "th",
    "tr",
    "gr",
    "cz",
    "hu",
    "ro",
    "ng",
)


def chad_targets(client: PreviewClient, root: Path) -> list[dict]:
    """CHAD originals get artist/title from the YouTube oEmbed title (metadata only)."""
    targets = []
    for original in chad_originals(root):
        video_title = client.youtube_title(original["youtube_id"])
        parsed = split_video_title(video_title) if video_title else {"artist": "", "title": ""}
        targets.append({**original, **parsed, "video_title": video_title})
    return targets


def search_term(target: dict) -> str:
    return f"{target['artist']} {target['title']}".strip()


def match_target(client: PreviewClient, target: dict) -> dict:
    if not target["title"]:
        return {**target, "itunes": None, "deezer": None}
    term = search_term(target)
    return {
        **target,
        "itunes": best_match(client.itunes_search(term), target),
        "deezer": best_match(client.deezer_search(term), target),
    }


def target_rows(targets: list[dict]) -> list[dict]:
    rows = []
    for target in targets:
        for source in ("itunes", "deezer"):
            if target[source]:
                rows.append({**target[source], "role": "target", "target_id": target["target_id"]})
    return rows


def chart_rows(client: PreviewClient, limit: int) -> list[dict]:
    """Deezer chart playlists first, then Apple Music most-played charts per country."""
    rows = []
    for playlist in client.deezer_chart_playlists():
        rows += client.deezer_playlist_tracks(playlist)
        if len(rows) >= 2 * limit:
            break
    apple_ids = []
    for country in CHART_COUNTRIES:
        apple_ids += [i for i in client.apple_chart_ids(country) if i not in apple_ids]
    rows += client.itunes_lookup(apple_ids)
    return rows


def distractor_rows(charts: list[dict], targets: list[dict], limit: int) -> list[dict]:
    """Unique songs (artist|title and ISRC) with previews, none titled like a target."""
    target_titles = {plain(t["title"]) for t in targets if t["title"]}
    seen, rows = set(), []
    for row in charts:
        keys = {song_key(row["artist"], row["title"]), row.get("isrc") or ""} - {""}
        if not row["preview_url"] or keys & seen or plain(row["title"]) in target_titles:
            continue
        seen |= keys
        rows.append({**row, "role": "distractor", "target_id": None})
    return rows[:limit]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect preview metadata")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--name", default="previews_v1")
    parser.add_argument("--distractors", type=int, default=3000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    out_dir = args.data_root / "library" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    client = PreviewClient()
    raw = mlend_targets() + mtgqbh_targets(args.data_root) + chad_targets(client, args.data_root)
    LOGGER.info("%s targets; searching iTunes and Deezer", len(raw))
    targets = [match_target(client, target) for target in raw]
    write_jsonl(out_dir / "targets.jsonl", targets)
    found = {s: sum(bool(t[s]) for t in targets) for s in ("itunes", "deezer")}
    LOGGER.info("targets matched: %s of %s", found, len(targets))
    distractors = distractor_rows(chart_rows(client, args.distractors), targets, args.distractors)
    rows = target_rows(targets) + distractors
    write_jsonl(out_dir / "previews.jsonl", rows)
    LOGGER.info("%s previews (%s distractors) -> %s", len(rows), len(distractors), out_dir)
    print(json.dumps({"targets": len(targets), "matched": found, "distractors": len(distractors)}))


if __name__ == "__main__":
    main()
