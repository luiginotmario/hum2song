"""Whether the real whistle and sung-query targets can be scored.

The query clips ship with MLEnd and MTG-QBH. The eight MLEnd recordings, and the
MTG-QBH collection audio, do not. FMA is a different catalogue.
"""

from pathlib import Path

from hum2song.catalog.targets import MLEND_TARGETS

MLEND_OWNED = "owned/mlend_targets"
MTG_OWNED = "owned/mtgqbh"
MLEND_LABELS = tuple(MLEND_TARGETS)


def mlend_gate(root: Path) -> dict:
    """Ready only when each of the eight labels has a wav under owned/mlend_targets."""
    missing = [
        label for label in MLEND_LABELS if not (root / MLEND_OWNED / f"{label}.wav").is_file()
    ]
    if missing:
        return {
            "status": "blocked",
            "origin": "real",
            "missing": missing,
            "owned_dir": MLEND_OWNED,
            "reason": _mlend_reason(missing),
        }
    return {"status": "ready", "origin": "real", "missing": [], "owned_dir": MLEND_OWNED}


def mtg_gate(root: Path, song_ids: list[str]) -> dict:
    """Ready only when every queried song id has owned/mtgqbh/<id>.wav."""
    missing = [song_id for song_id in song_ids if not _mtg_wav(root, song_id).is_file()]
    if missing or not song_ids:
        return {
            "status": "blocked",
            "origin": "real",
            "missing": missing,
            "owned_dir": MTG_OWNED,
            "reason": _mtg_reason(missing, bool(song_ids)),
        }
    return {"status": "ready", "origin": "real", "missing": [], "owned_dir": MTG_OWNED}


def safe_song_id(song_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in song_id)


def _mtg_wav(root: Path, song_id: str) -> Path:
    return root / MTG_OWNED / f"{safe_song_id(song_id)}.wav"


def _mlend_reason(missing: list[str]) -> str:
    labels = ", ".join(missing)
    return (
        "Blocked. MLEnd ships the hum and whistle clips, not the eight target recordings "
        f"({labels}). Those titles are not in the FMA tracks on this box, and the YouTube "
        "copies that used to be indexed were deleted. YouTube download is not available here. "
        f"To score whistle-against-song, add a recording you already own at {MLEND_OWNED}/"
        "<label>.wav using the labels Potter, StarWars, Panther, Rain, Hakuna, Mamma, "
        "Showman, Frozen. This script does not fetch audio. "
        "eval_mlend.py still scores whistle-against-hum on the eight labels and does not "
        "need these files; that is a different protocol."
    )


def _mtg_reason(missing: list[str], have_ids: bool) -> str:
    if not have_ids:
        return (
            "Blocked. No MTG-QBH query list was found (pairs_real.jsonl). "
            "The Zenodo release has the 118 sung queries and no collection audio."
        )
    shown = ", ".join(missing[:8])
    extra = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
    return (
        "Blocked. MTG-QBH's release has the sung queries and does not include the "
        f"collection recordings. Missing owned files for: {shown}{extra}. "
        f"Add a recording you already own at {MTG_OWNED}/<song_id>.wav. "
        "The YouTube copies were deleted, and this script does not download them."
    )
