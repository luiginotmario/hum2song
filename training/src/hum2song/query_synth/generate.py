"""Write the synthetic whistle and wrong-lyrics set from vocal F0 or MIDI."""

import hashlib
import json
from pathlib import Path

import numpy as np

from hum2song.audio import write_wav
from hum2song.contour.augment import ContourAugment
from hum2song.contour.config import ContourConfig
from hum2song.contour.features import FRAME_S, midi_contour
from hum2song.contour.hum_pairs import voiced_crop
from hum2song.contour.song_pairs import load_song_routes, training_songs
from hum2song.contour.train import augment_spec, split_list
from hum2song.midi_render import parse_midi
from hum2song.query_synth.manifest import split_name, write_manifest
from hum2song.query_synth.render import (
    ORIGIN_SYNTHETIC,
    PRESENTATION,
    STYLES,
    RenderedQuery,
    render_query,
)
from hum2song.query_synth.track import track_singing_rmvpe

OUT_NAME = "query_synth"
MIN_VOICED_S = 3.0


def stable_seed(*parts: object) -> int:
    """Process-stable seed. Independent of hash randomization."""
    text = "|".join(str(part) for part in parts)
    raw = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
    return raw % (2**63 - 1)


def safe_id(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def library_melodies(root: Path, config: ContourConfig) -> dict[str, np.ndarray]:
    """Vocal contours on disk. Titles of evaluation targets are dropped."""
    rows = training_songs(root, split_list(config.song_libraries), config.song_fma_parity)
    routes = load_song_routes(rows)
    return {song_id: route["ref"] for song_id, route in routes.items()}


def midi_melodies(root: Path, midi_dir: str) -> dict[str, np.ndarray]:
    """Melody MIDI under midi_dir, keyed midi:<stem>. Empty when midi_dir is blank."""
    if not midi_dir:
        return {}
    folder = root / midi_dir
    melodies = {}
    for path in sorted(folder.rglob("*.mid")):
        contour = midi_contour(parse_midi(path.read_bytes()))
        if _voiced_s(contour) >= MIN_VOICED_S:
            melodies[f"midi:{path.stem}"] = contour
    return melodies


def cut_windows(
    contour: np.ndarray, rng: np.random.Generator, count: int, query_s: tuple[float, float]
) -> list[tuple[np.ndarray, int, int]]:
    windows = []
    for _index in range(count):
        crop, start, length = voiced_crop(contour, rng, query_s)
        if _voiced_s(crop) < 1.0:
            continue
        windows.append((crop, start, length))
    return windows


def render_song(payload: dict) -> list[dict]:
    """One song: both styles, every window. Existing sidecars are kept."""
    song_id = payload["song_id"]
    contour = np.asarray(payload["contour"], dtype=np.float32)
    root = Path(payload["root"])
    out = root / OUT_NAME
    ref_rel = _write_ref(root, out, song_id, contour)
    spec = ContourAugment(**payload["augment"])
    query_s = (payload["query_min_s"], payload["query_max_s"])
    rng = np.random.default_rng(stable_seed(payload["seed"], song_id, "windows"))
    rows = []
    windows = cut_windows(contour, rng, payload["per_song"], query_s)
    for index, (crop, start, length) in enumerate(windows):
        for style in STYLES:
            rows.append(
                _one_query(
                    root, out, song_id, crop, start, length, index, style, spec, ref_rel, payload
                )
            )
    return rows


def build_rows(payloads: list[dict], workers: int) -> list[dict]:
    if workers <= 1:
        rows = []
        for payload in payloads:
            rows.extend(render_song(payload))
        return rows
    from concurrent.futures import ProcessPoolExecutor

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for song_rows in pool.map(render_song, payloads):
            rows.extend(song_rows)
    return rows


def payloads_for(
    root: Path,
    melodies: dict[str, np.ndarray],
    config: ContourConfig,
    per_song: int,
    source: str,
    sing_tracker: str,
) -> list[dict]:
    spec = augment_spec(config)
    return [
        {
            "song_id": song_id,
            "contour": contour,
            "root": str(root),
            "augment": spec.__dict__,
            "query_min_s": config.query_min_s,
            "query_max_s": config.query_max_s,
            "per_song": per_song,
            "seed": config.seed,
            "source": source,
            "sing_tracker": sing_tracker,
        }
        for song_id, contour in sorted(melodies.items())
        if _voiced_s(contour) >= MIN_VOICED_S
    ]


def _one_query(
    root: Path,
    out: Path,
    song_id: str,
    crop: np.ndarray,
    start: int,
    length: int,
    index: int,
    style: str,
    spec: ContourAugment,
    ref_rel: str,
    payload: dict,
) -> dict:
    query_id = f"{song_id}:{style}:{index}"
    meta_path = out / "meta" / f"{safe_id(query_id)}.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(stable_seed(payload["seed"], query_id))
    rendered = render_query(crop, rng, style, spec)
    rendered = _maybe_rmvpe(rendered, payload["sing_tracker"], root)
    wav_path = out / "wav" / f"{safe_id(query_id)}.wav"
    contour_path = out / "f0" / f"{safe_id(query_id)}.npy"
    write_wav(wav_path, rendered.audio, rendered.sample_rate)
    contour_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(contour_path, rendered.tracked.astype(np.float32))
    row = {
        "query_id": query_id,
        "song_id": song_id,
        "style": style,
        "split": split_name(song_id),
        "origin": ORIGIN_SYNTHETIC,
        "presentation": PRESENTATION,
        "source": payload["source"],
        "wav": wav_path.relative_to(root).as_posix(),
        "contour": contour_path.relative_to(root).as_posix(),
        "ref_contour": ref_rel,
        "start_frame": int(start),
        "length_frames": int(length),
        "syllables": list(rendered.syllables),
        "h2_db": rendered.h2_db,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(row, sort_keys=True), encoding="utf-8")
    return row


_RMVPE: dict = {}


def _rmvpe(root: Path):
    """One RMVPE per process. Workers each load it once."""
    if "model" not in _RMVPE:
        import torch

        from hum2song.contour.trackers import load_rmvpe

        weights = root / "fig_contours" / "rmvpe.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"--sing-tracker rmvpe needs {weights}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _RMVPE["model"] = load_rmvpe(weights, device)
    return _RMVPE["model"]


def _maybe_rmvpe(rendered: RenderedQuery, tracker: str, root: Path) -> RenderedQuery:
    if tracker != "rmvpe" or rendered.style != "wrong_lyrics":
        return rendered
    tracked = track_singing_rmvpe(rendered.audio, _rmvpe(root))
    return RenderedQuery(
        audio=rendered.audio,
        sample_rate=rendered.sample_rate,
        control=rendered.control,
        tracked=tracked.astype(np.float32),
        style=rendered.style,
        syllables=rendered.syllables,
        h2_db=rendered.h2_db,
    )


def _write_ref(root: Path, out: Path, song_id: str, contour: np.ndarray) -> str:
    path = out / "refs" / f"{safe_id(song_id)}.npy"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, contour.astype(np.float32))
    return path.relative_to(root).as_posix()


def _voiced_s(contour: np.ndarray) -> float:
    if len(contour) == 0:
        return 0.0
    return float(np.sum(~np.isnan(contour)) * FRAME_S)


def write_set(root: Path, rows: list[dict]) -> Path:
    path = root / OUT_NAME / "manifest.jsonl"
    write_manifest(path, rows)
    return path
