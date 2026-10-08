"""Eval report: real sets when the audio is on disk, synthetic hold-out always labeled synthetic."""

from pathlib import Path

import numpy as np

from hum2song.audio import load_audio
from hum2song.contour.data import seconds_to_frames
from hum2song.contour.evaluate import embed_contours, query_contour, score_set
from hum2song.contour.features import rmvpe_contour, trim_unvoiced
from hum2song.contour.mlend import clips_in, load_contours, mlend_clips, read_split
from hum2song.contour.trackers import load_rmvpe, rmvpe_track
from hum2song.eval.metrics import ranks_for_queries, retrieval_scores
from hum2song.manifest import read_pairs
from hum2song.query_synth.gates import (
    MLEND_LABELS,
    MLEND_OWNED,
    mlend_gate,
    mtg_gate,
    safe_song_id,
)
from hum2song.query_synth.manifest import read_manifest, require_synthetic
from hum2song.query_synth.render import PRESENTATION

RMVPE_WEIGHTS = "fig_contours/rmvpe.pt"
WINDOW_S = 8.0
HUM_NAMES = ("humtrans_test", "mir48_anywhere")


def spread_windows(contour: np.ndarray, count: int) -> list[np.ndarray]:
    """Up to `count` windows of WINDOW_S spread from the first voiced frame to the last."""
    trimmed = trim_unvoiced(contour)
    length = min(seconds_to_frames(WINDOW_S), len(trimmed))
    if length <= 0:
        return []
    span = len(trimmed) - length
    if span <= 0:
        return [trimmed]
    starts = np.unique(np.linspace(0, span, num=min(max(count, 1), span + 1)).astype(int))
    return [trimmed[int(start) : int(start) + length] for start in starts]


def score_lists(
    model,
    queries: list[np.ndarray],
    query_ids: list[str],
    refs: list[np.ndarray],
    ref_ids: list[str],
    device,
) -> dict:
    """top-1, top-10, and MRR. Empty inputs get a count of 0 and null rates."""
    if not queries or not refs:
        return {
            "count": len(queries),
            "n_songs": len(set(ref_ids)),
            "top1": None,
            "top10": None,
            "mrr": None,
        }
    query_emb = embed_contours(model, queries, device)
    ref_emb = embed_contours(model, refs, device)
    scores = retrieval_scores(ranks_for_queries(query_emb, query_ids, ref_emb, ref_ids))
    return {
        "count": scores.count,
        "n_songs": len(set(ref_ids)),
        "top1": scores.top1,
        "top10": scores.top10,
        "mrr": scores.mrr,
    }


def synthetic_section(root, model, device, manifest: str, ref_windows: int, limit: int) -> dict:
    path = root / manifest
    try:
        rows = read_manifest(path)
    except FileNotFoundError as exc:
        return {
            "status": "missing",
            "origin": "synthetic",
            "presentation": PRESENTATION,
            "reason": str(exc),
        }
    for row in rows:
        require_synthetic(row)
    test = [row for row in rows if row["split"] == "test"]
    if limit > 0:
        test = test[:limit]
    if not test:
        return {
            "status": "missing",
            "origin": "synthetic",
            "presentation": PRESENTATION,
            "reason": "no synthetic test rows",
        }
    songs = _song_contours(root, rows)
    queries = [query_contour(np.load(root / row["contour"])) for row in test]
    query_ids = [row["song_id"] for row in test]
    refs, ref_ids = _index(songs, ref_windows)
    body = score_lists(model, queries, query_ids, refs, ref_ids, device)
    body.update(
        status="ok",
        origin="synthetic",
        presentation=PRESENTATION,
        styles=sorted({row["style"] for row in test}),
        note="Held-out synthetic queries. Not recordings of people.",
    )
    return body


def hum_section(root: Path, model, device, config) -> dict:
    """HumTrans test and MIR-QBSH anywhere, MIDI references. The small real-hum baseline."""
    try:
        from hum2song.contour.train import load_contours, validation_sets

        records = read_pairs(root / "pairs_real.jsonl")
        contours = load_contours(root, records, config.shift_semitones)
        sets = validation_sets(records, contours, config, humtrans_split="test")
    except FileNotFoundError as exc:
        return {"status": "missing", "origin": "real", "reason": str(exc)}
    chosen = [
        item
        for item in sets
        if item.name in HUM_NAMES or item.name.startswith("humtrans_test_shift")
    ]
    metrics = {}
    for item in chosen:
        block = _from_score_set(score_set(model, item, device), item.name, len(item.queries))
        metrics[item.name] = block
    return {
        "status": "ok",
        "origin": "real",
        "note": "Hum baseline on HumTrans and MIR-QBSH. MIDI references. Not the synthetic set.",
        "sets": metrics,
    }


def mlend_section(root: Path, model, device, ref_windows: int) -> dict:
    gate = mlend_gate(root)
    if gate["status"] != "ready":
        return gate
    weights = root / RMVPE_WEIGHTS
    if not weights.is_file():
        return {**gate, "status": "blocked", "reason": f"RMVPE weights missing at {weights}"}
    try:
        split = read_split()
        clips = clips_in(mlend_clips(root), split, "test", "whistle")
        contours = load_contours(root, "peak", clips)
    except FileNotFoundError as exc:
        return {**gate, "status": "missing", "reason": str(exc)}
    rmvpe = load_rmvpe(weights, device)
    refs, ref_ids = _owned_mlend(root, rmvpe, ref_windows)
    queries, query_ids = _mlend_queries(clips, contours)
    body = score_lists(model, queries, query_ids, refs, ref_ids, device)
    body.update(status="ok", origin="real", protocol="mlend test whistles vs owned song audio")
    return body


def mtg_section(root: Path, model, device, ref_windows: int) -> dict:
    try:
        records = [
            r
            for r in read_pairs(root / "pairs_real.jsonl")
            if r.group == "mtgqbh" and r.qtype == "sing"
        ]
    except FileNotFoundError as exc:
        return {"status": "missing", "origin": "real", "reason": str(exc)}
    gate = mtg_gate(root, sorted({r.song_id for r in records}))
    if gate["status"] != "ready":
        return gate
    weights = root / RMVPE_WEIGHTS
    if not weights.is_file():
        return {**gate, "status": "blocked", "reason": f"RMVPE weights missing at {weights}"}
    npz = root / "f0" / "rmvpe_mtgqbh.npz"
    if not npz.is_file():
        return {**gate, "status": "missing", "reason": f"missing {npz}"}
    from hum2song.contour.data import load_query_contours

    rmvpe = load_rmvpe(weights, device)
    queries, query_ids = [], []
    tracks = load_query_contours(npz)
    for record in records:
        if record.query_path not in tracks:
            continue
        queries.append(query_contour(tracks[record.query_path]))
        query_ids.append(record.song_id)
    refs, ref_ids = [], []
    for song_id in sorted({r.song_id for r in records}):
        wav = root / "owned" / "mtgqbh" / f"{safe_song_id(song_id)}.wav"
        audio = load_audio(wav, 16000, trim=False)
        contour = rmvpe_contour(rmvpe_track(rmvpe, audio))
        for window in spread_windows(contour, ref_windows):
            refs.append(window)
            ref_ids.append(song_id)
    body = score_lists(model, queries, query_ids, refs, ref_ids, device)
    body.update(status="ok", origin="real", protocol="MTG-QBH sung queries vs owned song audio")
    return body


def build_report(root, model, device, config, manifest: str, ref_windows: int, limit: int) -> dict:
    return {
        "replaces_e2b": False,
        "encoder": "ContourEncoder",
        "checkpoint_run": config.run_name,
        "sets": {
            "mlend_whistle": mlend_section(root, model, device, ref_windows),
            "mtgqbh_sing": mtg_section(root, model, device, ref_windows),
            "hum_baseline": hum_section(root, model, device, config),
            "synthetic_holdout": synthetic_section(
                root, model, device, manifest, ref_windows, limit
            ),
        },
    }


def _from_score_set(metrics: dict, name: str, count: int) -> dict:
    return {
        "count": count,
        "top1": metrics.get(f"val/{name}_top1"),
        "top10": metrics.get(f"val/{name}_top10"),
        "mrr": metrics.get(f"val/{name}_mrr"),
        "origin": "real",
    }


def _song_contours(root: Path, rows: list[dict]) -> dict[str, np.ndarray]:
    songs = {}
    for row in rows:
        if row["song_id"] not in songs:
            songs[row["song_id"]] = np.load(root / row["ref_contour"])
    return songs


def _index(songs: dict[str, np.ndarray], count: int) -> tuple[list[np.ndarray], list[str]]:
    refs, ids = [], []
    for song_id, contour in sorted(songs.items()):
        for window in spread_windows(contour, count):
            refs.append(window)
            ids.append(song_id)
    return refs, ids


def _owned_mlend(root, model, count: int) -> tuple[list[np.ndarray], list[str]]:
    refs, ids = [], []
    for label in MLEND_LABELS:
        audio = load_audio(root / MLEND_OWNED / f"{label}.wav", 16000, trim=False)
        contour = rmvpe_contour(rmvpe_track(model, audio))
        for window in spread_windows(contour, count):
            refs.append(window)
            ids.append(label)
    return refs, ids


def _mlend_queries(clips, contours) -> tuple[list[np.ndarray], list[str]]:
    queries, ids = [], []
    for clip in clips:
        if clip.path not in contours:
            continue
        label = _label(clip.song)
        if label is None:
            continue
        queries.append(query_contour(contours[clip.path]))
        ids.append(label)
    return queries, ids


def _label(song_id: str) -> str | None:
    for label in MLEND_LABELS:
        if song_id == label or song_id.endswith(":" + label):
            return label
    return None
