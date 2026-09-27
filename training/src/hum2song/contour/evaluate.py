"""Contour-branch retrieval eval: MIR-QBSH vs its 48 targets, HumTrans val/test.

Queries are whole clips (RMVPE contour, unvoiced ends trimmed, capped at
QUERY_MAX_S). Reference protocols for MIR-QBSH, all against the 48 targets only:
  start10     first 10 s from the first note: the same crop Stage A2's eval embeds.
  start_multi first 6/8/10/12/15 s from the first note, song score = best window. This
              is test-time tempo search on the reference side, like the pitch baseline's
              13 tempo scales, and relies on MIR-QBSH queries starting at the song start.
  anywhere    windows of 6/10/14 s every 1 s over the whole song, best window. Does not
              assume the query starts at the beginning.
HumTrans references are whole segment MIDIs; the shifted set uses RMVPE run on audio
transposed by +7 semitones (extract_f0.py --shift 7). Distractor melodies (e.g. 2,000
Essen folk songs, as in MIREX QBSH) can be added to the MIR-QBSH reference side.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from hum2song.contour.data import collate_contours, features_tensor, seconds_to_frames
from hum2song.contour.features import midi_contour, trim_unvoiced
from hum2song.eval.metrics import ranks_for_queries, retrieval_scores
from hum2song.manifest import PairRecord
from hum2song.midi_render import parse_midi

QUERY_MAX_S = 20.0
START_MULTI_S = (6.0, 8.0, 10.0, 12.0, 15.0)
ANYWHERE_S = (6.0, 10.0, 14.0)
ANYWHERE_HOP_S = 1.0
REPORTED = ("top1", "top10", "mrr")


@dataclass
class ContourSet:
    """Query contours and reference windows (each window carries its song id)."""

    name: str
    queries: list[np.ndarray]
    query_ids: list[str]
    refs: list[np.ndarray]
    ref_ids: list[str]


def start_windows(contour: np.ndarray, lengths_s: tuple[float, ...]) -> list[np.ndarray]:
    trimmed = trim_unvoiced(contour)
    return [trimmed[: seconds_to_frames(length)] for length in lengths_s]


def sliding_windows(contour: np.ndarray, lengths_s: tuple[float, ...], hop_s: float) -> list:
    trimmed = trim_unvoiced(contour)
    hop = seconds_to_frames(hop_s)
    windows = []
    for length in (seconds_to_frames(value) for value in lengths_s):
        last = max(len(trimmed) - length, 0)
        windows.extend(trimmed[start : start + length] for start in range(0, last + 1, hop))
    return [trim_unvoiced(window) for window in windows]


def query_contour(contour: np.ndarray) -> np.ndarray:
    return trim_unvoiced(contour)[: seconds_to_frames(QUERY_MAX_S)]


def build_set(
    name: str,
    records: list[PairRecord],
    queries: dict[str, np.ndarray],
    references: dict[str, np.ndarray],
    windows,
    distractors: dict[str, np.ndarray] | None = None,
) -> ContourSet:
    """`windows(contour) -> list of reference windows` sets the reference protocol.

    `distractors` maps extra song ids to contours that are ranked alongside the targets.
    """
    chosen = sorted(
        (r for r in records if r.query_path in queries and r.song_path in references),
        key=lambda record: record.pair_id,
    )
    songs = {r.song_id: None for r in chosen}
    song_of = {str(r.song_path): r.song_id for r in chosen}
    contours = [(song_of[path], references[path]) for path in sorted(song_of)]
    contours += [
        (song_id, c) for song_id, c in sorted((distractors or {}).items()) if song_id not in songs
    ]
    refs: list[np.ndarray] = []
    ref_ids: list[str] = []
    for song_id, contour in contours:
        for window in windows(contour):
            refs.append(window)
            ref_ids.append(song_id)
    return ContourSet(
        name=name,
        queries=[query_contour(queries[r.query_path]) for r in chosen],
        query_ids=[r.song_id for r in chosen],
        refs=refs,
        ref_ids=ref_ids,
    )


def mir_protocols() -> dict:
    """Reference-window functions for the three MIR-QBSH protocols (module docstring)."""
    return {
        "start10": lambda contour: start_windows(contour, (10.0,)),
        "start_multi": lambda contour: start_windows(contour, START_MULTI_S),
        "anywhere": lambda contour: sliding_windows(contour, ANYWHERE_S, ANYWHERE_HOP_S),
    }


def mir_sets(
    records, queries, references, distractors: dict[str, np.ndarray] | None = None
) -> list[ContourSet]:
    """One set per protocol. With distractors the names read mir48+<count>_<protocol>."""
    prefix = f"mir48+{len(distractors)}" if distractors else "mir48"
    return [
        build_set(f"{prefix}_{name}", records, queries, references, windows, distractors)
        for name, windows in mir_protocols().items()
    ]


def distractor_contours(midi_dir: Path, count: int) -> dict[str, np.ndarray]:
    """`count` MIDIs spread evenly over the sorted files of midi_dir, keyed distractor:<stem>."""
    paths = sorted(midi_dir.rglob("*.mid"))
    if count < len(paths):
        paths = [paths[int(i)] for i in np.linspace(0, len(paths) - 1, count).round()]
    return {f"distractor:{p.stem}": midi_contour(parse_midi(p.read_bytes())) for p in paths}


def whole_reference(contour: np.ndarray) -> list[np.ndarray]:
    return [trim_unvoiced(contour)]


def embed_contours(model, contours: list[np.ndarray], device, batch_size: int = 256) -> np.ndarray:
    """Embed in length-sorted batches; rows come back in input order."""
    order = np.argsort([len(c) for c in contours], kind="stable")
    chunks: list[tuple[np.ndarray, np.ndarray]] = []
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            features, valid = collate_contours([features_tensor(contours[i]) for i in index])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                embedding = model(features.to(device), valid.to(device))
            chunks.append((index, embedding.float().cpu().numpy()))
    output = np.zeros((len(contours), chunks[0][1].shape[1]), dtype=np.float32)
    for index, embedding in chunks:
        output[index] = embedding
    return output


def score_set(model, item: ContourSet, device) -> dict[str, float]:
    query_emb = embed_contours(model, item.queries, device)
    ref_emb = embed_contours(model, item.refs, device)
    scores = retrieval_scores(ranks_for_queries(query_emb, item.query_ids, ref_emb, item.ref_ids))
    values = scores.as_dict()
    return {f"val/{item.name}_{metric}": float(values[metric]) for metric in REPORTED}


def evaluate(model, sets: list[ContourSet], device) -> dict[str, float]:
    was_training = model.training
    model.eval()
    metrics: dict[str, float] = {}
    for item in sets:
        metrics.update(score_set(model, item, device))
    model.train(was_training)
    return metrics
