"""Embed queries and references, then write a JSON report next to the CHAD numbers."""

import json
from pathlib import Path

import numpy as np
import torch

from hum2song.audio import fit_length, load_audio
from hum2song.config import EvalConfig
from hum2song.eval.metrics import (
    CHAD_TOP10,
    MATCHING_NOTE,
    MISSING_RANK,
    PRIMARY_CHAD_TOP10,
    PROTOCOL_NOTE,
    RetrievalScores,
    ranks_for_queries,
    retrieval_scores,
)
from hum2song.logutil import get_logger
from hum2song.manifest import QTYPES, PairRecord, read_jsonl, read_pairs

LOGGER = get_logger(__name__)
COMPARABLE_DISTRACTORS = 2000


def evaluate_pairs(model: torch.nn.Module, pairs: list[PairRecord], config: EvalConfig) -> dict:
    """Score test queries. The model is whatever tower the caller built."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    selected = _select_pairs(pairs, config.set_names())
    results = []
    for set_name in _result_names(selected, config.set_names()):
        group_pairs = [pair for pair in selected if set_name == "all" or pair.group == set_name]
        results.append(_evaluate_group(model, group_pairs, set_name, config, device))
    return {
        "results": results,
        "baselines": CHAD_TOP10,
        "protocol_note": PROTOCOL_NOTE,
        "matching": MATCHING_NOTE,
        "query_types": list(QTYPES),
    }


def _evaluate_group(
    model,
    pairs: list[PairRecord],
    set_name: str,
    config: EvalConfig,
    device,
) -> dict:
    targets = {pair.song_id for pair in pairs}
    distractors = _distractors(config, targets)
    ref_paths, ref_ids = _references(pairs, distractors, config.resolved_data_root())
    query_paths = [pair.query_path for pair in pairs]
    query_ids = [pair.song_id for pair in pairs]
    query_types = [pair.qtype for pair in pairs]
    distractor_count = len({song_id for song_id in ref_ids if song_id not in targets})
    if not pairs or not ref_paths:
        scores = retrieval_scores([])
        return _report_body(
            set_name, scores, {}, distractor_count, len(set(ref_ids)), missing_refs=not ref_paths
        )
    query_emb = _embed_paths(model, query_paths, config, device)
    ref_emb = _embed_paths(model, ref_paths, config, device)
    ranks = ranks_for_queries(query_emb, query_ids, ref_emb, ref_ids)
    per_qtype = {
        qtype: retrieval_scores(grouped).as_dict()
        for qtype, grouped in _ranks_by_qtype(ranks, query_types).items()
    }
    return _report_body(
        set_name,
        retrieval_scores(ranks),
        per_qtype,
        distractor_count,
        len(set(ref_ids)),
        missing_refs=False,
    )


def _report_body(
    set_name: str,
    scores: RetrievalScores,
    per_qtype: dict,
    distractor_count: int,
    reference_songs: int,
    missing_refs: bool,
) -> dict:
    baseline = PRIMARY_CHAD_TOP10.get(set_name)
    ours = scores.top10
    delta = None
    if ours is not None and baseline is not None:
        delta = round(ours - baseline, 6)
    comparable = set_name == "mirqbsh" and distractor_count >= COMPARABLE_DISTRACTORS
    return {
        "set": set_name,
        "split": "test",
        "count": scores.count,
        "reference_songs": reference_songs,
        "distractor_count": distractor_count,
        "missing_references": missing_refs,
        "comparable_to_chad": comparable,
        "metrics": scores.as_dict(),
        "per_query_type": per_qtype,
        "per_qtype": per_qtype,
        "chad_top10": baseline,
        "ours_top10": ours,
        "delta_top10_vs_chad": delta,
        "baselines": CHAD_TOP10,
        "protocol_note": PROTOCOL_NOTE,
    }


def _embed_paths(model, paths: list[str], config: EvalConfig, device: torch.device) -> np.ndarray:
    target = max(int(round(config.crop_seconds * config.sample_rate)), 1)
    rng = np.random.default_rng(0)
    waves = []
    root = config.resolved_data_root()
    for path in paths:
        audio = load_audio(root / path, config.sample_rate, trim=True)
        waves.append(torch.from_numpy(fit_length(audio, target, rng, random_start=False)))
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(waves), config.batch_size):
            batch = torch.stack(waves[start : start + config.batch_size]).to(device)
            chunks.append(model(batch).embedding.float().cpu().numpy())
    if not chunks:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _references(
    pairs: list[PairRecord],
    distractors: list[tuple[str, str]],
    data_root: Path,
) -> tuple[list[str], list[str]]:
    paths: list[str] = []
    song_ids: list[str] = []
    seen: set[str] = set()
    for pair in pairs:
        if not pair.song_path or pair.song_path in seen:
            continue
        if not (data_root / pair.song_path).exists():
            continue
        seen.add(pair.song_path)
        paths.append(pair.song_path)
        song_ids.append(pair.song_id)
    for song_id, audio_path in distractors:
        if audio_path in seen or not (data_root / audio_path).exists():
            continue
        seen.add(audio_path)
        paths.append(audio_path)
        song_ids.append(song_id)
    return paths, song_ids


def _distractors(config: EvalConfig, target_ids: set[str]) -> list[tuple[str, str]]:
    if config.distractors <= 0:
        return []
    songs_path = config.resolved_songs()
    if not songs_path.exists():
        return []
    extras: list[tuple[str, str]] = []
    for row in read_jsonl(songs_path):
        song_id = row.get("song_id")
        audio_path = row.get("audio_path")
        if not song_id or not audio_path or song_id in target_ids:
            continue
        extras.append((str(song_id), str(audio_path)))
    extras = sorted(extras)
    return extras[: config.distractors]


def _select_pairs(pairs: list[PairRecord], set_names: list[str]) -> list[PairRecord]:
    test_pairs = [pair for pair in pairs if pair.split == "test"]
    if set_names == ["all"]:
        return test_pairs
    allowed = set(set_names)
    return [pair for pair in test_pairs if pair.group in allowed]


def _result_names(pairs: list[PairRecord], set_names: list[str]) -> list[str]:
    if set_names == ["all"]:
        return sorted({pair.group for pair in pairs})
    return list(set_names)


def _ranks_by_qtype(ranks: list[int], qtypes: list[str]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {name: [] for name in QTYPES}
    for rank, qtype in zip(ranks, qtypes, strict=True):
        grouped.setdefault(qtype, []).append(rank)
    return grouped


def log_eval_plan(config: EvalConfig) -> None:
    manifest = config.resolved_manifest()
    if not manifest.exists():
        LOGGER.info("dry-run eval: manifest missing at %s", manifest)
        LOGGER.info("chad top-10 baselines %s", CHAD_TOP10)
        return
    pairs = _select_pairs(read_pairs(manifest), config.set_names())
    LOGGER.info("dry-run eval would score %s test rows from %s", len(pairs), manifest)
    LOGGER.info("chad top-10 baselines %s", CHAD_TOP10)
    LOGGER.info("missing-rank sentinel %s", MISSING_RANK)


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
