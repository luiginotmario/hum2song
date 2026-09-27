"""Embed queries and references, then write a JSON report next to the CHAD numbers."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from hum2song.audio import fit_length, load_audio
from hum2song.augment import transpose_and_stretch
from hum2song.config import EvalConfig
from hum2song.eval.metrics import (
    CHAD_TOP10,
    MATCHING_NOTE,
    MIRQBSH_PROTOCOL_SONGS,
    MISSING_RANK,
    PRIMARY_CHAD_TOP10,
    PROTOCOL_NOTE,
    RetrievalScores,
    chad_comparability,
    ranks_for_queries,
    retrieval_scores,
)
from hum2song.logutil import get_logger
from hum2song.manifest import QTYPES, PairRecord, read_jsonl, read_pairs

LOGGER = get_logger(__name__)
EXCLUDED_DISTRACTOR_SOURCES = frozenset({"humtrans"})
TARGETS_ONLY_NOTE = (
    "targets_only ranks each query against the set's own reference songs and nothing else "
    "(48 songs for MIR-QBSH). It is always reported, and it is the headline number when no "
    "distractors are available. HumTrans renders are never used as distractors: they are "
    "the training distribution, share its synthesis, and are locked to HumTrans hums. "
    "Each reference is one clip, its first crop_seconds; songs are not chunked in this eval."
)


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
    distractor_count = len({song_id for song_id in ref_ids if song_id not in targets})
    target_songs = len({song_id for song_id in ref_ids if song_id in targets})
    if not pairs or not ref_paths:
        reason = _skip_reason(set_name) if not ref_paths else "no test queries"
        LOGGER.warning(reason)
        body = _report_body(
            set_name,
            retrieval_scores([]),
            _empty_per_qtype(),
            distractor_count,
            target_songs,
            missing_refs=not ref_paths,
            skipped=True,
            skip_reason=reason,
        )
        return {**body, **_targets_only_block(retrieval_scores([]), distractor_count)}
    query_paths = [pair.query_path for pair in pairs]
    query_ids = [pair.song_id for pair in pairs]
    query_types = [pair.qtype for pair in pairs]
    query_emb = _embed_paths(model, query_paths, config, device)
    ref_emb = _embed_paths(model, ref_paths, config, device)
    ranks = ranks_for_queries(query_emb, query_ids, ref_emb, ref_ids)
    target_ranks = _targets_only_ranks(query_emb, query_ids, ref_emb, ref_ids, targets)
    per_qtype = {
        qtype: retrieval_scores(grouped).as_dict()
        for qtype, grouped in _ranks_by_qtype(ranks, query_types).items()
    }
    body = _report_body(
        set_name,
        retrieval_scores(ranks),
        per_qtype,
        distractor_count,
        target_songs,
        missing_refs=False,
        skipped=False,
        skip_reason=None,
    )
    return {**body, **_targets_only_block(retrieval_scores(target_ranks), distractor_count)}


def _targets_only_ranks(
    query_emb: np.ndarray,
    query_ids: list[str],
    ref_emb: np.ndarray,
    ref_ids: list[str],
    targets: set[str],
) -> list[int]:
    keep = [index for index, song_id in enumerate(ref_ids) if song_id in targets]
    return ranks_for_queries(query_emb, query_ids, ref_emb[keep], [ref_ids[i] for i in keep])


def _targets_only_block(scores: RetrievalScores, distractor_count: int) -> dict:
    """The no-distractor line, and which line is the headline number."""
    headline = "metrics" if distractor_count > 0 else "targets_only"
    LOGGER.info("targets-only %s (headline: %s)", scores.as_dict(), headline)
    return {
        "targets_only": scores.as_dict(),
        "headline": headline,
        "targets_only_note": TARGETS_ONLY_NOTE,
    }


def _report_body(
    set_name: str,
    scores: RetrievalScores,
    per_qtype: dict,
    distractor_count: int,
    target_songs: int,
    missing_refs: bool,
    skipped: bool,
    skip_reason: str | None,
) -> dict:
    baseline = PRIMARY_CHAD_TOP10.get(set_name)
    ours = scores.top10
    delta = None
    if ours is not None and baseline is not None:
        delta = round(ours - baseline, 6)
    comparable, label = chad_comparability(set_name, target_songs, distractor_count)
    if skipped:
        comparable = False
        label = f"not comparable to CHAD 0.921: {skip_reason}"
    LOGGER.info(label)
    return {
        "set": set_name,
        "split": "test",
        "count": scores.count,
        "reference_songs": target_songs,
        "target_songs": target_songs,
        "protocol_songs": MIRQBSH_PROTOCOL_SONGS if set_name == "mirqbsh" else None,
        "distractor_count": distractor_count,
        "missing_references": missing_refs,
        "skipped": skipped,
        "skip_reason": skip_reason,
        "comparable_to_chad": comparable,
        "chad_comparison": label,
        "metrics": scores.as_dict(),
        "per_query_type": per_qtype,
        "per_qtype": per_qtype,
        "chad_top10": baseline,
        "ours_top10": ours,
        "delta_top10_vs_chad": delta,
        "baselines": CHAD_TOP10,
        "protocol_note": PROTOCOL_NOTE,
    }


def _skip_reason(set_name: str) -> str:
    return (
        f"skipping {set_name}: no reference audio on disk. "
        "MLEnd and MTG-QBH ship queries only, so those sets are not scored "
        "until song audio is added."
    )


def _empty_per_qtype() -> dict:
    return {name: retrieval_scores([]).as_dict() for name in QTYPES}


def _embed_paths(model, paths: list[str], config: EvalConfig, device: torch.device) -> np.ndarray:
    waves = _load_waves(paths, config)
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(waves), config.batch_size):
            batch = torch.stack(waves[start : start + config.batch_size]).to(device)
            chunks.append(model(batch).embedding.float().cpu().numpy())
    if not chunks:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _load_waves(paths: list[str], config: EvalConfig) -> list[torch.Tensor]:
    return load_fitted_waves(
        paths,
        config.resolved_data_root(),
        config.sample_rate,
        config.crop_seconds,
        config.resolved_num_workers(),
    )


def load_fitted_waves(
    paths: list[str],
    data_root: Path,
    sample_rate: int,
    crop_seconds: float,
    workers: int,
    semitones: float = 0.0,
) -> list[torch.Tensor]:
    """Load clips as fixed-length eval crops (from the start, looped when short).

    `semitones` transposes each clip first, for key-robustness checks.
    """
    target = max(int(round(crop_seconds * sample_rate)), 1)
    jobs = [(path, str(data_root), sample_rate, target, semitones) for path in paths]
    if workers <= 1 or len(jobs) <= 1:
        return [_load_fitted_wave(job) for job in jobs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_load_fitted_wave, jobs))


def _load_fitted_wave(job: tuple[str, str, int, int, float]) -> torch.Tensor:
    path, root, sample_rate, target, semitones = job
    rng = np.random.default_rng(0)
    audio = load_audio(Path(root) / path, sample_rate, trim=True)
    if semitones:
        audio = transpose_and_stretch(audio, semitones, 1.0)
    return torch.from_numpy(fit_length(audio, target, rng, random_start=False))


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
        if not song_id or not audio_path or song_id in target_ids or _excluded_source(row):
            continue
        extras.append((str(song_id), str(audio_path)))
    extras = sorted(extras)
    return extras[: config.distractors]


def _excluded_source(row: dict) -> bool:
    source = row.get("source") or str(row.get("song_id", "")).split(":", 1)[0]
    return source in EXCLUDED_DISTRACTOR_SOURCES


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
