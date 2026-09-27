"""Contour-branch training: InfoNCE between hummed and MIDI contours (D-011).

Checkpoint selection uses validation data only, never MIR-QBSH or MLEnd test people, so
those numbers stay held-out results (D-005). `select_metric` is one metric name or a
comma list whose mean is used, e.g. HumTrans val plus MLEnd val whistles and hums (D-015).
Optional whistle training (D-015): whistle-like augmentation of hum queries and pairs of
MLEnd train-split whistles with other train-split people's hums.
"""

import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from hum2song.contour.augment import ContourAugment, WhistleAugment
from hum2song.contour.config import ContourConfig
from hum2song.contour.data import (
    ContourPairDataset,
    PairWindows,
    SyntheticPairDataset,
    collate_pairs,
    load_query_contours,
    load_reference_contours,
    midi_index,
    set_epoch,
)
from hum2song.contour.evaluate import (
    ContourSet,
    build_set,
    distractor_paths,
    evaluate,
    mir_sets,
    whole_reference,
)
from hum2song.contour.features import midi_contour
from hum2song.contour.mlend import MLEndValidation, training_pairs
from hum2song.contour.model import ContourEncoder
from hum2song.logutil import get_logger
from hum2song.losses import info_nce_symmetric
from hum2song.manifest import PairRecord, read_pairs
from hum2song.midi_render import parse_midi
from hum2song.seed import seed_everything

LOGGER = get_logger(__name__)
GROUPS = ("humtrans", "mirqbsh")


def build_model(config: ContourConfig) -> ContourEncoder:
    return ContourEncoder(
        dim=config.dim,
        layers=config.layers,
        heads=config.heads,
        dropout=config.dropout,
        out_dim=config.out_dim,
        temperature_init=config.temperature_init,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[ContourEncoder, ContourConfig, int]:
    """Model in eval mode on `device`, its config, and the step it was saved at."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = ContourConfig(**payload["config"])
    model = build_model(config)
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), config, int(payload["step"])


def augment_spec(config: ContourConfig) -> ContourAugment:
    return ContourAugment(
        stretch_min=config.stretch_min,
        stretch_max=config.stretch_max,
        warp_depth=config.warp_depth,
        interval_scale=config.interval_scale,
        drift_semitones=config.drift_semitones,
        jitter_semitones=config.jitter_semitones,
        dropout_prob=config.dropout_prob,
        octave_error_prob=config.octave_error_prob,
    )


def whistle_spec(config: ContourConfig) -> WhistleAugment:
    return WhistleAugment(
        probability=config.whistle_aug_prob,
        compress_min=config.whistle_compress_min,
        compress_max=config.whistle_compress_max,
        gap_prob=config.whistle_gap_prob,
        gap_max_s=config.whistle_gap_max_s,
    )


def window_spec(config: ContourConfig) -> PairWindows:
    return PairWindows(
        query_min_s=config.query_min_s,
        query_max_s=config.query_max_s,
        ref_start_jitter_s=config.ref_start_jitter_s,
        ref_extra_min_s=config.ref_extra_min_s,
        ref_extra_max_s=config.ref_extra_max_s,
    )


def load_contours(root: Path, records: list[PairRecord], shift: float) -> dict:
    """Query contours (plain and shifted HumTrans) and MIDI references for both groups."""
    queries: dict[str, np.ndarray] = {}
    references: dict[str, np.ndarray] = {}
    for group in GROUPS:
        queries.update(load_query_contours(root / "f0" / f"rmvpe_{group}.npz"))
        song_paths = {str(r.song_path) for r in records if r.group == group and r.song_path}
        references.update(load_reference_contours(song_paths, midi_index(root / "raw" / group)))
    shifted = load_query_contours(root / "f0" / f"rmvpe_humtrans_shift{shift:+g}.npz")
    return {"queries": queries, "shifted": shifted, "references": references}


def split_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def synthetic_melodies(root: Path, config: ContourConfig) -> dict[str, np.ndarray]:
    """Melody MIDIs for synthetic pairs, minus the held-out distractors and excluded stems."""
    held_out = set(distractor_paths(root / config.holdout_midi_dir, config.holdout_count))
    excluded = set(split_list(config.synthetic_exclude))
    melodies: dict[str, np.ndarray] = {}
    for folder in split_list(config.synthetic_midi_dirs):
        for path in sorted((root / folder).rglob("*.mid")):
            if path not in held_out and path.stem not in excluded:
                melodies[f"synthetic:{path.stem}"] = midi_contour(parse_midi(path.read_bytes()))
    return melodies


def training_dataset(root: Path, records, contours: dict, config: ContourConfig) -> Dataset:
    """HumTrans hum pairs, plus MIDI-only synthetic pairs and MLEnd whistle pairs if enabled."""
    parts: list[Dataset] = [
        ContourPairDataset(
            select(records, "humtrans", "train"),
            contours["queries"],
            contours["references"],
            augment_spec(config),
            window_spec(config),
            config.seed,
            whistle_spec(config),
        )
    ]
    melodies = synthetic_melodies(root, config)
    if melodies:
        parts.append(
            SyntheticPairDataset(melodies, augment_spec(config), window_spec(config), config.seed)
        )
    if config.mlend_pairs:
        parts.append(training_pairs(root, augment_spec(config), config.seed, config.mlend_repeat))
    LOGGER.info("training pairs: %s", [f"{type(p).__name__}={len(p)}" for p in parts])
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def select(records: list[PairRecord], group: str, split: str) -> list[PairRecord]:
    return [r for r in records if r.group == group and r.split == split and r.song_path]


def validation_sets(
    records, contours: dict, config: ContourConfig, humtrans_split: str = "val"
) -> list[ContourSet]:
    """MIR-QBSH (three reference protocols) plus HumTrans `humtrans_split`, plain and shifted."""
    queries, references = contours["queries"], contours["references"]
    humtrans = select(records, "humtrans", humtrans_split)
    name = f"humtrans_{humtrans_split}"
    shift_name = f"{name}_shift{config.shift_semitones:+g}"
    return [
        *mir_sets(select(records, "mirqbsh", "test"), queries, references),
        build_set(name, humtrans, queries, references, whole_reference),
        build_set(shift_name, humtrans, contours["shifted"], references, whole_reference),
    ]


def lr_multiplier(step: int, warmup: int, total: int) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / float(warmup)
    progress = min(max((step - warmup) / float(max(total - warmup, 1)), 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_step(model, batch: dict, optimizer, device, config: ContourConfig) -> float:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        query = model(batch["query"].to(device), batch["query_valid"].to(device))
        song = model(batch["song"].to(device), batch["song_valid"].to(device))
    loss = info_nce_symmetric(query.float(), song.float(), batch["song_id"], model.temperature())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()
    return float(loss.detach())


def save_checkpoint(path: Path, model, config: ContourConfig, step: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "config": asdict(config),
        "step": step,
        "metrics": metrics,
    }
    torch.save(payload, path)


def batches(loader: DataLoader, dataset: Dataset):
    """Endless stream of batches; each pass over the data reseeds the crops."""
    epoch = 0
    while True:
        set_epoch(dataset, epoch)
        yield from loader
        epoch += 1


def selection_score(metrics: dict[str, float], select_metric: str) -> float:
    """Mean of the comma-listed metrics; -inf when any is missing."""
    names = split_list(select_metric)
    if not names or any(name not in metrics for name in names):
        return float("-inf")
    return float(np.mean([metrics[name] for name in names]))


def validation_metrics(model, sets, mlend: MLEndValidation | None, device) -> dict:
    metrics = evaluate(model, sets, device)
    if mlend is None:
        return metrics
    was_training = model.training
    model.eval()
    metrics.update(mlend.metrics(model, device))
    model.train(was_training)
    return metrics


def run_validation(model, sets, device, step: int, tracker, best: dict, config, mlend) -> dict:
    metrics = validation_metrics(model, sets, mlend, device)
    metrics["val/select"] = selection_score(metrics, config.select_metric)
    tracker.log(metrics, step=step)
    LOGGER.info("step %s %s", step, " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items())))
    score = metrics["val/select"]
    if score > best.get("score", float("-inf")):
        best.update(score=score, step=step, metrics=metrics)
        save_checkpoint(config.resolved_ckpt_dir() / "best.pt", model, config, step, metrics)
    return metrics


def run_training(config: ContourConfig, tracker) -> dict:
    """Train, validate every val_every steps, and return the final and selected metrics."""
    seed_everything(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = config.resolved_data_root()
    records = read_pairs(root / "pairs_real.jsonl")
    contours = load_contours(root, records, config.shift_semitones)
    dataset = training_dataset(root, records, contours, config)
    sets = validation_sets(records, contours, config)
    mlend = MLEndValidation(root, "val") if config.mlend_val else None
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        collate_fn=collate_pairs,
        persistent_workers=config.num_workers > 0,
    )
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    best: dict = {}
    last_metrics = run_validation(model, sets, device, 0, tracker, best, config, mlend)
    started = time.time()
    stream = batches(loader, dataset)
    for step in range(1, config.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = config.lr * lr_multiplier(step, config.warmup_steps, config.steps)
        loss = train_step(model, next(stream), optimizer, device, config)
        if step % config.log_every == 0:
            rate = step / max(time.time() - started, 1.0e-9)
            tracker.log(
                {
                    "train/loss": loss,
                    "train/temperature": float(model.temperature().detach()),
                    "train/steps_per_s": rate,
                },
                step=step,
            )
            LOGGER.info(
                "step %s loss %.4f temp %.4f %.1f steps/s",
                step,
                loss,
                float(model.temperature().detach()),
                rate,
            )
        if step % config.val_every == 0 or step == config.steps:
            last_metrics = run_validation(model, sets, device, step, tracker, best, config, mlend)
    save_checkpoint(
        config.resolved_ckpt_dir() / "last.pt", model, config, config.steps, last_metrics
    )
    return {
        "last": last_metrics,
        "selected_step": best.get("step"),
        "selected": best.get("metrics"),
    }
