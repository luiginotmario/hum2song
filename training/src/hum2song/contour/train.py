"""Contour-branch training: InfoNCE between hummed and MIDI contours (D-011).

Checkpoint selection uses validation data only, never MIR-QBSH or MLEnd test people, so
those numbers stay held-out results (D-005). `select_metric` is one metric name or a
comma list whose mean is used, e.g. HumTrans val plus MLEnd val whistles and hums (D-015).
Optional whistle training (D-015): whistle-like augmentation of hum queries and pairs of
MLEnd train-split whistles with other train-split people's hums. Optional CHAD pairs
(D-019): real hums with the hummed window of the real recording, train-split songs only.
Optional song-window pairs (D-026, E2a): a second loader of self-supervised windows from
real song melody tracks, trained with the CLEWS loss next to the InfoNCE loss.
"""

import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from hum2song.contour import chad
from hum2song.contour.augment import ContourAugment, WhistleAugment
from hum2song.contour.config import ContourConfig
from hum2song.contour.cover_pairs import CoverPairDataset, cover_items
from hum2song.contour.data import (
    ContourPairDataset,
    PairWindows,
    SyntheticPairDataset,
    collate_pairs,
    load_query_contours,
    load_reference_contours,
    midi_index,
    set_epoch,
    set_input_kind,
)
from hum2song.contour.evaluate import (
    ContourSet,
    build_set,
    distractor_paths,
    evaluate,
    mir_sets,
    whole_reference,
)
from hum2song.contour.features import FEATURE_DIM, SALIENCE_FEATURE_DIM, midi_contour
from hum2song.contour.mlend import MLEndValidation, read_song_holdout, training_pairs
from hum2song.contour.model import ContourEncoder
from hum2song.contour.song_pairs import (
    SongWindowDataset,
    collate_song_windows,
    load_song_routes,
    training_songs,
)
from hum2song.logutil import get_logger
from hum2song.losses import clews_loss, info_nce_symmetric
from hum2song.manifest import PairRecord, read_pairs
from hum2song.midi_render import parse_midi
from hum2song.seed import seed_everything

LOGGER = get_logger(__name__)
GROUPS = ("humtrans", "mirqbsh")


def input_dim(config: ContourConfig) -> int:
    return SALIENCE_FEATURE_DIM if config.input_kind == "salience" else FEATURE_DIM


def build_model(config: ContourConfig) -> ContourEncoder:
    return ContourEncoder(
        dim=config.dim,
        layers=config.layers,
        heads=config.heads,
        dropout=config.dropout,
        out_dim=config.out_dim,
        temperature_init=config.temperature_init,
        in_dim=input_dim(config),
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[ContourEncoder, ContourConfig, int]:
    """Model in eval mode on `device`, its config, and the step it was saved at."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = ContourConfig(**payload["config"])
    set_input_kind(config.input_kind)
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


def song_holdout(config: ContourConfig) -> dict:
    """Held-out MLEnd songs and HumTrans exclusions (D-016); empty when switched off."""
    if not config.song_holdout:
        return {"songs": [], "humtrans_exclude": []}
    return read_song_holdout()


def training_dataset(root: Path, records, contours: dict, config: ContourConfig) -> Dataset:
    """HumTrans hum pairs, plus MIDI-only synthetic pairs and MLEnd whistle pairs if enabled."""
    holdout = song_holdout(config)
    excluded = set(holdout["humtrans_exclude"])
    humtrans = [r for r in select(records, "humtrans", "train") if r.song_id not in excluded]
    parts: list[Dataset] = [
        ContourPairDataset(
            humtrans,
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
        parts.append(
            training_pairs(
                root,
                augment_spec(config),
                config.seed,
                config.mlend_repeat,
                set(holdout["songs"]),
            )
        )
    if config.chad_pairs:
        parts.append(
            chad.training_pairs(root, augment_spec(config), config.seed, config.chad_repeat, config)
        )
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


def cover_pair_dataset(root: Path, config: ContourConfig) -> CoverPairDataset:
    items = cover_items(root)
    groups = len({item["group"] for item in items})
    LOGGER.info("cover pairs: %s covers of %s songs", len(items), groups)
    return CoverPairDataset(
        items,
        augment_spec(config),
        config.seed,
        refs=config.song_refs,
        offset_s=config.song_offset_s,
        mix_prob=config.song_mix_prob,
        query_s=(config.query_min_s, config.query_max_s),
    )


def song_dataset(root: Path, config: ContourConfig) -> SongWindowDataset:
    """Song-window (D-026) or cover-pair (D-028) dataset for the CLEWS loss."""
    builders = {"windows": song_window_dataset, "covers": cover_pair_dataset}
    if config.song_source not in builders:
        raise ValueError(f"unknown song_source: {config.song_source}")
    return builders[config.song_source](root, config)


def song_window_dataset(root: Path, config: ContourConfig) -> SongWindowDataset:
    rows = training_songs(root, split_list(config.song_libraries), config.song_fma_parity)
    routes = load_song_routes(rows)
    with_mix = sum(route["alt"] is not None for route in routes.values())
    LOGGER.info("song-window pairs: %s songs (%s with a mix route)", len(routes), with_mix)
    return SongWindowDataset(
        routes,
        augment_spec(config),
        config.seed,
        refs=config.song_refs,
        offset_s=config.song_offset_s,
        mix_prob=config.song_mix_prob,
        query_s=(config.query_min_s, config.query_max_s),
    )


def song_window_loss(model, batch: dict, device, config: ContourConfig) -> torch.Tensor:
    """CLEWS loss of one song-window batch (D-026)."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        query = model(batch["query"].to(device), batch["query_valid"].to(device))
        refs = model(batch["refs"].to(device), batch["refs_valid"].to(device))
    refs = refs.float().reshape(query.shape[0], batch["refs_per_song"], -1)
    return clews_loss(query.float(), refs, batch["song_id"], config.clews_gamma, config.clews_eps)


def train_step(
    model, batch: dict, optimizer, device, config: ContourConfig, song_batch=None
) -> float:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        query = model(batch["query"].to(device), batch["query_valid"].to(device))
        song = model(batch["song"].to(device), batch["song_valid"].to(device))
    loss = info_nce_symmetric(query.float(), song.float(), batch["song_id"], model.temperature())
    if song_batch is not None:
        loss = loss + config.song_loss_weight * song_window_loss(model, song_batch, device, config)
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


def song_batches(root: Path, config: ContourConfig):
    """Endless song-window batches (D-026), or None when switched off."""
    if not config.song_pairs:
        return None
    dataset = song_dataset(root, config)
    loader = DataLoader(
        dataset,
        batch_size=config.song_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=max(config.num_workers // 2, 0),
        collate_fn=collate_song_windows,
        persistent_workers=config.num_workers > 1,
    )
    return batches(loader, dataset)


def selection_score(metrics: dict[str, float], select_metric: str) -> float:
    """Mean of the comma-listed metrics; -inf when any is missing."""
    names = split_list(select_metric)
    if not names or any(name not in metrics for name in names):
        return float("-inf")
    return float(np.mean([metrics[name] for name in names]))


def validation_metrics(model, sets, extra: list, device) -> dict:
    """Contour sets plus optional MLEnd (D-015) and CHAD (D-019) validators."""
    metrics = evaluate(model, sets, device)
    was_training = model.training
    model.eval()
    for validator in extra:
        metrics.update(validator.metrics(model, device))
    model.train(was_training)
    return metrics


def run_validation(model, sets, device, step: int, tracker, best: dict, config, validators) -> dict:
    metrics = validation_metrics(model, sets, validators, device)
    metrics["val/select"] = selection_score(metrics, config.select_metric)
    tracker.log(metrics, step=step)
    LOGGER.info("step %s %s", step, " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items())))
    score = metrics["val/select"]
    if score > best.get("score", float("-inf")):
        best.update(score=score, step=step, metrics=metrics)
        save_checkpoint(config.resolved_ckpt_dir() / "best.pt", model, config, step, metrics)
    return metrics


def load_initial_weights(model, config: ContourConfig, device) -> None:
    """Fine-tuning: copy the weights of `init_ckpt` into the freshly built model."""
    if not config.init_ckpt:
        return
    path = config.resolved_data_root() / config.init_ckpt
    state = torch.load(path, map_location=device, weights_only=False)["model"]
    current = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
    missing = model.load_state_dict(compatible, strict=False)
    LOGGER.info(
        "initialised from %s (%s/%s tensors; skipped %s)",
        path,
        len(compatible),
        len(state),
        len(missing.missing_keys) + len(missing.unexpected_keys),
    )


def should_stop(best: dict, step: int, config: ContourConfig) -> bool:
    """Early stopping: no better val/select for `early_stop_patience` validations."""
    if config.early_stop_patience <= 0:
        return False
    return step - best.get("step", 0) >= config.early_stop_patience * config.val_every


def run_training(config: ContourConfig, tracker) -> dict:
    """Train, validate every val_every steps, and return the final and selected metrics."""
    set_input_kind(config.input_kind)
    seed_everything(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = config.resolved_data_root()
    records = read_pairs(root / "pairs_real.jsonl")
    contours = load_contours(root, records, config.shift_semitones)
    dataset = training_dataset(root, records, contours, config)
    sets = validation_sets(records, contours, config)
    held_out = set(song_holdout(config)["songs"])
    validators = [MLEndValidation(root, "val", exclude_songs=held_out)] if config.mlend_val else []
    validators += [chad.ChadValidation(root)] if config.chad_val else []
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        collate_fn=collate_pairs,
        persistent_workers=config.num_workers > 0,
    )
    song_stream = song_batches(root, config)
    model = build_model(config).to(device)
    load_initial_weights(model, config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    best: dict = {}
    last_metrics = run_validation(model, sets, device, 0, tracker, best, config, validators)
    started = time.time()
    stream = batches(loader, dataset)
    step = 0
    for step in range(1, config.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = config.lr * lr_multiplier(step, config.warmup_steps, config.steps)
        song_batch = next(song_stream) if song_stream is not None else None
        loss = train_step(model, next(stream), optimizer, device, config, song_batch)
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
            last_metrics = run_validation(
                model, sets, device, step, tracker, best, config, validators
            )
            if should_stop(best, step, config):
                LOGGER.info("early stop at step %s (best step %s)", step, best.get("step"))
                break
    save_checkpoint(config.resolved_ckpt_dir() / "last.pt", model, config, step, last_metrics)
    return {
        "last": last_metrics,
        "last_step": step,
        "selected_step": best.get("step"),
        "selected": best.get("metrics"),
    }
