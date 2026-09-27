"""Stage A training loop: AdamW, mixed precision, checkpoints, resume."""

import math
import subprocess
import time
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hum2song.config import TrainConfig, dump_config
from hum2song.data.dataset import PairDataset, collate_pairs, worker_init
from hum2song.eval.validation import Validator
from hum2song.logutil import get_logger
from hum2song.losses import (
    batch_ece,
    confidence_loss,
    info_nce_symmetric,
    masked_logits,
    qtype_loss,
)
from hum2song.manifest import QTYPES, file_sha256, read_pairs
from hum2song.seed import seed_everything
from hum2song.tracking import NullTracker

LOGGER = get_logger(__name__)


def run_training(config: TrainConfig, model: torch.nn.Module, tracker=None) -> Path:
    """Train and return the path of the last checkpoint."""
    _check_stage(config.stage)
    seed_everything(config.seed)
    torch.manual_seed(config.seed)
    device = _pick_device()
    model.to(device)
    ckpt_dir = config.resolved_ckpt_dir()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, ckpt_dir / "config.yaml")
    records = _load_train_records(config)
    if records is None:
        return ckpt_dir / "last.pt"
    dataset = PairDataset(records, config.resolved_data_root(), config, training=True)
    loader = _make_loader(dataset, config, device)
    steps_per_epoch = max(len(loader), 1)
    optimizer = torch.optim.AdamW(
        _param_groups(model, config),
        weight_decay=config.weight_decay,
    )
    if config.init_weights:
        _load_weights(Path(config.init_weights), model)
    start = 0
    if config.resume:
        start = _load_resume(Path(config.resume), model, optimizer)
    end = config.steps
    if config.dry_run:
        end = min(config.steps, start + 1)
    if start >= end:
        LOGGER.info("checkpoint already at step %s; target is %s", start, end)
        return ckpt_dir / "last.pt"
    use_amp, amp_dtype = _amp_settings(device, config.precision)
    if device.type != "cuda" and config.precision != "fp32":
        LOGGER.info("mixed precision is disabled on CPU; running fp32")
    scaler = _make_scaler(use_amp and amp_dtype == torch.float16)
    manifest = config.resolved_manifest()
    manifest_sha = file_sha256(manifest) if manifest.exists() else ""
    active = tracker or NullTracker()
    batches = _cycle(loader)
    validator = _make_validator(config)
    autocast = _autocast_factory(device, use_amp, amp_dtype)
    current_top: int | None = None
    clock = _StepClock(start)
    for step in range(start, end):
        if _should_validate(step, start, end, config):
            active.log(validator.run(model, device, autocast), step)
            clock.restart(step)
        n_top = _trainable_top(step, config)
        if n_top != current_top:
            model.set_trainable_top(n_top)
            current_top = n_top
            LOGGER.info("trainable top layers set to %s at step %s", n_top, step)
        model.train()
        _set_lr(optimizer, step, config)
        optimizer.zero_grad(set_to_none=True)
        metric_rows: list[dict[str, float]] = []
        for _micro in range(max(config.grad_accum_steps, 1)):
            dataset.set_epoch(step // steps_per_epoch)
            batch = _move_batch(next(batches), device)
            loss, metrics = _forward_loss(model, batch, config, device, use_amp, amp_dtype)
            _backward(loss / max(config.grad_accum_steps, 1), scaler)
            metric_rows.append(metrics)
        _optimizer_step(model, optimizer, scaler, config.grad_clip)
        completed = step + 1
        averaged = _average(metric_rows)
        if completed % config.log_every == 0 or completed == end:
            averaged["steps_per_sec"] = clock.rate(completed)
            LOGGER.info("step %s %s", completed, _format_metrics(averaged))
            active.log(averaged, completed)
        if completed % config.save_every == 0 or completed == end:
            _save_checkpoint(
                ckpt_dir / f"step_{completed}.pt",
                ckpt_dir / "last.pt",
                model,
                optimizer,
                completed,
                config,
                manifest_sha,
                n_top,
            )
    if _should_validate(end, start, end, config):
        active.log(validator.run(model, device, autocast), end)
    active.finish()
    return ckpt_dir / "last.pt"


class _StepClock:
    """Optimizer steps per second since the previous log line."""

    def __init__(self, step: int) -> None:
        self.restart(step)

    def restart(self, step: int) -> None:
        """Start timing from `step`, e.g. after a validation pass."""
        self.step = step
        self.time = time.perf_counter()

    def rate(self, step: int) -> float:
        now = time.perf_counter()
        elapsed = max(now - self.time, 1.0e-9)
        value = (step - self.step) / elapsed
        self.step = step
        self.time = now
        return value


def _make_validator(config: TrainConfig) -> Validator | None:
    if config.val_every <= 0:
        return None
    records = read_pairs(config.resolved_manifest())
    workers = config.resolved_num_workers()
    return Validator(records, config.resolved_data_root(), config, workers)


def _should_validate(step: int, start: int, end: int, config: TrainConfig) -> bool:
    """Before the first step of a run, every val_every steps, and after the last step."""
    if config.val_every <= 0 or config.dry_run:
        return False
    return step in (start, end) or step % config.val_every == 0


def _make_loader(dataset, config: TrainConfig, device: torch.device) -> DataLoader:
    workers = config.resolved_num_workers()
    kwargs = {
        "batch_size": config.batch_size,
        "shuffle": True,
        "num_workers": workers,
        "collate_fn": collate_pairs,
        "drop_last": False,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["worker_init_fn"] = worker_init
    LOGGER.info(
        "data loader workers %s pin_memory %s persistent_workers %s",
        workers,
        kwargs["pin_memory"],
        workers > 0,
    )
    return DataLoader(dataset, **kwargs)


def _check_stage(stage: str) -> None:
    if stage == "A":
        return
    raise NotImplementedError(
        f"stage {stage} is not implemented. Stage B/C need synthetic or aligned pairs "
        "from hum2song.synth, which is a Phase 2 stub."
    )


def _load_train_records(config: TrainConfig):
    manifest = config.resolved_manifest()
    if not manifest.exists():
        message = f"manifest not found at {manifest}"
        if config.dry_run:
            LOGGER.warning(message)
            return None
        raise FileNotFoundError(message)
    records = [record for record in read_pairs(manifest) if record.split == "train"]
    records = sorted(records, key=_pair_id)
    if config.limit is not None:
        records = records[: config.limit]
    if records:
        LOGGER.info("train query_type counts %s", _query_type_counts(records))
        return records
    message = (
        "no train rows. Stage A trains on HumTrans and on MIR-QBSH songs that "
        "have sung clips. MTG-QBH, MLEnd, and the other MIR-QBSH songs stay in test."
    )
    if config.dry_run:
        LOGGER.warning(message)
        return None
    raise RuntimeError(message)


def _pair_id(record) -> str:
    return record.pair_id


def _query_type_counts(records: list) -> dict[str, int]:
    counts = {name: 0 for name in QTYPES}
    for record in records:
        counts[record.query_type] = counts.get(record.query_type, 0) + 1
    return counts


def _cycle(loader: DataLoader) -> Iterator:
    while True:
        yield from loader


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        "query": batch["query"].to(device),
        "song": batch["song"].to(device),
        "qtype": batch["qtype"].to(device),
        "song_id": batch["song_id"],
    }


def _forward_loss(model, batch, config, device, use_amp: bool, amp_dtype):
    with _autocast(device, use_amp, amp_dtype):
        both = torch.cat([batch["query"], batch["song"]], dim=0)
        output = model(both)
    batch_size = batch["query"].shape[0]
    query_emb = output.embedding[:batch_size].float()
    song_emb = output.embedding[batch_size:].float()
    qtype_logits = output.qtype_logits[:batch_size].float()
    temperature = output.temperature.float()
    nce = info_nce_symmetric(query_emb, song_emb, batch["song_id"], temperature)
    qtype = qtype_loss(qtype_logits, batch["qtype"])
    logits = masked_logits(query_emb, song_emb, batch["song_id"], temperature)
    conf = confidence_loss(logits, config.conf_gamma)
    total = nce + config.qtype_loss_weight * qtype + config.conf_loss_weight * conf
    metrics = {
        "loss": float(total.detach()),
        "nce": float(nce.detach()),
        "qtype": float(qtype.detach()),
        "conf": float(conf.detach()),
        "ece": batch_ece(logits),
        "temperature": float(temperature.detach()),
    }
    return total, metrics


def _backward(loss: torch.Tensor, scaler) -> None:
    if scaler is None:
        loss.backward()
        return
    scaler.scale(loss).backward()


def _optimizer_step(model, optimizer, scaler, grad_clip: float) -> None:
    if scaler is not None:
        scaler.unscale_(optimizer)
    trainable = [param for param in model.parameters() if param.grad is not None]
    if trainable:
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
    if scaler is None:
        optimizer.step()
        return
    scaler.step(optimizer)
    scaler.update()


def _param_groups(model: torch.nn.Module, config: TrainConfig) -> list[dict]:
    head = [param for name, param in model.named_parameters() if not name.startswith("encoder.")]
    backbone = [param for name, param in model.named_parameters() if name.startswith("encoder.")]
    return [
        {"params": head, "lr": config.lr_head},
        {"params": backbone, "lr": config.lr_backbone},
    ]


def _set_lr(optimizer, step: int, config: TrainConfig) -> None:
    scale = _lr_multiplier(step, config.warmup_steps, config.steps)
    optimizer.param_groups[0]["lr"] = config.lr_head * scale
    optimizer.param_groups[1]["lr"] = config.lr_backbone * scale


def _lr_multiplier(step: int, warmup: int, total: int) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / float(warmup)
    span = max(total - warmup, 1)
    progress = min(max((step - warmup) / float(span), 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _trainable_top(step: int, config: TrainConfig) -> int:
    if step < config.freeze_steps:
        return 0
    return config.unfreeze_top_layers


def _amp_settings(device: torch.device, precision: str) -> tuple[bool, torch.dtype | None]:
    if device.type != "cuda" or precision == "fp32":
        return False, None
    if precision == "fp16":
        return True, torch.float16
    return True, torch.bfloat16


def _autocast_factory(device: torch.device, use_amp: bool, amp_dtype):
    def factory():
        return _autocast(device, use_amp, amp_dtype)

    return factory


def _autocast(device: torch.device, use_amp: bool, amp_dtype):
    if not use_amp or amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def _make_scaler(enabled: bool):
    if not enabled:
        return None
    return torch.cuda.amp.GradScaler(enabled=True)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _save_checkpoint(
    step_path: Path,
    last_path: Path,
    model,
    optimizer,
    step: int,
    config: TrainConfig,
    manifest_sha: str,
    n_top: int,
) -> None:
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "git_sha": _git_sha(),
        "manifest_sha256": manifest_sha,
        "trainable_top": n_top,
    }
    _torch_save(step_path, payload)
    _torch_save(last_path, payload)
    LOGGER.info("saved %s", last_path)


def _torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_weights(path: Path, model) -> None:
    """Start from a checkpoint's model weights only: fresh optimizer, step 0, new schedule."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    LOGGER.info("initialized weights from %s (trained %s steps)", path, payload.get("step"))


def _load_resume(path: Path, model, optimizer) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    try:
        optimizer.load_state_dict(payload["optimizer"])
    except (ValueError, RuntimeError) as exc:
        LOGGER.warning("optimizer state not restored: %s", exc)
    LOGGER.info("resumed from %s at step %s", path, payload["step"])
    return int(payload["step"])


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _average(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def _format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
