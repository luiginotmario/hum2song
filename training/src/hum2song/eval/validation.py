"""In-training validation: MIR-QBSH against its 48 targets only, and HumTrans val.

The MIR-QBSH line ranks every test query against the 48 MIR-QBSH references and
nothing else, so it can be compared across runs without a distractor set. HumTrans
val is also scored with each query transposed by val_shift_semitones, because its
references share the hum's key and an unshifted score cannot show key robustness.
Clips are loaded once, on first use, and kept in memory as float16.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from hum2song.eval.metrics import ranks_for_queries, retrieval_scores
from hum2song.eval.run_eval import load_fitted_waves
from hum2song.logutil import get_logger
from hum2song.manifest import PairRecord

LOGGER = get_logger(__name__)
MIR_GROUP = "mirqbsh"
HUMTRANS_GROUP = "humtrans"
REPORTED_METRICS = ("top1", "top10", "mrr")


@dataclass
class ValidationSet:
    """Cached eval crops for one retrieval task."""

    name: str
    queries: torch.Tensor
    query_ids: list[str]
    refs: torch.Tensor
    ref_ids: list[str]


class Validator:
    """Scores the model on the validation sets. Builds them lazily on the first call."""

    def __init__(self, records: list[PairRecord], data_root: Path, config, workers: int) -> None:
        self.records = records
        self.data_root = Path(data_root)
        self.config = config
        self.workers = workers
        self.sets: list[ValidationSet] | None = None

    def run(self, model: torch.nn.Module, device: torch.device, autocast=nullcontext) -> dict:
        """Return val/<set>_<metric> scalars. Leaves the model in eval mode."""
        if self.sets is None:
            self.sets = build_validation_sets(
                self.records, self.data_root, self.config, self.workers
            )
        model.eval()
        metrics: dict[str, float] = {}
        for item in self.sets:
            metrics.update(score_set(model, item, device, self.config.val_batch_size, autocast))
        LOGGER.info("validation %s", " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        return metrics


def build_validation_sets(
    records: list[PairRecord],
    data_root: Path,
    config,
    workers: int,
) -> list[ValidationSet]:
    """MIR-QBSH test vs its 48 targets, HumTrans val, and HumTrans val transposed."""
    mir = _with_reference(records, MIR_GROUP, "test")
    humtrans = _with_reference(records, HUMTRANS_GROUP, "val")
    loader = _Loader(data_root, config, workers)
    sets = [loader.build("mir48", mir)]
    humtrans_set = loader.build("humtrans_val", humtrans)
    shift = config.val_shift_semitones
    sets.extend([humtrans_set, loader.build(f"humtrans_val_shift{shift:+g}", humtrans, shift)])
    return [item for item in sets if item is not None]


def score_set(
    model: torch.nn.Module,
    item: ValidationSet,
    device: torch.device,
    batch_size: int,
    autocast=nullcontext,
) -> dict[str, float]:
    query_emb = embed(model, item.queries, device, batch_size, autocast)
    ref_emb = embed(model, item.refs, device, batch_size, autocast)
    scores = retrieval_scores(ranks_for_queries(query_emb, item.query_ids, ref_emb, item.ref_ids))
    values = scores.as_dict()
    return {f"val/{item.name}_{metric}": float(values[metric]) for metric in REPORTED_METRICS}


def embed(
    model: torch.nn.Module,
    waves: torch.Tensor,
    device: torch.device,
    batch_size: int,
    autocast=nullcontext,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with torch.inference_mode(), autocast():
        for start in range(0, waves.shape[0], batch_size):
            batch = waves[start : start + batch_size].to(device).float()
            chunks.append(model(batch).embedding.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


class _Loader:
    def __init__(self, data_root: Path, config, workers: int) -> None:
        self.data_root = data_root
        self.config = config
        self.workers = workers
        self.cache: dict[tuple[str, float], torch.Tensor] = {}

    def build(
        self,
        name: str,
        records: list[PairRecord],
        semitones: float = 0.0,
    ) -> ValidationSet | None:
        if not records:
            LOGGER.warning("validation set %s is empty; skipped", name)
            return None
        refs, ref_ids = _unique_references(records)
        return ValidationSet(
            name=name,
            queries=self._stack([record.query_path for record in records], semitones),
            query_ids=[record.song_id for record in records],
            refs=self._stack(refs, 0.0),
            ref_ids=ref_ids,
        )

    def _stack(self, paths: list[str], semitones: float) -> torch.Tensor:
        key = ("\n".join(paths), semitones)
        if key not in self.cache:
            waves = load_fitted_waves(
                paths,
                self.data_root,
                self.config.sample_rate,
                self.config.crop_seconds,
                self.workers,
                semitones,
            )
            self.cache[key] = torch.stack(waves).to(torch.float16)
        return self.cache[key]


def _with_reference(records: list[PairRecord], group: str, split: str) -> list[PairRecord]:
    chosen = [r for r in records if r.group == group and r.split == split and r.song_path]
    return sorted(chosen, key=_pair_id)


def _unique_references(records: list[PairRecord]) -> tuple[list[str], list[str]]:
    by_path = {str(record.song_path): record.song_id for record in records}
    paths = sorted(by_path)
    return paths, [by_path[path] for path in paths]


def _pair_id(record: PairRecord) -> str:
    return record.pair_id
