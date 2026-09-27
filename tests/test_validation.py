"""In-training validation scores MIR-QBSH (48 targets only) and HumTrans val."""

from pathlib import Path

import torch

from hum2song.config import TrainConfig
from hum2song.eval.validation import Validator
from hum2song.manifest import PairRecord
from hum2song.model import RetrievalModel, TinyEncoder
from tests.support import write_tone


def _record(root: Path, name: str, group: str, split: str, frequency: float) -> PairRecord:
    write_tone(root / f"q/{name}.wav", frequency, seconds=0.5)
    write_tone(root / f"c/{name}.wav", frequency, seconds=0.5)
    return PairRecord(
        pair_id=f"{group}:{name}",
        query_path=f"q/{name}.wav",
        qtype="hum",
        qsource="real",
        song_id=f"{group}:{name}",
        song_start_s=0.0,
        song_dur_s=0.5,
        split=split,
        group=group,
        song_path=f"c/{name}.wav",
        title=None,
    )


def test_validator_reports_every_set(tmp_path: Path) -> None:
    records = [
        _record(tmp_path, "m1", "mirqbsh", "test", 200.0),
        _record(tmp_path, "m2", "mirqbsh", "test", 300.0),
        _record(tmp_path, "h1", "humtrans", "val", 250.0),
        _record(tmp_path, "h2", "humtrans", "val", 350.0),
        _record(tmp_path, "t1", "humtrans", "train", 400.0),
    ]
    config = TrainConfig(crop_seconds=0.25, val_batch_size=2, val_shift_semitones=7.0)
    model = RetrievalModel(TinyEncoder(2, 32), hidden_dim=32, n_layers=2, proj_dim=32)
    validator = Validator(records, tmp_path, config, workers=0)
    metrics = validator.run(model, torch.device("cpu"))
    expected = {
        f"val/{name}_{metric}"
        for name in ("mir48", "humtrans_val", "humtrans_val_shift+7")
        for metric in ("top1", "top10", "mrr")
    }
    assert set(metrics) == expected
    assert metrics["val/mir48_top10"] == 1.0
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
    assert validator.sets is not None
    assert validator.sets[0].queries.dtype == torch.float16
