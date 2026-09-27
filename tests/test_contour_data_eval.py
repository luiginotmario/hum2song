import numpy as np
import torch
from torch.utils.data import DataLoader

from hum2song.contour.augment import ContourAugment
from hum2song.contour.data import (
    ContourPairDataset,
    PairWindows,
    SyntheticPairDataset,
    collate_pairs,
    seconds_to_frames,
    set_epoch,
)
from hum2song.contour.evaluate import (
    build_set,
    distractor_contours,
    mir_sets,
    score_set,
    sliding_windows,
    start_windows,
)
from hum2song.contour.model import ContourEncoder
from hum2song.manifest import PairRecord
from tests.support import write_quarter_note


def record(index: int, song: str) -> PairRecord:
    return PairRecord(
        pair_id=f"p{index}",
        query_path=f"q{index}.wav",
        qtype="hum",
        qsource="real",
        song_id=song,
        song_start_s=0.0,
        song_dur_s=None,
        split="train",
        group="humtrans",
        song_path=f"s{index}.wav",
    )


def contours(count: int = 4) -> tuple[dict, dict]:
    rng = np.random.default_rng(0)
    refs = {
        f"s{i}.wav": np.repeat(rng.integers(55, 70, size=40), 25).astype(np.float32)
        for i in range(count)
    }
    queries = {f"q{i}.wav": refs[f"s{i}.wav"] + 3.0 for i in range(count)}
    return queries, refs


def test_dataset_pairs_collate_with_masks():
    queries, refs = contours()
    records = [record(i, f"song{i}") for i in range(4)]
    dataset = ContourPairDataset(records, queries, refs, ContourAugment(), PairWindows(), seed=0)
    batch = collate_pairs([dataset[i] for i in range(4)])
    assert batch["query"].shape[0] == 4 and batch["query"].shape[2] == 2
    assert batch["query_valid"].any(dim=1).all() and batch["song_valid"].any(dim=1).all()
    assert batch["song_id"] == ["song0", "song1", "song2", "song3"]


def test_reference_windows():
    contour = np.full(seconds_to_frames(20.0), 60.0, dtype=np.float32)
    assert [len(w) for w in start_windows(contour, (6.0, 10.0))] == [300, 500]
    windows = sliding_windows(contour, (10.0,), 5.0)
    assert len(windows) == 3


def test_score_set_runs_end_to_end():
    queries, refs = contours()
    records = [record(i, f"song{i}") for i in range(4)]
    item = build_set("toy", records, queries, refs, lambda c: start_windows(c, (10.0,)))
    torch.manual_seed(0)
    model = ContourEncoder(dim=32, layers=1, heads=2, dropout=0.0, out_dim=16).eval()
    metrics = score_set(model, item, torch.device("cpu"))
    assert set(metrics) == {"val/toy_top1", "val/toy_top10", "val/toy_mrr"}
    assert metrics["val/toy_top10"] == 1.0


def test_distractors_join_the_reference_side(tmp_path):
    for name in ("a", "b", "c"):
        write_quarter_note(tmp_path / f"{name}.mid")
    distractors = distractor_contours(tmp_path, 2)
    assert sorted(distractors) == ["distractor:a", "distractor:c"]
    queries, refs = contours()
    records = [record(i, f"song{i}") for i in range(4)]
    sets = mir_sets(records, queries, refs, distractors)
    assert sets[0].name == "mir48+2_start10"
    assert sets[0].ref_ids.count("distractor:a") == 1
    assert len(set(sets[0].ref_ids)) == 6


def test_new_epoch_reaches_persistent_workers():
    queries, refs = contours()
    records = [record(i, f"song{i}") for i in range(4)]
    dataset = ContourPairDataset(records, queries, refs, ContourAugment(), PairWindows(), seed=0)
    loader = DataLoader(
        dataset, batch_size=4, num_workers=1, persistent_workers=True, collate_fn=collate_pairs
    )
    first = next(iter(loader))["query"]
    set_epoch(dataset, 1)
    second = next(iter(loader))["query"]
    assert first.shape != second.shape or not torch.equal(first, second)


def test_synthetic_pairs_use_the_melody_id():
    _queries, refs = contours()
    dataset = SyntheticPairDataset(refs, ContourAugment(), PairWindows(), seed=0)
    item = dataset[0]
    assert item["song_id"] == "s0.wav"
    assert item["query"].shape[1] == 2 and item["song"].shape[0] > 0
