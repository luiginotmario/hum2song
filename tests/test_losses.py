"""InfoNCE masks same-song false negatives and stays finite."""

import torch

from hum2song.losses import MASK_LOGIT, info_nce_symmetric, masked_logits


def test_same_song_off_diagonal_is_masked() -> None:
    vectors = torch.eye(2)
    logits = masked_logits(vectors, vectors, ["song", "song"], torch.tensor(1.0))
    assert float(logits[0, 1]) == MASK_LOGIT
    assert float(logits[1, 0]) == MASK_LOGIT
    assert float(logits[0, 0]) != MASK_LOGIT


def test_aligned_batch_has_finite_loss() -> None:
    vectors = torch.eye(4)
    song_ids = ["a", "b", "c", "d"]
    loss = info_nce_symmetric(vectors, vectors, song_ids, torch.tensor(0.07))
    assert torch.isfinite(loss)
    assert float(loss) < 0.1
