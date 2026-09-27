import numpy as np
import torch

from hum2song.contour.data import collate_contours, features_tensor
from hum2song.contour.model import ContourEncoder


def tiny_model() -> ContourEncoder:
    torch.manual_seed(0)
    return ContourEncoder(dim=32, layers=1, heads=2, dropout=0.0, out_dim=16).eval()


def test_embeddings_are_unit_length():
    features, valid = collate_contours(
        [features_tensor(np.full(30, 60.0)), features_tensor(np.full(51, 62.0))]
    )
    embedding = tiny_model()(features, valid)
    assert embedding.shape == (2, 16)
    assert torch.allclose(embedding.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_padding_does_not_change_the_embedding():
    model = tiny_model()
    contour = 60.0 + np.arange(40) % 7
    alone = model(*collate_contours([features_tensor(contour)]))
    padded = model(
        *collate_contours([features_tensor(contour), features_tensor(np.full(90, 65.0))])
    )
    assert torch.allclose(alone[0], padded[0], atol=1e-4)


def test_transposed_contours_embed_identically():
    model = tiny_model()
    contour = 60.0 + np.arange(40) % 5
    first = model(*collate_contours([features_tensor(contour)]))
    second = model(*collate_contours([features_tensor(contour + 5.0)]))
    assert torch.allclose(first, second, atol=1e-6)
