import torch

from hum2song.contour.config import ContourConfig
from hum2song.contour.train import build_model, load_initial_weights, save_checkpoint, should_stop


def tiny_config(tmp_path, **overrides) -> ContourConfig:
    fields = {"dim": 32, "layers": 1, "heads": 2, "out_dim": 16, "data_root": str(tmp_path)}
    return ContourConfig(**{**fields, **overrides})


def test_should_stop_after_patience_validations():
    config = ContourConfig(val_every=100, early_stop_patience=3)
    assert not should_stop({"step": 200}, 400, config)
    assert should_stop({"step": 200}, 500, config)
    assert not should_stop({"step": 0}, 10_000, ContourConfig(early_stop_patience=0))


def test_load_initial_weights_copies_checkpoint(tmp_path):
    source = build_model(tiny_config(tmp_path, seed=1))
    save_checkpoint(tmp_path / "ckpt" / "a.pt", source, tiny_config(tmp_path), 5, {})
    torch.manual_seed(123)
    target = build_model(tiny_config(tmp_path, init_ckpt="ckpt/a.pt"))
    load_initial_weights(target, tiny_config(tmp_path, init_ckpt="ckpt/a.pt"), torch.device("cpu"))
    for a, b in zip(source.state_dict().values(), target.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_no_init_ckpt_leaves_model_untouched(tmp_path):
    model = build_model(tiny_config(tmp_path))
    before = [p.clone() for p in model.parameters()]
    load_initial_weights(model, tiny_config(tmp_path), torch.device("cpu"))
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
