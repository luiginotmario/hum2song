"""Shipped YAML configs load, and unknown keys are rejected."""

from pathlib import Path

import pytest

from hum2song.config import load_eval_config, load_train_config

ROOT = Path(__file__).resolve().parents[1]


def test_stage_a_yaml_has_the_training_hyperparameters() -> None:
    config = load_train_config(ROOT / "configs" / "train_stage_a.yaml")
    assert config.stage == "A"
    assert config.encoder == "mert"
    assert config.mert_name == "m-a-p/MERT-v1-95M"
    assert config.lr_head == pytest.approx(1.0e-3)
    assert config.lr_backbone == pytest.approx(1.0e-5)
    assert config.unfreeze_top_layers == 6
    assert config.temperature_init == pytest.approx(0.07)
    assert config.precision == "bf16"


def test_eval_yaml_names_mirqbsh_and_can_be_overridden(tmp_path: Path) -> None:
    config = load_eval_config(
        ROOT / "configs" / "eval.yaml",
        {"out": str(tmp_path / "report.json"), "sets": "mirqbsh"},
    )
    assert config.sets == "mirqbsh"
    assert config.resolved_out() == tmp_path / "report.json"


def test_unknown_yaml_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("stage: A\nnot_a_field: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_train_config(path)
