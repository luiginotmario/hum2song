"""Load configs/decide.yaml (thresholds + OpenRouter model slugs)."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from hum2song.config import DEFAULT_DATA_ROOT


@dataclass
class DecideConfig:
    temperature: float = 0.35
    t_show: float = 0.42
    show_min_rel_gap: float = 0.40
    t_few: float = 0.70
    min_voiced_s: float = 1.0
    retry_min_entropy: float = 1.85
    min_results_for_show: int = 1
    followup_n: int = 3
    max_turns: int = 2
    llm_model: str = "openai/gpt-4o-mini"
    jev_model: str = ""
    openrouter_timeout_s: float = 8.0
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    @property
    def api_key(self) -> str:
        return os.environ.get("OPENROUTER_API_KEY", "").strip()

    @property
    def openrouter_ready(self) -> bool:
        return bool(self.api_key)


def default_decide_path() -> Path:
    root = Path(os.environ.get("H2S_DATA", DEFAULT_DATA_ROOT))
    repo = Path(__file__).resolve().parents[4]
    for path in (
        Path("configs/decide.yaml"),
        repo / "configs" / "decide.yaml",
        root / "dev" / "hum2song" / "configs" / "decide.yaml",
        root / "hum2song" / "configs" / "decide.yaml",
    ):
        if path.is_file():
            return path
    return Path("configs/decide.yaml")


def load_decide_config(path: Path | None = None) -> DecideConfig:
    payload: dict = {}
    config_path = path or default_decide_path()
    if config_path.is_file():
        payload.update(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
    known = {item.name for item in fields(DecideConfig)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"unknown decide config keys: {unknown}")
    return DecideConfig(**payload)
