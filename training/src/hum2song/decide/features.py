"""Numeric features from a ranked search response (SPEC §8)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DecisionFeatures:
    """Score-derived features. LLM phrasing may see titles; decisions use numbers only."""

    n_results: int
    voiced_s: float
    s1: float
    s2: float
    s5: float
    gap12: float
    gap15: float
    p_top1: float
    p_top5: float
    entropy_top10: float
    turn: int

    def as_dict(self) -> dict:
        return asdict(self)


def scores_of(results: list[dict]) -> list[float]:
    return [float(item.get("score") or 0.0) for item in results]


def softmax(values: list[float], temperature: float) -> list[float]:
    if not values:
        return []
    temp = max(float(temperature), 1e-6)
    peak = max(values)
    weights = [math.exp((value - peak) / temp) for value in values]
    total = sum(weights) or 1.0
    return [weight / total for weight in weights]


def entropy(probs: list[float]) -> float:
    return float(-sum(p * math.log(p) for p in probs if p > 0.0))


def score_temperature(scores: list[float], fraction: float) -> float:
    """Softmax temperature from the top-list spread so chunk cosine and window fuse both work."""
    if len(scores) < 2:
        return 1.0
    spread = scores[0] - scores[-1]
    return max(fraction * spread, 1e-3)


def extract_features(
    results: list[dict],
    voiced_s: float = 0.0,
    turn: int = 0,
    temperature: float = 0.35,
) -> DecisionFeatures:
    scores = scores_of(results)
    padded = scores + [0.0] * max(0, 5 - len(scores))
    temp = score_temperature(scores, temperature) if scores else 1.0
    probs = softmax(scores[:10], temp)
    p_top1 = probs[0] if probs else 0.0
    p_top5 = float(sum(probs[:5])) if probs else 0.0
    return DecisionFeatures(
        n_results=len(results),
        voiced_s=float(voiced_s),
        s1=padded[0],
        s2=padded[1],
        s5=padded[4],
        gap12=padded[0] - padded[1],
        gap15=padded[0] - padded[4],
        p_top1=p_top1,
        p_top5=p_top5,
        entropy_top10=entropy(probs) if probs else 0.0,
        turn=int(turn),
    )
