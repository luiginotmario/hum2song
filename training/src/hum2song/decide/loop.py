"""Decide show / follow-up / retry from a search response (SPEC §8, D-032).

Melody rank scores choose the branch. OpenRouter only phrases text when the key is set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from hum2song.decide import phrases
from hum2song.decide.config import DecideConfig, load_decide_config
from hum2song.decide.features import DecisionFeatures, extract_features
from hum2song.decide.policy import ACTION_FOLLOWUP, ACTION_RETRY, ACTION_SHOW, choose_action


@dataclass(frozen=True)
class Decision:
    action: str
    message: str
    results: list[dict]
    followup: dict | None
    features: dict
    source: str  # "local" (score policy); OpenRouter never picks the song

    def as_dict(self) -> dict:
        return asdict(self)


def decide_from_search(
    search: dict,
    *,
    turn: int = 0,
    config: DecideConfig | None = None,
) -> Decision:
    """Branch on voiced_s + ranked `results` from catalog.search.search_audio."""
    config = config or load_decide_config()
    results = list(search.get("results") or [])
    voiced_s = float(search.get("voiced_s") or 0.0)
    features = extract_features(
        results, voiced_s=voiced_s, turn=turn, temperature=config.temperature
    )
    action = choose_action(features, config)
    return _build(action, results, features, config)


def _build(
    action: str, results: list[dict], features: DecisionFeatures, config: DecideConfig
) -> Decision:
    if action == ACTION_SHOW:
        winner = results[0]
        return Decision(
            action=ACTION_SHOW,
            message=phrases.phrase_show(config, winner),
            results=results,
            followup=None,
            features=features.as_dict(),
            source="local",
        )
    if action == ACTION_FOLLOWUP:
        top = results[: config.followup_n]
        question, options = phrases.phrase_followup(config, top)
        return Decision(
            action=ACTION_FOLLOWUP,
            message=question,
            results=results,
            followup={"question": question, "options": options},
            features=features.as_dict(),
            source="local",
        )
    return Decision(
        action=ACTION_RETRY,
        message=phrases.phrase_retry(config, features.voiced_s),
        results=results,
        followup=None,
        features=features.as_dict(),
        source="local",
    )
