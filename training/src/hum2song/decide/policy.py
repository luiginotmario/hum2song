"""Local threshold policy: melody scores decide the branch (SPEC fallback / primary)."""

from __future__ import annotations

from hum2song.decide.config import DecideConfig
from hum2song.decide.features import DecisionFeatures

ACTION_SHOW = "show"
ACTION_FOLLOWUP = "ask_followup"
ACTION_RETRY = "ask_retry"


def relative_gap12(features: DecisionFeatures) -> float:
    """How much of the top-list spread sits between #1 and #2 (0 = tied, 1 = alone at the top)."""
    spread = features.s1 - min(features.s5, features.s2) if features.n_results else 0.0
    # Prefer full list spread when we have s5; fall back to gap12 itself.
    if features.n_results >= 5:
        spread = features.gap15 if features.gap15 > 0 else features.gap12
    elif features.n_results >= 2:
        spread = features.gap12
    if spread <= 1e-9:
        return 0.0
    return float(features.gap12 / spread)


def choose_action(features: DecisionFeatures, config: DecideConfig) -> str:
    """Pick show / ask_followup / ask_retry from score features only. No LLM."""
    if features.voiced_s < config.min_voiced_s or features.n_results == 0:
        return ACTION_RETRY
    clear = (
        features.n_results >= config.min_results_for_show
        and features.p_top1 >= config.t_show
        and relative_gap12(features) >= config.show_min_rel_gap
    )
    if clear:
        return ACTION_SHOW
    if features.entropy_top10 >= config.retry_min_entropy:
        return ACTION_RETRY
    if (
        features.turn < config.max_turns
        and features.p_top5 >= config.t_few
        and features.n_results >= 2
    ):
        return ACTION_FOLLOWUP
    return ACTION_RETRY
