"""User-facing text: templates always work; OpenRouter may rewrite when the key is set."""

from __future__ import annotations

import json

from hum2song.decide.config import DecideConfig
from hum2song.decide.openrouter import OpenRouterError, chat


def song_label(item: dict) -> str:
    title = (item.get("title") or "").strip() or item.get("song_id", "this song")
    artist = (item.get("artist") or "").strip()
    return f"{title} — {artist}" if artist else title


def template_show(winner: dict) -> str:
    return f"Best match: {song_label(winner)}."


def template_followup(candidates: list[dict]) -> tuple[str, list[str]]:
    options = [song_label(item) for item in candidates]
    options.append("None of these / not sure")
    question = (
        "A few songs are close. Which of these sounds right, "
        "or do you remember a lyric, year, or artist?"
    )
    return question, options


def template_retry(voiced_s: float) -> str:
    if voiced_s < 1.0:
        return "I barely caught a melody. Try humming the chorus for about 10 seconds."
    return "I'm not sure yet. Try humming a clearer bit of the tune (chorus, ~10 s)."


def llm_rewrite(config: DecideConfig, system: str, user: str, fallback: str) -> str:
    if not config.openrouter_ready:
        return fallback
    try:
        text = chat(
            config,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
        ).strip()
        return text or fallback
    except OpenRouterError:
        return fallback


def phrase_show(config: DecideConfig, winner: dict) -> str:
    fallback = template_show(winner)
    return llm_rewrite(
        config,
        "You write one short sentence confirming a song match. Do not invent titles. No markdown.",
        f"Confirm this match briefly: {song_label(winner)}",
        fallback,
    )


def phrase_followup(config: DecideConfig, candidates: list[dict]) -> tuple[str, list[str]]:
    question, options = template_followup(candidates)
    if not config.openrouter_ready:
        return question, options
    listing = "\n".join(f"- {option}" for option in options[:-1])
    try:
        raw = chat(
            config,
            [
                {
                    "role": "system",
                    "content": (
                        "Return JSON {\"question\": str, \"options\": [str,...]}. "
                        "Ask ONE short follow-up. Do not reveal which song is likeliest. "
                        "Keep the given song labels as options, plus a not-sure option."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Candidates:\n{listing}\nAlso keep a not-sure option.",
                },
            ],
            json_mode=True,
        )
        data = json.loads(raw)
        q = str(data.get("question") or question).strip()
        opts = [str(item) for item in (data.get("options") or options) if str(item).strip()]
        return (q or question), (opts or options)
    except (OpenRouterError, json.JSONDecodeError, TypeError, ValueError):
        return question, options


def phrase_retry(config: DecideConfig, voiced_s: float) -> str:
    fallback = template_retry(voiced_s)
    return llm_rewrite(
        config,
        "You write one short tip asking the user to hum again. No markdown.",
        f"voiced_seconds={voiced_s:.2f}. Suggest a clearer ~10 s hum of the chorus.",
        fallback,
    )
