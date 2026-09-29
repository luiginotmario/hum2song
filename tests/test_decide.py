"""Decision layer: score policy picks the branch; OpenRouter is never required for that."""

from hum2song.decide.features import softmax
from hum2song.decide.loop import decide_from_search
from hum2song.decide.policy import ACTION_FOLLOWUP, ACTION_RETRY, ACTION_SHOW


def hit(score: float, song_id: str = "s", title: str = "Song") -> dict:
    return {
        "song_id": song_id,
        "title": title,
        "artist": "A",
        "score": score,
        "best_start_s": 0.0,
    }


def test_softmax_peaks_on_the_winner():
    probs = softmax([3.0, 1.0, 1.0], temperature=0.5)
    assert probs[0] > 0.8
    assert abs(sum(probs) - 1.0) < 1e-6


def test_clear_winner_when_top_is_ahead():
    results = [hit(9.0, "a", "Alpha"), hit(5.0, "b", "Beta"), hit(4.5, "c", "Gamma")]
    decision = decide_from_search({"voiced_s": 6.0, "results": results})
    assert decision.action == ACTION_SHOW
    assert "Alpha" in decision.message
    assert decision.followup is None


def test_followup_when_top_few_are_close():
    results = [hit(6.0 + 0.05 * i, f"s{i}", f"Song {i}") for i in range(5, 0, -1)]
    decision = decide_from_search({"voiced_s": 6.0, "results": results})
    assert decision.action == ACTION_FOLLOWUP
    assert decision.followup is not None
    assert len(decision.followup["options"]) >= 3


def test_retry_when_silent_or_empty():
    assert decide_from_search({"voiced_s": 0.2, "results": [hit(9.0)]}).action == ACTION_RETRY
    assert decide_from_search({"voiced_s": 6.0, "results": []}).action == ACTION_RETRY


def test_retry_when_top_list_is_almost_flat():
    results = [hit(3.2 - 0.02 * i) for i in range(10)]
    decision = decide_from_search({"voiced_s": 6.0, "results": results})
    assert decision.action == ACTION_RETRY
