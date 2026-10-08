"""CPU smoke render: one whistle file and one wrong-lyrics file."""

import json
from pathlib import Path

import numpy as np

from hum2song.audio import write_wav
from hum2song.contour.augment import ContourAugment
from hum2song.contour.features import FRAME_S
from hum2song.contour.whistle import SAMPLE_RATE
from hum2song.query_synth.render import (
    ORIGIN_SYNTHETIC,
    PRESENTATION,
    STYLE_WHISTLE,
    STYLE_WRONG_LYRICS,
    render_control,
    render_query,
)
from hum2song.query_synth.signal import component_ratio_db, semitone_to_hz
from hum2song.query_synth.track import best_median_abs

SMOKE_AUGMENT = ContourAugment(
    stretch_min=0.8,
    stretch_max=1.2,
    warp_depth=0.15,
    interval_scale=0.1,
    drift_semitones=0.4,
    jitter_semitones=0.1,
    dropout_prob=0.3,
    octave_error_prob=0.0,
)


def demo_contour() -> np.ndarray:
    """A short invented tune. Not a recording, and not taken from a song lyric."""
    notes = [67, 67, 69, 71, 69, 67, 66, 64, 64, 66, 67, 67]
    hold = int(round(0.28 / FRAME_S))
    gap = int(round(0.06 / FRAME_S))
    parts = []
    for note in notes:
        parts.append(np.full(hold, float(note), dtype=np.float32))
        parts.append(np.full(gap, np.nan, dtype=np.float32))
    return np.concatenate(parts)


def run_smoke(dest: Path) -> dict:
    """Write whistle.wav and wrong_lyrics.wav plus smoke.json. Returns the report."""
    dest.mkdir(parents=True, exist_ok=True)
    contour = demo_contour()
    whistle = render_query(contour, np.random.default_rng(0), STYLE_WHISTLE, SMOKE_AUGMENT)
    singing = render_query(contour, np.random.default_rng(1), STYLE_WRONG_LYRICS, SMOKE_AUGMENT)
    write_wav(dest / "whistle.wav", whistle.audio, whistle.sample_rate)
    write_wav(dest / "wrong_lyrics.wav", singing.audio, singing.sample_rate)
    probe = _timbre_probe()
    report = {
        "origin": ORIGIN_SYNTHETIC,
        "presentation": PRESENTATION,
        "files": ["whistle.wav", "wrong_lyrics.wav"],
        "whistle": _style_report(whistle, probe["whistle_h2_db"]),
        "wrong_lyrics": _style_report(singing, probe["singing_h2_db"]),
        "timbre_probe": probe,
    }
    (dest / "smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _style_report(rendered, probe_h2: float) -> dict:
    voiced = rendered.control[~np.isnan(rendered.control)]
    median_hz = float(semitone_to_hz(np.median(voiced))) if len(voiced) else None
    return {
        "origin": rendered.origin,
        "style": rendered.style,
        "seconds": round(len(rendered.audio) / rendered.sample_rate, 3),
        "median_hz": median_hz,
        "tracker_median_abs_st": best_median_abs(rendered.control, rendered.tracked),
        "syllables": list(rendered.syllables),
        "h2_db_drawn": rendered.h2_db,
        "probe_h2_db": probe_h2,
        "peak": float(np.max(np.abs(rendered.audio))) if len(rendered.audio) else 0.0,
    }


def _timbre_probe() -> dict:
    """Steady tones, no room, so the harmonic ratio is the oscillator and not the noise."""
    whistle_f0 = np.full(int(1.5 / FRAME_S), 88.0, dtype=np.float32)
    sing_f0 = np.full(int(1.5 / FRAME_S), 60.0, dtype=np.float32)
    whistle = render_control(whistle_f0, np.random.default_rng(2), STYLE_WHISTLE, room=False)
    singing = render_control(sing_f0, np.random.default_rng(3), STYLE_WRONG_LYRICS, room=False)
    return {
        "whistle_h2_db": component_ratio_db(whistle.audio, SAMPLE_RATE, 1318.5, 2),
        "singing_h2_db": component_ratio_db(singing.audio, SAMPLE_RATE, 261.6, 2),
        "whistle_hz": 1318.5,
        "singing_hz": 261.6,
    }
