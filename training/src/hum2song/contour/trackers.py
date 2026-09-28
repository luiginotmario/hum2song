"""F0 trackers that produce the shared (2, frames) track layout at a 10 ms hop.

Row 0 is F0 in Hz, row 1 a voicing confidence (features.CONFIDENCE_THRESHOLD applies).
  rmvpe       RMVPE (vendored RVC build, docs/paper/scripts/rmvpe_rvc.py) on 16 kHz audio.
  rmvpe_half  RMVPE on audio played at half speed, F0 doubled (D-007's "RMVPE-fix" for
              whistles): 32 kHz samples are fed as if they were 16 kHz.
  peak        Spectral peak with the tonal-ratio gate (whistle.py, D-007), 32 kHz audio.
"""

import sys
from pathlib import Path

import numpy as np
import torch

from hum2song.contour.features import salience_to_f0
from hum2song.contour.whistle import spectral_peak_track

RMVPE_DIR = Path(__file__).resolve().parents[4] / "docs" / "paper" / "scripts"
RMVPE_SAMPLE_RATE = 16000
SAMPLE_RATES = {"rmvpe": 16000, "rmvpe_half": 32000, "peak": 32000}
RMVPE_HOP = 160
# cuDNN's GRU rejects very long sequences (a >10 min YouTube upload crashed D-019's
# extraction); longer audio is tracked in segments. Every earlier track was shorter.
MAX_SEGMENT_S = 540


def load_rmvpe(weights: Path, device: torch.device):
    """The vendored RVC RMVPE, built on CPU in fp32 and then moved to `device`."""
    sys.path.insert(0, str(RMVPE_DIR))
    from rmvpe_rvc import RMVPE

    model = RMVPE(str(weights), is_half=False, device="cpu")
    model.model = model.model.to(device)
    model.mel_extractor = model.mel_extractor.to(device)
    model.device = device
    return model


def rmvpe_salience(model, audio: np.ndarray) -> np.ndarray:
    """16 kHz audio -> (frames, bins) RMVPE salience, frames = len // hop + 1."""
    with torch.no_grad():
        mel = model.extract_mel(torch.from_numpy(audio.astype(np.float32)), center=True)
        return model.mel2hidden(mel).squeeze(0).float().cpu().numpy()


def segment_bounds(samples: int, segment: int) -> list[tuple[int, int]]:
    """Hop-aligned [start, end) sample ranges covering `samples`."""
    return [(start, min(start + segment, samples)) for start in range(0, samples, segment)]


def segmented_salience(model, audio: np.ndarray, max_s: float = MAX_SEGMENT_S) -> np.ndarray:
    """Salience in segments of at most max_s, frames matching one pass over the audio."""
    segment = int(max_s * RMVPE_SAMPLE_RATE) // RMVPE_HOP * RMVPE_HOP
    parts = [rmvpe_salience(model, audio[a:b]) for a, b in segment_bounds(len(audio), segment)]
    trimmed = [part[: segment // RMVPE_HOP] for part in parts[:-1]] + parts[-1:]
    return np.concatenate(trimmed)[: len(audio) // RMVPE_HOP + 1]


def rmvpe_track(model, audio: np.ndarray, max_s: float = MAX_SEGMENT_S) -> np.ndarray:
    """16 kHz audio -> (2, frames): F0 in Hz and RMVPE peak salience."""
    salience = segmented_salience(model, audio, max_s)
    f0, confidence = salience_to_f0(salience)
    return np.stack([f0, confidence])


def rmvpe_half_track(model, audio_32k: np.ndarray) -> np.ndarray:
    """32 kHz audio read as 16 kHz: pitch an octave down, twice as long. Undo both."""
    slowed = rmvpe_track(model, audio_32k)
    frames = slowed.shape[1] // 2
    halved = slowed[:, : frames * 2 : 2].copy()
    halved[0] *= 2.0
    return halved


def track_audio(method: str, model, audio: np.ndarray) -> np.ndarray:
    """Run one tracker on audio already at SAMPLE_RATES[method]."""
    if method == "peak":
        return spectral_peak_track(audio)
    if method == "rmvpe_half":
        return rmvpe_half_track(model, audio)
    if method == "rmvpe":
        return rmvpe_track(model, audio)
    raise ValueError(f"unknown tracker {method}; known: {sorted(SAMPLE_RATES)}")
