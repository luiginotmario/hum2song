"""Main-melody F0 from recorded songs, to build references from audio (docs/DECISIONS.md D-014).

Every method returns the extract_f0 track layout: (2, frames) at a 10 ms hop from time 0,
row 0 = F0 in Hz, row 1 = voicing confidence (features.CONFIDENCE_THRESHOLD applies).
  rmvpe_mix     RMVPE straight on the mixture (RMVPE is trained to ignore accompaniment)
  rmvpe_vocals  htdemucs vocal stem -> RMVPE
  fcpe_vocals   htdemucs vocal stem -> FCPE (torchfcpe), a fast monophonic tracker
  crepe_vocals  htdemucs vocal stem -> CREPE full (torchcrepe), confidence = periodicity
  melodia_mix   Melodia (essentia PredominantPitchMelodia), a dedicated melody extractor
Frame metrics use mir_eval's MIREX melody measures (RPA, RCA, VR, VFA, OA).
"""

import numpy as np
import torch

from hum2song.audio import resample_audio
from hum2song.contour.features import CONFIDENCE_THRESHOLD, RMVPE_HOP_S
from hum2song.contour.trackers import rmvpe_track

TRACK_RATE = 16000
SEPARATOR_NAME = "htdemucs"
SEPARATOR_RATE = 44100
MELODIA_RATE = 44100
MELODIA_HOP = 128
F0_RANGE_HZ = (50.0, 1100.0)
FCPE_THRESHOLD = 0.006
CREPE_BATCH = 2048
METHODS = ("rmvpe_mix", "rmvpe_vocals", "fcpe_vocals", "crepe_vocals", "melodia_mix")
VOCAL_METHODS = ("rmvpe_vocals", "fcpe_vocals", "crepe_vocals")
FRAME_METRICS = {
    "Raw Pitch Accuracy": "rpa",
    "Raw Chroma Accuracy": "rca",
    "Voicing Recall": "vr",
    "Voicing False Alarm": "vfa",
    "Overall Accuracy": "oa",
}


def load_separator(device: torch.device):
    from demucs.pretrained import get_model

    model = get_model(SEPARATOR_NAME)
    return model.to(device).eval()


def separate_vocals(separator, audio: np.ndarray, rate: int, device: torch.device) -> np.ndarray:
    """Mono audio at `rate` -> mono vocal stem at `rate` (demucs CLI normalization)."""
    from demucs.apply import apply_model

    upsampled = resample_audio(audio.astype(np.float32), rate, SEPARATOR_RATE)
    wav = torch.from_numpy(np.stack([upsampled, upsampled])).to(device)
    mean, std = wav.mean(), wav.std().clamp_min(1.0e-8)
    with torch.no_grad():
        stems = apply_model(separator, ((wav - mean) / std)[None], device=device, shifts=0)
    vocals = stems[0, separator.sources.index("vocals")] * std + mean
    mono = vocals.mean(dim=0).float().cpu().numpy()
    return resample_audio(mono, SEPARATOR_RATE, rate)


def frame_count(audio: np.ndarray, rate: int) -> int:
    return int(len(audio) / rate / RMVPE_HOP_S) + 1


def fcpe_track(model, audio: np.ndarray, device: torch.device) -> np.ndarray:
    """16 kHz audio -> FCPE F0; unvoiced frames come back as 0 Hz, confidence 0/1."""
    frames = frame_count(audio, TRACK_RATE)
    wav = torch.from_numpy(audio.astype(np.float32))[None, :, None].to(device)
    with torch.no_grad():
        f0 = model.infer(
            wav,
            sr=TRACK_RATE,
            decoder_mode="local_argmax",
            threshold=FCPE_THRESHOLD,
            f0_min=F0_RANGE_HZ[0],
            f0_max=F0_RANGE_HZ[1],
            interp_uv=False,
            output_interp_target_length=frames,
        )
    hz = f0.squeeze().float().cpu().numpy()
    return np.stack([hz, (hz > 0).astype(np.float32)]).astype(np.float32)


def crepe_track(audio: np.ndarray, device: torch.device) -> np.ndarray:
    """16 kHz audio -> CREPE full F0 (Viterbi decoding) and periodicity."""
    import torchcrepe

    wav = torch.from_numpy(audio.astype(np.float32))[None]
    f0, periodicity = torchcrepe.predict(
        wav,
        TRACK_RATE,
        hop_length=int(TRACK_RATE * RMVPE_HOP_S),
        fmin=F0_RANGE_HZ[0],
        fmax=F0_RANGE_HZ[1],
        model="full",
        batch_size=CREPE_BATCH,
        device=str(device),
        return_periodicity=True,
    )
    return np.stack([f0[0].cpu().numpy(), periodicity[0].cpu().numpy()]).astype(np.float32)


def melodia_track(audio: np.ndarray, rate: int) -> np.ndarray:
    """Melodia pitch (0 = unvoiced) resampled from its 2.9 ms hop to the 10 ms grid."""
    import essentia.standard as es

    upsampled = resample_audio(audio.astype(np.float32), rate, MELODIA_RATE)
    extractor = es.PredominantPitchMelodia(
        sampleRate=MELODIA_RATE,
        hopSize=MELODIA_HOP,
        minFrequency=F0_RANGE_HZ[0],
        maxFrequency=F0_RANGE_HZ[1],
    )
    pitch, _confidence = extractor(es.EqualLoudness(sampleRate=MELODIA_RATE)(upsampled))
    source_times = np.arange(len(pitch)) * MELODIA_HOP / MELODIA_RATE
    target_times = np.arange(frame_count(audio, rate)) * RMVPE_HOP_S
    nearest = np.clip(np.searchsorted(source_times, target_times), 0, len(pitch) - 1)
    hz = np.asarray(pitch, dtype=np.float32)[nearest]
    return np.stack([hz, (hz > 0).astype(np.float32)]).astype(np.float32)


def vocal_track(method: str, models: dict, vocals: np.ndarray, device) -> np.ndarray:
    if method == "rmvpe_vocals":
        return rmvpe_track(models["rmvpe"], vocals)
    if method == "fcpe_vocals":
        return fcpe_track(models["fcpe"], vocals, device)
    if method == "crepe_vocals":
        return crepe_track(vocals, device)
    raise ValueError(f"unknown vocal method {method}")


def track_to_hz(track: np.ndarray) -> np.ndarray:
    """MIREX estimate: F0 where voiced, -F0 (a pitch guess) where judged unvoiced.

    The guess lets raw pitch/chroma accuracy measure pitch apart from voicing. Trackers
    that report 0 Hz when unvoiced (FCPE, Melodia) make no guess there.
    """
    voiced = (track[1] >= CONFIDENCE_THRESHOLD) & (track[0] > 0)
    return np.where(voiced, track[0], -np.maximum(track[0], 0.0)).astype(np.float64)


def frame_scores(ref_times: np.ndarray, ref_hz: np.ndarray, track: np.ndarray) -> dict:
    """MIREX melody measures of one track against a (times, Hz; 0 = unvoiced) reference."""
    import mir_eval

    est_times = np.arange(track.shape[1]) * RMVPE_HOP_S
    scores = mir_eval.melody.evaluate(ref_times, ref_hz, est_times, track_to_hz(track))
    return {short: float(scores[name]) for name, short in FRAME_METRICS.items()}
