"""Song melody -> 10 s contour chunks -> embeddings (docs/DECISIONS.md D-014, D-017).

Melody: htdemucs vocal stem -> RMVPE (D-014), kept as the (2, frames) 10 ms track so it
can be cached and re-chunked. Songs are decoded by ffmpeg straight to 44.1 kHz mono, so
htdemucs gets the full band, and the stem is resampled to 16 kHz with an anti-alias filter.
Chunks: CHUNK_S windows every HOP_S seconds on the 20 ms
contour grid; windows with less than MIN_VOICED voiced frames (instrumental passages,
where the vocal stem is empty) are skipped because a contour without melody cannot match.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio

from hum2song.audio import resample_audio
from hum2song.contour.evaluate import embed_contours
from hum2song.contour.features import FRAME_S, rmvpe_contour, trim_unvoiced, voiced_fraction
from hum2song.contour.melody import SEPARATOR_RATE, TRACK_RATE
from hum2song.contour.trackers import rmvpe_track

CHUNK_S = 10.0
HOP_S = 5.0
MIN_VOICED = 0.25
SEGMENT_OVERLAP = 0.25
SEGMENT_BATCH = 16


@dataclass
class Chunk:
    start_s: float
    voiced: float
    contour: np.ndarray


def decode_audio(path: Path | str, rate: int = SEPARATOR_RATE) -> np.ndarray:
    """Any ffmpeg-readable file -> mono float32 at `rate`."""
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(rate)]
    result = subprocess.run([*command, "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def to_track_rate(audio: np.ndarray, rate: int) -> np.ndarray:
    resampled = torchaudio.functional.resample(torch.from_numpy(audio), rate, TRACK_RATE)
    return resampled.numpy()


def segment_weight(length: int, device) -> torch.Tensor:
    """demucs apply_model's triangular overlap-add weight (transition_power 1)."""
    rising = torch.arange(1, length // 2 + 1, device=device)
    falling = torch.arange(length - length // 2, 0, -1, device=device)
    weight = torch.cat([rising, falling]).float()
    return weight / weight.max()


def segment_windows(length: int, segment: int, overlap: float) -> list[tuple[int, int, int]]:
    """(offset, kept length, left pad) per segment, as demucs TensorChunk.padded centers it."""
    stride = int((1 - overlap) * segment)
    windows = []
    for offset in range(0, length, stride):
        kept = min(segment, length - offset)
        windows.append((offset, kept, (segment - kept) // 2))
    return windows


def separate_vocals_batched(separator, audio: np.ndarray, device, batch: int = SEGMENT_BATCH):
    """apply_model(split=True, shifts=0) for one-model htdemucs, with segments batched.

    Mono 44.1 kHz in, mono 44.1 kHz vocal stem out. Same normalization, segment length,
    overlap and weights as separate_vocals; only the GPU batching differs.
    """
    (model,) = separator.models
    segment = int(model.samplerate * model.segment)
    wav = torch.from_numpy(np.stack([audio, audio]).astype(np.float32)).to(device)
    mean, std = wav.mean(), wav.std().clamp_min(1.0e-8)
    padded = torch.nn.functional.pad((wav - mean) / std, (segment, segment))
    windows = segment_windows(wav.shape[-1], segment, SEGMENT_OVERLAP)
    weight = segment_weight(segment, device)
    vocals_index = model.sources.index("vocals")
    out = torch.zeros_like(wav)
    total = torch.zeros(wav.shape[-1], device=device)
    for first in range(0, len(windows), batch):
        group = windows[first : first + batch]
        starts = [segment + offset - left for offset, _kept, left in group]
        inputs = torch.stack([padded[:, start : start + segment] for start in starts])
        with torch.no_grad():
            stems = model(inputs)[:, vocals_index]
        for stem, (offset, kept, left) in zip(stems, group, strict=True):
            out[:, offset : offset + kept] += weight[:kept] * stem[:, left : left + kept]
            total[offset : offset + kept] += weight[:kept]
    vocals = out / total * std + mean
    return vocals.mean(dim=0).float().cpu().numpy()


def melody_track(separator, rmvpe, audio: np.ndarray, device, rate: int = SEPARATOR_RATE):
    """Mono song audio at `rate` -> (2, frames) RMVPE track of its htdemucs vocal stem."""
    full_band = audio if rate == SEPARATOR_RATE else resample_audio(audio, rate, SEPARATOR_RATE)
    vocals = separate_vocals_batched(separator, full_band, device)
    return rmvpe_track(rmvpe, to_track_rate(vocals, SEPARATOR_RATE))


def chunk_contour(
    contour: np.ndarray,
    chunk_s: float = CHUNK_S,
    hop_s: float = HOP_S,
    min_voiced: float = MIN_VOICED,
) -> list[Chunk]:
    """Fixed windows over the whole song; the last window may be shorter."""
    length, hop = int(round(chunk_s / FRAME_S)), int(round(hop_s / FRAME_S))
    starts = range(0, max(len(contour) - length, 0) + 1, hop)
    windows = [(start, contour[start : start + length]) for start in starts]
    return [
        Chunk(start * FRAME_S, voiced_fraction(window), trim_unvoiced(window))
        for start, window in windows
        if voiced_fraction(window) >= min_voiced
    ]


def song_chunks(track: np.ndarray) -> list[Chunk]:
    return chunk_contour(rmvpe_contour(track))


def embed_chunks(model, chunks: list[Chunk], device) -> np.ndarray:
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return embed_contours(model, [chunk.contour for chunk in chunks], device)
