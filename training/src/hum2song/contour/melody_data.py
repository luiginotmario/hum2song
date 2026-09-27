"""Songs with audio and a frame-level melody annotation (docs/DECISIONS.md D-014).

  mir1k     MIR-1K: 1,000 karaoke clips (16 kHz stereo, left = accompaniment, right =
            vocal), manual vocal pitch in semitones every 20 ms, first frame at 20 ms.
  adc2004   ISMIR 2004 melody contest set, 20 excerpts, <name>REF.txt = time, Hz.
  mirex05   MIREX 2005 training set, 13 excerpts, same REF.txt layout.
Mixtures are the channel mean. Reference Hz is 0 where the melody is unvoiced.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from hum2song.audio import peak_normalize, resample_audio
from hum2song.contour.features import FRAME_S, hz_to_semitones

MIR1K_DIR = "raw/mir1k/MIR-1K"
MIR1K_HOP_S = 0.02
REF_DIRS = {
    "adc2004": "raw/melody/adc2004_full_set",
    "mirex05": "raw/melody/mirex05TrainFiles",
}
DATASETS = ("mir1k", *REF_DIRS)
VOCAL_PREFIXES = {"adc2004": ("daisy", "opera", "pop")}


@dataclass
class LabelledClip:
    """One recording with its melody annotation. `song` groups clips of one song."""

    name: str
    wav: Path
    song: str
    ref_times: np.ndarray
    ref_hz: np.ndarray


def semitones_to_hz(semitones: np.ndarray) -> np.ndarray:
    return np.where(semitones > 0, 440.0 * 2.0 ** ((semitones - 69.0) / 12.0), 0.0)


def mir1k_clip(wav: Path, pitch_dir: Path) -> LabelledClip:
    semitones = np.loadtxt(pitch_dir / f"{wav.stem}.pv", ndmin=1)
    times = MIR1K_HOP_S * (np.arange(len(semitones)) + 1)
    song = wav.stem.rsplit("_", 1)[0]
    return LabelledClip(wav.stem, wav, song, times, semitones_to_hz(semitones))


def mir1k_clips(root: Path) -> list[LabelledClip]:
    base = root / MIR1K_DIR
    return [
        mir1k_clip(wav, base / "PitchLabel") for wav in sorted((base / "Wavfile").glob("*.wav"))
    ]


def ref_txt_clip(ref: Path) -> LabelledClip:
    """<name>REF.txt next to <name>.wav (MIREX05 also has train13MIDI.wav)."""
    name = ref.name.removesuffix("REF.txt")
    wav = next(path for path in sorted(ref.parent.glob(f"{name}*.wav")))
    table = np.loadtxt(ref, ndmin=2)
    return LabelledClip(name, wav, name, table[:, 0], table[:, 1])


def ref_txt_clips(directory: Path) -> list[LabelledClip]:
    return [ref_txt_clip(ref) for ref in sorted(directory.glob("*REF.txt"))]


def dataset_clips(root: Path, dataset: str) -> list[LabelledClip]:
    if dataset == "mir1k":
        return mir1k_clips(root)
    if dataset in REF_DIRS:
        return ref_txt_clips(root / REF_DIRS[dataset])
    raise ValueError(f"unknown dataset {dataset}; known: {DATASETS}")


def vocal_subset(dataset: str, clips: list[LabelledClip]) -> list[LabelledClip]:
    """The 12 sung ADC2004 excerpts (synthesized voice, opera, pop) used in the literature."""
    prefixes = VOCAL_PREFIXES.get(dataset)
    return [c for c in clips if c.name.startswith(prefixes)] if prefixes else []


def read_channels(wav: Path, rate: int) -> np.ndarray:
    """(channels, samples) float32 at `rate`."""
    data, file_rate = sf.read(str(wav), dtype="float32", always_2d=True)
    return np.stack([resample_audio(channel, int(file_rate), rate) for channel in data.T])


def mixture(wav: Path, rate: int) -> np.ndarray:
    return peak_normalize(read_channels(wav, rate).mean(axis=0))


def mir1k_vocal(wav: Path, rate: int) -> np.ndarray:
    """The isolated singing channel of a MIR-1K clip."""
    return peak_normalize(read_channels(wav, rate)[1])


def reference_contour(times: np.ndarray, hz: np.ndarray) -> np.ndarray:
    """Annotation -> semitone contour on the FRAME_S grid (nearest label at each frame center)."""
    count = max(int(np.ceil(times[-1] / FRAME_S)), 1)
    centers = (np.arange(count) + 0.5) * FRAME_S
    nearest = np.clip(np.searchsorted(times, centers), 0, len(times) - 1)
    previous = np.clip(nearest - 1, 0, len(times) - 1)
    closer = np.abs(times[previous] - centers) < np.abs(times[nearest] - centers)
    picked = hz[np.where(closer, previous, nearest)]
    return np.where(picked > 0, hz_to_semitones(picked), np.nan).astype(np.float32)
