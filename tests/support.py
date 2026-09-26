"""Small audio and MIDI fixtures shared by the tests."""

import struct
from pathlib import Path

import numpy as np

from hum2song.audio import write_wav


def write_quarter_note(path: Path) -> None:
    """Format-0 MIDI: middle C for one quarter note at 120 bpm (0.5 s)."""
    track = bytes(
        [
            0x00,
            0xFF,
            0x51,
            0x03,
            0x07,
            0xA1,
            0x20,
            0x00,
            0x90,
            0x3C,
            0x64,
            0x83,
            0x60,
            0x80,
            0x3C,
            0x00,
            0x00,
            0xFF,
            0x2F,
            0x00,
        ]
    )
    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, 480)
    path.write_bytes(header + b"MTrk" + struct.pack(">I", len(track)) + track)


def write_tone(
    path: Path,
    frequency: float,
    seconds: float = 0.3,
    sample_rate: int = 24000,
) -> None:
    """Write a short sine wav."""
    time = np.linspace(0.0, seconds, int(sample_rate * seconds), endpoint=False)
    wave = (0.2 * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)
    write_wav(path, wave, sample_rate)
