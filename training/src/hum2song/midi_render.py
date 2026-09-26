"""Parse monophonic MIDI and render it to a simple harmonic waveform.

MIR-QBSH and HumTrans ship melody MIDI rather than song audio. Rendering that
MIDI is the reference side for contrastive pairs and for closed-set eval.
This is not the Stage 2 Demucs/CREPE synthesizer.
"""

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hum2song.audio import audio_duration_s, peak_normalize, write_wav

DEFAULT_TEMPO_US = 500_000
NOTE_RELEASE_S = 0.02
SILENT_DURATION_S = 0.1


@dataclass(frozen=True)
class MidiNote:
    """One note in seconds."""

    start_s: float
    end_s: float
    pitch: int
    velocity: int


def parse_midi(data: bytes) -> list[MidiNote]:
    """Read note-on/note-off events from a Standard MIDI file."""
    if data[:4] != b"MThd":
        raise ValueError("not a MIDI file")
    header_len = struct.unpack(">I", data[4:8])[0]
    _fmt, track_count, division = struct.unpack(">HHH", data[8 : 8 + header_len])
    if division & 0x8000:
        raise ValueError("SMPTE MIDI division is not supported")
    cursor = 8 + header_len
    notes: list[MidiNote] = []
    for _track_index in range(track_count):
        if data[cursor : cursor + 4] != b"MTrk":
            raise ValueError("missing MIDI track")
        cursor += 4
        track_len = struct.unpack(">I", data[cursor : cursor + 4])[0]
        cursor += 4
        track_end = cursor + track_len
        notes.extend(_parse_track(data[cursor:track_end], division))
        cursor = track_end
    return notes


def render_notes(notes: list[MidiNote], sample_rate: int) -> np.ndarray:
    """Render notes as a harmonic series with a short release. Peak-normalized."""
    if not notes:
        return np.zeros(max(int(sample_rate * SILENT_DURATION_S), 1), dtype=np.float32)
    end_s = max(note.end_s for note in notes) + NOTE_RELEASE_S
    length = max(int(round(end_s * sample_rate)), 1)
    buffer = np.zeros(length, dtype=np.float64)
    time = np.arange(length, dtype=np.float64) / float(sample_rate)
    for note in notes:
        _add_note(buffer, time, note, sample_rate)
    return peak_normalize(buffer.astype(np.float32))


def render_midi_file(source: Path, dest: Path, sample_rate: int) -> float:
    """Render a MIDI file to wav. Returns duration in seconds. Skips existing wavs."""
    if dest.exists() and dest.stat().st_size > 44:
        return audio_duration_s(dest)
    samples = render_notes(parse_midi(source.read_bytes()), sample_rate)
    write_wav(dest, samples, sample_rate)
    return len(samples) / float(sample_rate)


def _parse_track(track: bytes, ticks_per_quarter: int) -> list[MidiNote]:
    cursor = 0
    tempo_us = DEFAULT_TEMPO_US
    seconds_per_tick = tempo_us / 1_000_000.0 / float(ticks_per_quarter)
    running: int | None = None
    open_notes: dict[int, tuple[float, int]] = {}
    notes: list[MidiNote] = []
    elapsed_s = 0.0
    while cursor < len(track):
        delta, cursor = _read_varlen(track, cursor)
        elapsed_s += delta * seconds_per_tick
        status_byte = track[cursor]
        if status_byte & 0x80:
            cursor += 1
            running = status_byte
        status = status_byte if status_byte & 0x80 else (running if running is not None else 0)
        if status == 0xFF:
            cursor, tempo_us = _read_meta(track, cursor, tempo_us)
            seconds_per_tick = tempo_us / 1_000_000.0 / float(ticks_per_quarter)
            continue
        if status in (0xF0, 0xF7):
            length, cursor = _read_varlen(track, cursor)
            cursor += length
            continue
        event = status & 0xF0
        cursor = _consume_channel_event(track, cursor, event, status, elapsed_s, open_notes, notes)
    for pitch, (start_s, velocity) in open_notes.items():
        notes.append(MidiNote(start_s, elapsed_s, pitch, velocity))
    return notes


def _consume_channel_event(
    track: bytes,
    cursor: int,
    event: int,
    status: int,
    elapsed_s: float,
    open_notes: dict[int, tuple[float, int]],
    notes: list[MidiNote],
) -> int:
    if event in (0x80, 0x90):
        pitch = track[cursor]
        velocity = track[cursor + 1]
        cursor += 2
        _apply_note(event, pitch, velocity, elapsed_s, open_notes, notes)
        return cursor
    if event in (0xA0, 0xB0, 0xE0):
        return cursor + 2
    if event in (0xC0, 0xD0):
        return cursor + 1
    raise ValueError(f"unsupported MIDI status {status:#x}")


def _apply_note(
    event: int,
    pitch: int,
    velocity: int,
    elapsed_s: float,
    open_notes: dict[int, tuple[float, int]],
    notes: list[MidiNote],
) -> None:
    note_off = event == 0x80 or velocity == 0
    if note_off:
        started = open_notes.pop(pitch, None)
        if started is None:
            return
        start_s, start_velocity = started
        notes.append(MidiNote(start_s, max(elapsed_s, start_s), pitch, start_velocity))
        return
    open_notes[pitch] = (elapsed_s, velocity)


def _read_meta(track: bytes, cursor: int, tempo_us: int) -> tuple[int, int]:
    meta_type = track[cursor]
    cursor += 1
    length, cursor = _read_varlen(track, cursor)
    payload = track[cursor : cursor + length]
    cursor += length
    if meta_type == 0x51 and len(payload) == 3:
        tempo_us = (payload[0] << 16) | (payload[1] << 8) | payload[2]
    return cursor, tempo_us


def _read_varlen(data: bytes, cursor: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[cursor]
        cursor += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, cursor


def _add_note(
    buffer: np.ndarray,
    time: np.ndarray,
    note: MidiNote,
    sample_rate: int,
) -> None:
    start = max(int(note.start_s * sample_rate), 0)
    end = min(int((note.end_s + NOTE_RELEASE_S) * sample_rate), len(buffer))
    if end <= start:
        return
    frequency = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
    local = time[start:end] - note.start_s
    duration = max(note.end_s - note.start_s, 1.0 / sample_rate)
    envelope = np.ones(end - start, dtype=np.float64)
    release_n = min(int(NOTE_RELEASE_S * sample_rate), end - start)
    if release_n > 0:
        envelope[-release_n:] = np.linspace(1.0, 0.0, release_n)
    phase = 2.0 * np.pi * frequency * local
    harmonic = np.sin(phase) + 0.3 * np.sin(2.0 * phase) + 0.1 * np.sin(3.0 * phase)
    amplitude = (note.velocity / 127.0) * envelope
    window = np.clip(local / duration, 0.0, 1.0)
    buffer[start:end] += harmonic * amplitude * (0.2 + 0.8 * (1.0 - window))
