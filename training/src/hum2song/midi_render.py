"""Parse monophonic MIDI and render it to a simple harmonic waveform.

MIR-QBSH and HumTrans ship melody MIDI rather than song audio. Rendering that
MIDI is the reference side for contrastive pairs and for closed-set eval.
This is not the Stage 2 Demucs/CREPE synthesizer.
"""

import bisect
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


@dataclass(frozen=True)
class TickNote:
    """One note in absolute MIDI ticks, before the tempo map is applied."""

    start_tick: int
    end_tick: int
    pitch: int
    velocity: int


@dataclass(frozen=True)
class TempoMap:
    """Piecewise-constant tempo. Converts absolute ticks to seconds for every track."""

    ticks: tuple[int, ...]
    seconds_at: tuple[float, ...]
    seconds_per_tick: tuple[float, ...]

    @classmethod
    def build(cls, changes: list[tuple[int, int]], ticks_per_quarter: int) -> "TempoMap":
        """Merge tempo events from all tracks. 120 bpm applies until the first event."""
        by_tick = {0: DEFAULT_TEMPO_US}
        for tick, tempo_us in sorted(changes, key=_first):
            by_tick[tick] = tempo_us
        ticks: list[int] = []
        seconds_at: list[float] = []
        rates: list[float] = []
        elapsed = 0.0
        for tick, tempo_us in by_tick.items():
            if ticks:
                elapsed += (tick - ticks[-1]) * rates[-1]
            ticks.append(tick)
            seconds_at.append(elapsed)
            rates.append(tempo_us / 1_000_000.0 / float(ticks_per_quarter))
        return cls(tuple(ticks), tuple(seconds_at), tuple(rates))

    def seconds(self, tick: int) -> float:
        """Absolute time in seconds of an absolute tick."""
        index = bisect.bisect_right(self.ticks, tick) - 1
        return self.seconds_at[index] + (tick - self.ticks[index]) * self.seconds_per_tick[index]


def parse_midi(data: bytes) -> list[MidiNote]:
    """Read notes from a Standard MIDI file, timed by one tempo map shared by all tracks.

    Type-1 files keep their tempo events in track 0 and their notes in later tracks,
    so tempo must be collected across every track before ticks become seconds.
    """
    if data[:4] != b"MThd":
        raise ValueError("not a MIDI file")
    header_len = struct.unpack(">I", data[4:8])[0]
    _fmt, track_count, division = struct.unpack(">HHH", data[8 : 8 + header_len])
    if division & 0x8000:
        raise ValueError("SMPTE MIDI division is not supported")
    tick_notes: list[TickNote] = []
    tempo_changes: list[tuple[int, int]] = []
    for track in _split_tracks(data, 8 + header_len, track_count):
        notes, tempos = _parse_track(track)
        tick_notes.extend(notes)
        tempo_changes.extend(tempos)
    tempo_map = TempoMap.build(tempo_changes, division)
    return [_note_in_seconds(note, tempo_map) for note in tick_notes]


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


def _split_tracks(data: bytes, cursor: int, track_count: int) -> list[bytes]:
    tracks: list[bytes] = []
    for _track_index in range(track_count):
        if data[cursor : cursor + 4] != b"MTrk":
            raise ValueError("missing MIDI track")
        track_len = struct.unpack(">I", data[cursor + 4 : cursor + 8])[0]
        tracks.append(data[cursor + 8 : cursor + 8 + track_len])
        cursor += 8 + track_len
    return tracks


def _parse_track(track: bytes) -> tuple[list[TickNote], list[tuple[int, int]]]:
    """Notes and tempo changes of one track, both timed in absolute ticks."""
    cursor = 0
    tick = 0
    running: int | None = None
    open_notes: dict[int, tuple[int, int]] = {}
    notes: list[TickNote] = []
    tempos: list[tuple[int, int]] = []
    while cursor < len(track):
        delta, cursor = _read_varlen(track, cursor)
        tick += delta
        status_byte = track[cursor]
        if status_byte & 0x80:
            cursor += 1
            running = status_byte
        status = status_byte if status_byte & 0x80 else (running if running is not None else 0)
        if status == 0xFF:
            cursor, tempo_us = _read_meta(track, cursor)
            if tempo_us is not None:
                tempos.append((tick, tempo_us))
            continue
        if status in (0xF0, 0xF7):
            length, cursor = _read_varlen(track, cursor)
            cursor += length
            continue
        cursor = _consume_channel_event(track, cursor, status, tick, open_notes, notes)
    for pitch, (start_tick, velocity) in open_notes.items():
        notes.append(TickNote(start_tick, tick, pitch, velocity))
    return notes, tempos


def _note_in_seconds(note: TickNote, tempo_map: TempoMap) -> MidiNote:
    return MidiNote(
        tempo_map.seconds(note.start_tick),
        tempo_map.seconds(note.end_tick),
        note.pitch,
        note.velocity,
    )


def _consume_channel_event(
    track: bytes,
    cursor: int,
    status: int,
    tick: int,
    open_notes: dict[int, tuple[int, int]],
    notes: list[TickNote],
) -> int:
    event = status & 0xF0
    if event in (0x80, 0x90):
        pitch = track[cursor]
        velocity = track[cursor + 1]
        _apply_note(event, pitch, velocity, tick, open_notes, notes)
        return cursor + 2
    if event in (0xA0, 0xB0, 0xE0):
        return cursor + 2
    if event in (0xC0, 0xD0):
        return cursor + 1
    raise ValueError(f"unsupported MIDI status {status:#x}")


def _apply_note(
    event: int,
    pitch: int,
    velocity: int,
    tick: int,
    open_notes: dict[int, tuple[int, int]],
    notes: list[TickNote],
) -> None:
    note_off = event == 0x80 or velocity == 0
    if not note_off:
        open_notes[pitch] = (tick, velocity)
        return
    started = open_notes.pop(pitch, None)
    if started is None:
        return
    start_tick, start_velocity = started
    notes.append(TickNote(start_tick, max(tick, start_tick), pitch, start_velocity))


def _read_meta(track: bytes, cursor: int) -> tuple[int, int | None]:
    """Skip one meta event. Returns the tempo in microseconds when it is a set-tempo."""
    meta_type = track[cursor]
    cursor += 1
    length, cursor = _read_varlen(track, cursor)
    payload = track[cursor : cursor + length]
    cursor += length
    if meta_type != 0x51 or len(payload) != 3:
        return cursor, None
    return cursor, (payload[0] << 16) | (payload[1] << 8) | payload[2]


def _first(item: tuple[int, int]) -> int:
    return item[0]


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
