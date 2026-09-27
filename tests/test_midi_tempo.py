"""Tempo lives in track 0 of type-1 MIDI and must time the notes in every track."""

from hum2song.midi_render import TempoMap, parse_midi
from tests.support import midi_file, note_event, tempo_event

QUARTER = bytes([0x83, 0x60])
ZERO = bytes([0x00])


def test_type1_tempo_in_track0_times_notes_in_track1() -> None:
    conductor = tempo_event(ZERO, bpm=60.0)
    melody = note_event(ZERO, 60, QUARTER)
    notes = parse_midi(midi_file([conductor, melody]))
    assert len(notes) == 1
    assert abs(notes[0].end_s - notes[0].start_s - 1.0) < 1.0e-9


def test_tempo_change_mid_song_applies_to_later_notes() -> None:
    conductor = tempo_event(ZERO, bpm=120.0) + tempo_event(QUARTER, bpm=60.0)
    melody = note_event(ZERO, 60, QUARTER) + note_event(ZERO, 62, QUARTER)
    first, second = sorted(parse_midi(midi_file([conductor, melody])), key=lambda n: n.start_s)
    assert abs(first.end_s - 0.5) < 1.0e-9
    assert abs(second.start_s - 0.5) < 1.0e-9
    assert abs(second.end_s - 1.5) < 1.0e-9


def test_tempo_map_defaults_to_120_bpm_until_first_event() -> None:
    tempo_map = TempoMap.build([(960, 1_000_000)], ticks_per_quarter=480)
    assert abs(tempo_map.seconds(480) - 0.5) < 1.0e-9
    assert abs(tempo_map.seconds(960) - 1.0) < 1.0e-9
    assert abs(tempo_map.seconds(1440) - 2.0) < 1.0e-9


def test_event_at_tick_zero_replaces_the_default_tempo() -> None:
    tempo_map = TempoMap.build([(0, 1_000_000)], ticks_per_quarter=480)
    assert abs(tempo_map.seconds(480) - 1.0) < 1.0e-9
