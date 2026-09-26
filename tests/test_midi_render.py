"""A one-note MIDI file renders to a non-silent 24 kHz waveform."""

from pathlib import Path

from hum2song.audio import TARGET_SAMPLE_RATE
from hum2song.midi_render import parse_midi, render_midi_file, render_notes
from tests.support import write_quarter_note


def test_quarter_note_lasts_half_a_second(tmp_path: Path) -> None:
    source = tmp_path / "note.mid"
    write_quarter_note(source)
    notes = parse_midi(source.read_bytes())
    assert len(notes) == 1
    assert notes[0].pitch == 60
    assert abs(notes[0].end_s - notes[0].start_s - 0.5) < 1.0e-6
    samples = render_notes(notes, TARGET_SAMPLE_RATE)
    assert float(samples.std()) > 0.01
    dest = tmp_path / "note.wav"
    duration = render_midi_file(source, dest, TARGET_SAMPLE_RATE)
    assert dest.exists()
    assert duration > 0.4
    again = render_midi_file(source, dest, TARGET_SAMPLE_RATE)
    assert again == duration
