"""Stage 2 stays a stub."""

import pytest

from hum2song.synth import SynthJob, extract_f0, separate_vocals, synthesize_query


def test_stage2_functions_are_not_implemented() -> None:
    job = SynthJob(song_id="yt:abc", song_path="song.wav", start_s=0.0, kind="hum")
    with pytest.raises(NotImplementedError):
        separate_vocals("song.wav", "stems")
    with pytest.raises(NotImplementedError):
        extract_f0("stems/vocals.wav", "f0/song.npz")
    with pytest.raises(NotImplementedError):
        synthesize_query(job, "f0/song.npz", "queries/synth/out.wav")
