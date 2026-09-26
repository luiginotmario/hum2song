"""Stage 2 synthetic-query interface. The functions are stubs."""

from hum2song.synth.stage2 import (
    SynthJob,
    augment_lyric_agnostic,
    extract_f0,
    separate_vocals,
    synthesize_query,
)

__all__ = [
    "SynthJob",
    "augment_lyric_agnostic",
    "extract_f0",
    "separate_vocals",
    "synthesize_query",
]
