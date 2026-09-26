"""Demucs + CREPE synthetic hums and whistles.

Phase 1 does not generate these queries. Callers should catch NotImplementedError
and keep training on real pairs only. The signatures are the Stage 2 contract.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SynthJob:
    """One synthetic query to render from a song stem."""

    song_id: str
    song_path: str
    start_s: float
    kind: str


def separate_vocals(song_path: str, out_dir: str, model_name: str = "htdemucs") -> str:
    """Separate a vocals stem with Demucs. Returns the stem path. Not implemented."""
    raise NotImplementedError(
        "Stage 2 Demucs separation is not implemented. "
        f"song_path={song_path} out_dir={out_dir} model_name={model_name}"
    )


def extract_f0(
    vocals_path: str,
    out_npz: str,
    model_capacity: str = "full",
    viterbi: bool = True,
) -> str:
    """Write CREPE f0, confidence, and rms at 100 Hz. Returns out_npz. Not implemented."""
    raise NotImplementedError(
        "Stage 2 CREPE f0 extraction is not implemented. "
        f"vocals_path={vocals_path} out_npz={out_npz} "
        f"model_capacity={model_capacity} viterbi={viterbi}"
    )


def synthesize_query(job: SynthJob, f0_path: str, out_wav: str) -> str:
    """Resynthesize a hum, whistle, or sung query. Returns out_wav. Not implemented."""
    raise NotImplementedError(
        "Stage 2 query resynthesis is not implemented. "
        f"kind={job.kind} song_id={job.song_id} f0_path={f0_path} out_wav={out_wav}"
    )
