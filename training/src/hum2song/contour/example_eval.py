"""Query-by-example eval for query sets without melody references (MLEnd, D-013).

MLEnd has hums and whistles of 8 songs by about 200 people, but no reference melody.
Each song's reference is built from other people's clips: the query's own performer is
always left out. Two song scores:
  centroid  cosine to the mean embedding of the song's reference clips (primary)
  max       best single reference clip
This is an 8-way closed set, so top-1 and MRR carry the information; chance top-1 is 1/8.
"""

from dataclasses import dataclass

import numpy as np

from hum2song.eval.metrics import retrieval_scores

REPORTED = ("top1", "top3", "mrr")


@dataclass
class ExampleSet:
    """Embeddings with song and performer labels."""

    embeddings: np.ndarray
    songs: list[str]
    people: list[str]


def song_scores(query: ExampleSet, refs: ExampleSet, mode: str) -> tuple[np.ndarray, list[str]]:
    """(queries, songs) scores with every query's own performer removed from the references."""
    songs = sorted(set(refs.songs))
    ref_song = np.array([songs.index(song) for song in refs.songs])
    same_person = np.array(query.people)[:, None] == np.array(refs.people)[None, :]
    similarity = query.embeddings @ refs.embeddings.T
    scores = np.full((len(query.songs), len(songs)), -np.inf)
    for index in range(len(songs)):
        usable = (ref_song == index)[None, :] & ~same_person
        scores[:, index] = score_one_song(similarity, refs.embeddings, query, usable, mode)
    return scores, songs


def score_one_song(similarity, ref_embeddings, query: ExampleSet, usable, mode: str) -> np.ndarray:
    if mode == "max":
        return np.where(usable, similarity, -np.inf).max(axis=1)
    if mode == "centroid":
        sums = usable.astype(np.float64) @ ref_embeddings
        norms = np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1.0e-9)
        return np.sum(query.embeddings * (sums / norms), axis=1)
    raise ValueError(f"unknown mode {mode}")


def example_metrics(query: ExampleSet, refs: ExampleSet, mode: str) -> dict[str, float]:
    scores, songs = song_scores(query, refs, mode)
    target = np.array([songs.index(song) for song in query.songs])
    target_score = scores[np.arange(len(target)), target]
    ranks = (scores > target_score[:, None]).sum(axis=1).tolist()
    values = retrieval_scores(ranks).as_dict()
    return {metric: float(values[metric]) for metric in REPORTED}
