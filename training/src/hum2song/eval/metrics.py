"""Top-k and MRR for song-level retrieval."""

from dataclasses import asdict, dataclass

import numpy as np

MISSING_RANK = 10**9
CHAD_TOP10 = {
    "chad_top10_jang_midi": 0.921,
    "chad_top10_jang_real": 0.868,
    "chad_top10_mtg_qbh": 0.883,
    "acrcloud_mirex_top10": 0.990,
}
PRIMARY_CHAD_TOP10 = {
    "mirqbsh": 0.921,
    "mtgqbh": 0.883,
}
PROTOCOL_NOTE = (
    "CHAD top-10 0.921 is the Jang MIDI-reference number on a database of about 2600 MIDIs "
    "(arXiv:2312.01092, Table 1). This eval ranks each query against the reference clips in "
    "the manifest, plus any extra distractor songs that were requested. The two top-10 "
    "numbers sit side by side in the report; they match the CHAD protocol only when the "
    "reference set does."
)


@dataclass
class RetrievalScores:
    """One block of retrieval metrics. None means the split had no queries."""

    count: int
    top1: float | None
    top3: float | None
    top10: float | None
    mrr: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def retrieval_scores(ranks: list[int]) -> RetrievalScores:
    """ranks are 0-based positions of the correct song. Lower is better."""
    if not ranks:
        return RetrievalScores(0, None, None, None, None)
    count = len(ranks)
    return RetrievalScores(
        count=count,
        top1=_hit_rate(ranks, 1),
        top3=_hit_rate(ranks, 3),
        top10=_hit_rate(ranks, 10),
        mrr=sum(1.0 / (rank + 1) for rank in ranks) / count,
    )


def ranks_for_queries(
    query_emb: np.ndarray,
    query_song_ids: list[str],
    ref_emb: np.ndarray,
    ref_song_ids: list[str],
) -> list[int]:
    """Rank the true song for each query. Score of a song is its best reference clip."""
    if len(query_song_ids) == 0:
        return []
    if len(ref_song_ids) == 0:
        return [MISSING_RANK for _song_id in query_song_ids]
    similarity = query_emb @ ref_emb.T
    songs, columns = _group_columns(ref_song_ids)
    song_index = {song_id: index for index, song_id in enumerate(songs)}
    ranks: list[int] = []
    for row, target in enumerate(query_song_ids):
        scores = np.array(
            [float(similarity[row, columns[song_id]].max()) for song_id in songs],
            dtype=np.float64,
        )
        order = np.argsort(-scores, kind="stable")
        target_index = song_index.get(target)
        if target_index is None:
            ranks.append(MISSING_RANK)
            continue
        ranks.append(int(np.where(order == target_index)[0][0]))
    return ranks


def _hit_rate(ranks: list[int], k: int) -> float:
    hits = sum(rank < k for rank in ranks)
    return hits / float(len(ranks))


def _group_columns(song_ids: list[str]) -> tuple[list[str], dict[str, np.ndarray]]:
    grouped: dict[str, list[int]] = {}
    order: list[str] = []
    for index, song_id in enumerate(song_ids):
        if song_id not in grouped:
            grouped[song_id] = []
            order.append(song_id)
        grouped[song_id].append(index)
    columns = {song_id: np.asarray(indexes, dtype=np.int64) for song_id, indexes in grouped.items()}
    return order, columns
