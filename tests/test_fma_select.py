import pandas as pd

from hum2song.catalog.fma import select_tracks


def tiny_catalog() -> pd.DataFrame:
    rows = []
    for track_id, genre in enumerate(["Pop", "Rock", "Electronic", "Pop"] * 5, start=1):
        rows.append(
            {
                "title": f"t{track_id}",
                "artist": "a",
                "genre": genre,
                "license": "CC",
                "duration_s": 120.0,
            }
        )
    return pd.DataFrame(rows, index=list(range(1, len(rows) + 1)))


def test_select_tracks_excludes_ids():
    tracks = tiny_catalog()
    first = select_tracks(tracks, count=4, seed=1, genres=("Pop", "Rock"))
    second = select_tracks(
        tracks,
        count=4,
        seed=1,
        genres=("Pop", "Rock"),
        exclude_ids=set(first.index.astype(int)),
    )
    assert set(first.index).isdisjoint(set(second.index))
    assert len(second) > 0
