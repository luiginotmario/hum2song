"""pgvector storage for the song library: schema, writes, and chunk search (D-017).

Search ranks chunks by cosine distance (HNSW index), then groups them by song; a song's
score is its best chunk (SPEC §1), and the matching chunk's start time is returned too.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_SQL = Path(__file__).resolve().parents[4] / "sql" / "001_library.sql"
CHUNK_CANDIDATES = 400
EF_SEARCH = 400
SONG_COLUMNS = (
    "song_id",
    "source",
    "source_tier",
    "coverage",
    "title",
    "artist",
    "genre",
    "license",
    "duration_s",
    "model_ver",
    "chunk_count",
)


@dataclass
class SongHit:
    song_id: str
    title: str
    artist: str
    score: float
    best_start_s: float


def connect(url: str):
    import psycopg
    from pgvector.psycopg import register_vector

    connection = psycopg.connect(url, autocommit=True)
    connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(connection)
    return connection


def ensure_schema(connection) -> None:
    connection.execute(SCHEMA_SQL.read_text(encoding="utf-8"))


def indexed_songs(connection) -> set[str]:
    return {row[0] for row in connection.execute("SELECT song_id FROM songs").fetchall()}


def searchable_songs(connection) -> list[str]:
    """Songs with at least one voiced chunk (fully instrumental songs cannot be hummed)."""
    rows = connection.execute("SELECT song_id FROM songs WHERE chunk_count > 0 ORDER BY song_id")
    return [row[0] for row in rows.fetchall()]


def library_counts(connection) -> dict[str, int]:
    songs = connection.execute("SELECT count(*) FROM songs").fetchone()[0]
    searchable = connection.execute("SELECT count(*) FROM songs WHERE chunk_count > 0")
    chunks = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
    return {"songs": int(songs), "searchable": int(searchable.fetchone()[0]), "chunks": int(chunks)}


def insert_song(connection, song: dict, model_ver: str, starts, voiced, embeddings) -> None:
    """One song row and its chunks in a single transaction (re-inserting replaces it)."""
    row = {**song, "model_ver": model_ver, "chunk_count": len(starts)}
    columns = ", ".join(SONG_COLUMNS)
    values = ", ".join(f"%({name})s" for name in SONG_COLUMNS)
    with connection.transaction():
        connection.execute("DELETE FROM songs WHERE song_id = %s", (song["song_id"],))
        connection.execute(f"INSERT INTO songs ({columns}) VALUES ({values})", row)
        with connection.cursor().copy(
            "COPY chunks (song_id, start_s, voiced, embedding) FROM STDIN"
        ) as copy:
            for start, fraction, vector in zip(starts, voiced, embeddings, strict=True):
                copy.write_row(
                    (song["song_id"], float(start), float(fraction), vector_text(vector))
                )


def vector_text(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{value:.6f}" for value in vector) + "]"


def nearest_chunks(connection, query: np.ndarray, limit: int = CHUNK_CANDIDATES) -> list[tuple]:
    """(song_id, start_s, cosine similarity) of the `limit` closest chunks."""
    connection.execute(f"SET hnsw.ef_search = {max(EF_SEARCH, limit)}")
    rows = connection.execute(
        "SELECT song_id, start_s, 1 - (embedding <=> %s) AS similarity "
        "FROM chunks ORDER BY embedding <=> %s LIMIT %s",
        (query.astype(np.float32), query.astype(np.float32), limit),
    ).fetchall()
    return [(song, float(start), float(similarity)) for song, start, similarity in rows]


def best_per_song(rows: list[tuple], top_k: int) -> list[tuple]:
    """Keep each song's best chunk, highest first, `top_k` songs."""
    best: dict[str, tuple] = {}
    for song, start, similarity in rows:
        if song not in best or similarity > best[song][2]:
            best[song] = (song, start, similarity)
    return sorted(best.values(), key=lambda row: -row[2])[:top_k]


def song_hits(connection, query: np.ndarray, top_k: int) -> list[SongHit]:
    ranked = best_per_song(nearest_chunks(connection, query), top_k)
    titles = dict(
        (song, (title, artist))
        for song, title, artist in connection.execute(
            "SELECT song_id, title, artist FROM songs WHERE song_id = ANY(%s)",
            ([row[0] for row in ranked],),
        ).fetchall()
    )
    return [
        SongHit(song, *titles.get(song, ("", "")), round(similarity, 4), start)
        for song, start, similarity in ranked
    ]
