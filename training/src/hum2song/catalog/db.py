"""pgvector storage for the song library: schema, writes, and chunk search (D-017).

Search ranks chunks by cosine distance (HNSW index), then groups them by song; a song's
score is its best chunk (SPEC §1), and the matching chunk's start time is returned too.
The window table (D-025) holds 5 s windows; `window_votes` is the window-level first stage:
each query window retrieves its nearest windows, and a song scores the mean over query
windows of its best window (a query window that does not reach the song counts that
window's weakest retrieved similarity).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SQL_DIR = Path(__file__).resolve().parents[4] / "sql"
SCHEMA_FILES = ("001_library.sql", "002_windows.sql", "003_contours.sql")
CHUNK_CANDIDATES = 400
WINDOW_CANDIDATES = 400
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


@dataclass
class SongVectors:
    """What the re-ranking stage needs of one song (D-027)."""

    chunks: np.ndarray
    window_starts: np.ndarray = field(default_factory=lambda: np.zeros(0))
    windows: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    contour: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


def connect(url: str):
    import psycopg
    from pgvector.psycopg import register_vector

    connection = psycopg.connect(url, autocommit=True)
    connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(connection)
    return connection


def ensure_schema(connection) -> None:
    for name in SCHEMA_FILES:
        connection.execute((SQL_DIR / name).read_text(encoding="utf-8"))


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


def windowed_songs(connection) -> set[str]:
    return {row[0] for row in connection.execute("SELECT DISTINCT song_id FROM windows")}


def insert_windows(connection, song_id: str, starts, embeddings) -> None:
    """A song's windows in one transaction (re-inserting replaces them)."""
    with connection.transaction():
        connection.execute("DELETE FROM windows WHERE song_id = %s", (song_id,))
        with connection.cursor().copy(
            "COPY windows (song_id, start_s, embedding) FROM STDIN"
        ) as copy:
            for start, vector in zip(starts, embeddings, strict=True):
                copy.write_row((song_id, float(start), vector_text(vector)))


def nearest_windows(connection, query: np.ndarray, limit: int = WINDOW_CANDIDATES) -> list:
    """(song_id, cosine similarity) of the `limit` closest windows."""
    connection.execute(f"SET hnsw.ef_search = {max(EF_SEARCH, limit)}")
    rows = connection.execute(
        "SELECT song_id, 1 - (embedding <=> %s) AS similarity "
        "FROM windows ORDER BY embedding <=> %s LIMIT %s",
        (query.astype(np.float32), query.astype(np.float32), limit),
    ).fetchall()
    return [(song, float(similarity)) for song, similarity in rows]


def vote(per_window: list[list[tuple]]) -> dict[str, float]:
    """Mean over query windows of each song's best window; a missing song gets that
    query window's weakest retrieved similarity."""
    best = [best_similarity(rows) for rows in per_window]
    floors = [min(b.values(), default=0.0) for b in best]
    songs = set().union(*best) if best else set()
    return {
        song: float(np.mean([b.get(song, f) for b, f in zip(best, floors, strict=True)]))
        for song in songs
    }


def best_similarity(rows: list[tuple]) -> dict[str, float]:
    best: dict[str, float] = {}
    for song, similarity in rows:
        best[song] = max(similarity, best.get(song, -2.0))
    return best


def window_votes(connection, query_windows: np.ndarray, limit: int = WINDOW_CANDIDATES) -> list:
    """(song_id, vote) of every song reached by a query window, best first."""
    votes = vote([nearest_windows(connection, window, limit) for window in query_windows])
    return sorted(votes.items(), key=lambda item: -item[1])


def has_windows(connection) -> bool:
    return bool(connection.execute("SELECT EXISTS (SELECT 1 FROM windows)").fetchone()[0])


def contoured_songs(connection) -> set[str]:
    return {row[0] for row in connection.execute("SELECT song_id FROM song_contours")}


def contour_bytes(contour: np.ndarray) -> bytes:
    return np.asarray(contour, dtype=np.float16).tobytes()


def contour_from_bytes(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float16).astype(np.float32)


def insert_contour(connection, song_id: str, contour: np.ndarray) -> None:
    connection.execute(
        "INSERT INTO song_contours (song_id, contour) VALUES (%s, %s) "
        "ON CONFLICT (song_id) DO UPDATE SET contour = EXCLUDED.contour",
        (song_id, contour_bytes(contour)),
    )


def stacked(rows: list[tuple]) -> dict[str, list]:
    """song_id -> list of the remaining columns, in row order."""
    grouped: dict[str, list] = {}
    for song, *rest in rows:
        grouped.setdefault(song, []).append(rest)
    return grouped


def as_matrix(vectors: list) -> np.ndarray:
    return np.stack([v.to_numpy().astype(np.float32) for v in vectors])


def song_vectors(connection, song_ids: list[str]) -> dict[str, SongVectors]:
    """Chunk embeddings, windows (start, embedding) and contour of the given songs."""
    chunks = stacked(
        connection.execute(
            "SELECT song_id, embedding FROM chunks WHERE song_id = ANY(%s)", (song_ids,)
        ).fetchall()
    )
    windows = stacked(
        connection.execute(
            "SELECT song_id, start_s, embedding FROM windows WHERE song_id = ANY(%s) "
            "ORDER BY song_id, start_s",
            (song_ids,),
        ).fetchall()
    )
    contours = dict(
        connection.execute(
            "SELECT song_id, contour FROM song_contours WHERE song_id = ANY(%s)", (song_ids,)
        ).fetchall()
    )
    return {
        song: SongVectors(
            chunks=as_matrix([row[0] for row in rows]),
            window_starts=np.array([row[0] for row in windows.get(song, [])], dtype=np.float32),
            windows=as_matrix([row[1] for row in windows[song]])
            if song in windows
            else np.zeros((0, 0), dtype=np.float32),
            contour=contour_from_bytes(contours[song])
            if song in contours
            else np.zeros(0, dtype=np.float32),
        )
        for song, rows in chunks.items()
    }
