-- Song library for contour search (docs/DECISIONS.md D-017). Applied by catalog/db.py.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS songs (
    song_id      text PRIMARY KEY,           -- e.g. fma:000002
    source       text NOT NULL,              -- fma_full
    source_tier  text NOT NULL,              -- P (open data, D-009)
    coverage     text NOT NULL,              -- full | preview
    title        text,
    artist       text,
    genre        text,
    license      text,
    duration_s   real,
    model_ver    text NOT NULL,              -- checkpoint that produced the chunks
    chunk_count  integer NOT NULL DEFAULT 0,
    added_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   bigserial PRIMARY KEY,
    song_id    text NOT NULL REFERENCES songs (song_id) ON DELETE CASCADE,
    start_s    real NOT NULL,
    voiced     real NOT NULL,                -- voiced fraction of the 10 s window
    embedding  vector(256) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_song_idx ON chunks (song_id);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
