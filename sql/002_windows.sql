-- Short-window index for the window-level first stage (docs/DECISIONS.md D-025).
-- 5 s windows every 1 s over each song's indexed melody track, same encoder as chunks.
CREATE TABLE IF NOT EXISTS windows (
    window_id  bigserial PRIMARY KEY,
    song_id    text NOT NULL REFERENCES songs (song_id) ON DELETE CASCADE,
    start_s    real NOT NULL,
    embedding  vector(256) NOT NULL
);

CREATE INDEX IF NOT EXISTS windows_song_idx ON windows (song_id);
CREATE INDEX IF NOT EXISTS windows_embedding_hnsw ON windows USING hnsw (embedding vector_cosine_ops);
