-- Song melody contours for the re-ranking stage of the window pipeline (D-025, D-027).
-- The 20 ms contour (semitones, NaN unvoiced) of the track the index used, as float16 bytes.
CREATE TABLE IF NOT EXISTS song_contours (
    song_id  text PRIMARY KEY REFERENCES songs (song_id) ON DELETE CASCADE,
    contour  bytea NOT NULL
);
