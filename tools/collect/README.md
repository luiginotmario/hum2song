# Real query collection (D-043)

Prolific page for real whistles and any/wrong-lyrics singing. Consent → 3 s mic check → 10 prompts
(alternating whistle / sing) from 48 well-known songs in the live index, each with an iTunes 30 s preview
(linked, played for 15 s). Skip swaps in another song. Clips upload one by one.

Storage: `/lambda/nfs/hum2song-data/owned/prolific/<STUDY_ID>/clips/*.webm|m4a` + per-clip `.json` + `manifest.csv`.

Run on Lambda (port 8300, never touches the :8000 matcher):

    cd /lambda/nfs/hum2song-data/dev/collect && COMPLETION_CODE=XXXX ./run.sh   # prints the new https URL

The Cloudflare quick tunnel URL changes on every restart; update the Prolific study URL if it does.
`songs.json` is rebuilt from `songs_seed.tsv` with `python resolve_previews.py`.
