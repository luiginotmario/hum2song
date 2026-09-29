# D-028 (E2b) download plan

Reproduces the download list of the CHAD cover experiment. No audio is stored anywhere; only the
melody features extracted on the GPU host are kept.

1. Copy CHAD's `metadata/dataset.csv` (github.com/amanteur/CHAD) here.
2. `python make_filter.py` builds `title_regex.txt` (yt-dlp `--match-filter`: skip MTG-QBH and MLEnd songs).
3. `python make_list.py` builds `e2b_plan.json` (per CHAD cover group: original + covers ranked by
   summed CHAD correlation; every CHAD hum group outside the train split is excluded, with all its videos).
4. `python make_list2.py` builds `e2b_v2.tsv`: the 2,000 groups with the most aligned-fragment evidence
   (39 fully downloaded in the first run, from `done_snap.txt` / `have_snap.txt`), original + best cover,
   with the shared fragments ±4 s as sections.
5. `get_one.sh` (segments, xargs -P 6) ran first; `run_v3.sh` (6 long-lived workers, whole files at the
   lowest bitrate, standalone deno for YouTube's JS challenge) fetched the rest.
