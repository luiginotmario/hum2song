# D-038 · Grow open FMA library + live E2b real-hum eval
Date: 2026-10-02 (ET). Lambda job finished ~2026-10-02 20:46 UTC.

## Goal
Grow the live song library with free FMA full tracks (no Mac, no piracy), re-window with E2b, and re-measure CHAD + MTG-QBH on the same live windows pipeline (eval_live_search.py, contour_e2b_s0/best.pt).

## Added
| Library | Count | Seed | Notes |
|---|---:|---:|---|
| `fma_full_extra2_3k` | 3000 | 20261002 | Vocal genres; exclude prior FMA 7k |
| `fma_electronic_extra_2k` | 2000 | 20261002 | Electronic only |
| **Total new** | **5000** | | All downloaded via remotezip; built+windowed on Lambda |

Genre mix (vocal): Rock 1699, Hip-Hop 476, Folk 371, Pop 230, International 172, Country 21, Soul-RnB 20, Blues 11.

## Library size
| | Songs | Searchable | Chunks | Windows (songs / rows) |
|---|---:|---:|---:|---:|
| Before (health) | 8,060 | 7,304 | 189,919 | 7,304 / 924,305 |
| After (health) | **13,060** | **11,467** | **284,472** | **11,467 / 1,384,567** |
| Δ | +5,000 | +4,163 | +94,553 | +4,163 / +460,262 |

Model unchanged: `contour_e2b_s0/best.pt`. `window_index=true`.

## Live real-hum eval (`eval_live_search.py`, windows mode, max 300/set)
Same code path as POST /search after contour extract.

| Set | Before (8.1k) top-1 / top-10 (MRR) | After (13.1k) top-1 / top-10 (MRR) |
|---|---|---|
| CHAD test (300) | **0.513 / 0.670** (0.570) | **0.483 / 0.657** (0.545) |
| CHAD val (300) | 0.257 / 0.360 (0.290) | 0.250 / 0.347 (0.281) |
| MTG-QBH sung (110) | **0.645 / 0.800** (0.701) | **0.655 / 0.791** (0.704) |

Mean windows search time rose modestly (CHAD test 0.36→0.38 s; MTG 0.65→0.68 s).

## Vs prior headline bars
Prior cited bars (E2b D-025 / live era): CHAD ~**53% / 64%**, MTG ~**70% / 75%** (smaller pools / different snapshots).

| | Prior cited | This run before (8k) | This run after (13k) |
|---|---|---|---|
| CHAD test top-1/10 | ~0.53 / 0.64 | 0.513 / 0.670 | 0.483 / 0.657 |
| MTG top-1/10 | ~0.70 / 0.75 | 0.645 / 0.800 | 0.655 / 0.791 |

Takeaway: growing open FMA distractors by +5k costs ~3 CHAD top-1 points vs the 8k snapshot; MTG holds. Still in the same ballpark as the old ~53/64 CHAD bar; not a model regression (weights unchanged).

## Artifacts
- `/lambda/nfs/hum2song-data/results/d038/live_before.json`
- `/lambda/nfs/hum2song-data/results/d038/live_after.json`
- `/lambda/nfs/hum2song-data/results/d038/health_{before,after}.json`
- Job: `/lambda/nfs/hum2song-data/jobs/d038_grow/run.sh`
- Logs: `/lambda/nfs/hum2song-data/logs/d038_*.log`

## Next
- More FMA or MTG-Jamendo if catalog breadth still matters.
- Chart/commercial coverage still the live-demo gap (not this batch).
- Optional: re-tune HNSW when past ~10k searchable (now 11.5k).
