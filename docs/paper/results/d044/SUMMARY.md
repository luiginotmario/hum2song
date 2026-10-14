# D-044 summary — closed-set MLEnd whistle fine-tune

Source: `results.json` / `run.log` (CPU run EXIT 0). Numbers are exact from the file; 3-decimal display matches the paper tables.

## Protocol
- 8 MLEnd songs; performer-disjoint train/val/test.
- Counts (clips, performers): `{"train": {"hum": [2812, 136], "whistle": [1148, 82]}, "val": {"hum": [1020, 45], "whistle": [310, 21]}, "test": {"hum": [972, 45], "whistle": [339, 26]}}`.
- Training steps: 300. Seed 0 only for each train-frac.
- Score: whistle → other people's hums (centroid), query performer left out; 8-way closed set (chance top-1 0.125).
- CIs: clip bootstrap and performer bootstrap (see `run_d044.py`).
- MRR below is mean $1/(\mathrm{rank}+1)$ from the committed ranks (not stored as a separate field).

## whistle→hums (test, n=339)

| Model | train performers | top-1 | clip 95% CI | performer 95% CI | top-3 | clip 95% CI | MRR | hits top-1 | hits top-3 |
|---|---:|---:|---|---|---:|---|---:|---:|---:|
| e2b_zero_shot | 0 | 0.631268 | [0.581121, 0.681416] | [0.499968, 0.753057] | 0.769912 | [0.722714, 0.814159] | 0.730552 | 214/339 | 261/339 |
| ft_frac0.1_s0 | 14 | 0.634218 | [0.584071, 0.684366] | [0.502822, 0.759908] | 0.775811 | [0.731563, 0.820059] | 0.734917 | 215/339 | 263/339 |
| ft_frac0.25_s0 | 34 | 0.625369 | [0.575221, 0.678466] | [0.484471, 0.754391] | 0.793510 | [0.749263, 0.834808] | 0.737863 | 212/339 | 269/339 |
| ft_frac0.5_s0 | 68 | 0.654867 | [0.604720, 0.705015] | [0.524203, 0.777786] | 0.811209 | [0.769912, 0.852507] | 0.757350 | 222/339 | 275/339 |
| ft_frac1.0_s0 | 136 | 0.672566 | [0.622419, 0.722714] | [0.536301, 0.800628] | 0.820059 | [0.778761, 0.858407] | 0.768373 | 228/339 | 278/339 |

## Paper display (3 decimals)

| Model | Train performers | Top-1 [clip 95% CI] | Top-3 | MRR |
|---|---:|---|---:|---:|
| E2b zero-shot | 0 | 0.631 [0.581, 0.681] | 0.770 | 0.731 |
| FT 10% train performers | 14 | 0.634 [0.584, 0.684] | 0.776 | 0.735 |
| FT 25% | 34 | 0.625 [0.575, 0.678] | 0.794 | 0.738 |
| FT 50% | 68 | 0.655 [0.605, 0.705] | 0.811 | 0.757 |
| FT 100% | 136 | 0.673 [0.622, 0.723] | 0.820 | 0.768 |

## Key result
Closed-set on MLEnd's 8 songs (performer-disjoint): E2b zero-shot whistle→hums top-1 **0.631** (214/339; clip CI [0.581, 0.681]) → full-train fine-tune **0.673** (228/339; [0.622, 0.723]); top-3 0.770 → 0.820. This is not open-set generalization to new songs.

