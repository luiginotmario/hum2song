# D-042 results: whistle + wrong-lyrics without MLEnd training data

Copied from `/lambda/nfs/hum2song-data/results/d042/` on 2026-10-09. One chain (`run_chain.sh`) scored all five checkpoints on identical sets. Every number below is printed by `python summarize.py` from the JSON files in this folder.

**Labels.** **[synthetic]** = rendered queries, not recordings of people. **[real]** = recordings of people. D-042 is the D-041 config with `mlend_pairs: false` and `mlend_val: false` (the only two config changes; compare `train_contour_qsynth_d042.yaml` with `configs/train_contour_qsynth_d041.yaml`). D-042, D-040 and E2b never trained on MLEnd, so for them all 8 MLEnd songs are a clean real test. D-041 trained on MLEnd train-performer whistles of the 6 songs other than Hakuna and Potter, so its other-6 and all-8 rows are **not clean**.

**Run.** `train_contour_qsynth_d042.yaml`, init E2b, 1 seed, 12,000 steps. Early stop did not fire (best step 11,250 was within the last 8 validations); `last.pt` is step 12,000. The HumTrans guard held at every validation (`guard_ok` = 1). `train exit=0`, all evals `exit=0`, `ALLDONE` (`chain.log`, times in UTC).

**MLEnd counts.** `counts.whistle_peak.clips` = 339 test whistles (26 test whistlers). Held-out songs (Hakuna + Potter) = 90, other 6 = 249; top-1 × count is a whole number for every model and section, which confirms the split.

## Comparison
| Synthetic hold-out [synthetic], top-1 / top-10 / MRR | D-042 best.pt (step 11250) | D-042 last.pt (step 12000) | D-041 best.pt (step 9500) | D-040 last.pt (step 3000) | E2b (live) (step 800) |
| --- | --- | --- | --- | --- | --- |
| Synthetic whistle (4,263 q) | 0.251 / 0.374 / 0.295 | 0.252 / 0.375 / 0.295 | 0.253 / 0.378 / 0.297 | 0.226 / 0.342 / 0.267 | 0.180 / 0.291 / 0.220 |
| Synthetic wrong-lyrics (4,263 q) | 0.264 / 0.399 / 0.312 | 0.265 / 0.398 / 0.313 | 0.269 / 0.401 / 0.314 | 0.238 / 0.365 / 0.283 | 0.193 / 0.314 / 0.235 |
| Both styles | 0.258 / 0.387 / 0.303 | 0.258 / 0.387 / 0.304 | 0.261 / 0.389 / 0.305 | 0.232 / 0.354 / 0.275 | 0.186 / 0.303 / 0.227 |

| Real hums vs MIDI [real], top-1 / top-10 / MRR | D-042 best.pt (step 11250) | D-042 last.pt (step 12000) | D-041 best.pt (step 9500) | D-040 last.pt (step 3000) | E2b (live) (step 800) |
| --- | --- | --- | --- | --- | --- |
| HumTrans test (769) | 0.992 / 1.000 / 0.996 | 0.992 / 1.000 / 0.996 | 0.993 / 1.000 / 0.997 | 0.991 / 1.000 / 0.995 | 0.996 / 1.000 / 0.998 |
| HumTrans test +7 st (769) | 0.993 / 1.000 / 0.996 | 0.993 / 1.000 / 0.996 | 0.995 / 1.000 / 0.997 | 0.995 / 1.000 / 0.997 | 0.996 / 1.000 / 0.998 |
| MIR-QBSH 48 anywhere (4431) | 0.950 / 0.992 / 0.966 | 0.949 / 0.992 / 0.966 | 0.949 / 0.991 / 0.965 | 0.944 / 0.991 / 0.963 | 0.955 / 0.993 / 0.970 |

| MLEnd test people, whistle_peak->hum/centroid [real], top-1 / top-3 / MRR | D-042 best.pt (step 11250) | D-042 last.pt (step 12000) | D-041 best.pt (step 9500) | D-040 last.pt (step 3000) | E2b (live) (step 800) |
| --- | --- | --- | --- | --- | --- |
| MLEnd held-out songs (Hakuna+Potter) [real, clean for all] | 0.622 / 0.722 / 0.709 | 0.611 / 0.711 / 0.703 | 0.456 / 0.700 / 0.620 | 0.656 / 0.789 / 0.741 | 0.589 / 0.711 / 0.698 |
| MLEnd other 6 songs [real, clean except D-041] | 0.618 / 0.763 / 0.724 | 0.618 / 0.763 / 0.724 | 0.707 / 0.863 / 0.803 | 0.594 / 0.779 / 0.720 | 0.647 / 0.791 / 0.742 |
| MLEnd all 8 songs [real, clean except D-041] | 0.619 / 0.752 / 0.720 | 0.617 / 0.749 / 0.719 | 0.640 / 0.820 / 0.754 | 0.611 / 0.782 / 0.725 | 0.631 / 0.770 / 0.731 |

| MLEnd correct top-1 / whistles [real] | D-042 best.pt | D-042 last.pt | D-041 best.pt | D-040 last.pt | E2b (live) |
| --- | --- | --- | --- | --- | --- |
| MLEnd held-out songs (Hakuna+Potter) [real, clean for all] | 56 / 90 | 55 / 90 | 41 / 90 | 59 / 90 | 53 / 90 |
| MLEnd other 6 songs [real, clean except D-041] | 154 / 249 | 154 / 249 | 176 / 249 | 148 / 249 | 161 / 249 |
| MLEnd all 8 songs [real, clean except D-041] | 210 / 339 | 209 / 339 | 217 / 339 | 207 / 339 | 214 / 339 |

**Reading.**
- **Synthetic [synthetic]:** D-042 best is within 0.005 top-1 of D-041 best (whistle 0.251 vs 0.253, wrong-lyrics 0.264 vs 0.269) and +0.072 / +0.071 over E2b. Dropping the MLEnd pairs cost almost nothing on the synthetic test. Same generator as training (new songs), so this is fit to the generator.
- **MLEnd held-out songs [real, clean]:** D-042 best 56 / 90 (0.622) vs E2b 53 (0.589), D-040 59 (0.656), D-041 41 (0.456). Removing the MLEnd pairs undid D-041's held-out drop. +3 of 90 over E2b is well inside the ~0.05 binomial standard error.
- **MLEnd other 6 songs [real, clean for D-042]:** D-042 best 154 / 249 (0.618) vs E2b 161 (0.647).
- **MLEnd all 8 songs [real, clean for D-042]:** D-042 best 210 / 339 (0.619) vs E2b 214 (0.631); top-3 0.752 vs 0.770. D-042 does **not** beat E2b on the full clean real test.
- **Hums [real]:** HumTrans test top-1 0.992 vs E2b 0.996 (763 vs 766 of 769); MIR-QBSH 48 0.950 vs 0.955 (4,209 vs 4,232 of 4,431). Small drops, under 1 point.
- **Decision:** no swap. D-042 does not beat E2b on real MLEnd (all 8) and costs hums slightly. E2b stays live.
- The D-041 best, D-040 last and E2b rows equal the committed `../d041/` files (re-scored in this chain).

## Selection curve (synthetic val, ~1.1k val songs; not comparable to the 11k-song test numbers)

Guard: HumTrans val +7 st top-1 at most 1 point below step 0 (0.9948).

| step | val/select | val/qsynth_whistle_top1 | val/qsynth_wrong_lyrics_top1 | val/humtrans_val_shift+7_top1 | val/guard_ok |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.2877 | 0.2817 | 0.2937 | 0.9948 | 1.0000 |
| 250 | 0.3013 | 0.2850 | 0.3177 | 0.9922 | 1.0000 |
| 500 | 0.3085 | 0.2927 | 0.3243 | 0.9922 | 1.0000 |
| 750 | 0.3145 | 0.2990 | 0.3300 | 0.9935 | 1.0000 |
| 1000 | 0.3160 | 0.3003 | 0.3317 | 0.9922 | 1.0000 |
| 1250 | 0.3238 | 0.3060 | 0.3417 | 0.9935 | 1.0000 |
| 1500 | 0.3265 | 0.3087 | 0.3443 | 0.9935 | 1.0000 |
| 1750 | 0.3308 | 0.3133 | 0.3483 | 0.9935 | 1.0000 |
| 2000 | 0.3332 | 0.3170 | 0.3493 | 0.9935 | 1.0000 |
| 2250 | 0.3362 | 0.3153 | 0.3570 | 0.9935 | 1.0000 |
| 2500 | 0.3380 | 0.3163 | 0.3597 | 0.9935 | 1.0000 |
| 2750 | 0.3412 | 0.3200 | 0.3623 | 0.9922 | 1.0000 |
| 3000 | 0.3427 | 0.3213 | 0.3640 | 0.9922 | 1.0000 |
| 3250 | 0.3418 | 0.3237 | 0.3600 | 0.9922 | 1.0000 |
| 3500 | 0.3477 | 0.3263 | 0.3690 | 0.9935 | 1.0000 |
| 3750 | 0.3475 | 0.3250 | 0.3700 | 0.9935 | 1.0000 |
| 4000 | 0.3507 | 0.3290 | 0.3723 | 0.9935 | 1.0000 |
| 4250 | 0.3565 | 0.3370 | 0.3760 | 0.9935 | 1.0000 |
| 4500 | 0.3548 | 0.3330 | 0.3767 | 0.9935 | 1.0000 |
| 4750 | 0.3525 | 0.3330 | 0.3720 | 0.9935 | 1.0000 |
| 5000 | 0.3542 | 0.3320 | 0.3763 | 0.9935 | 1.0000 |
| 5250 | 0.3577 | 0.3380 | 0.3773 | 0.9922 | 1.0000 |
| 5500 | 0.3583 | 0.3350 | 0.3817 | 0.9922 | 1.0000 |
| 5750 | 0.3607 | 0.3400 | 0.3813 | 0.9908 | 1.0000 |
| 6000 | 0.3585 | 0.3360 | 0.3810 | 0.9922 | 1.0000 |
| 6250 | 0.3595 | 0.3363 | 0.3827 | 0.9935 | 1.0000 |
| 6500 | 0.3608 | 0.3377 | 0.3840 | 0.9935 | 1.0000 |
| 6750 | 0.3620 | 0.3407 | 0.3833 | 0.9935 | 1.0000 |
| 7000 | 0.3648 | 0.3403 | 0.3893 | 0.9908 | 1.0000 |
| 7250 | 0.3670 | 0.3433 | 0.3907 | 0.9922 | 1.0000 |
| 7500 | 0.3662 | 0.3410 | 0.3913 | 0.9922 | 1.0000 |
| 7750 | 0.3688 | 0.3443 | 0.3933 | 0.9935 | 1.0000 |
| 8000 | 0.3678 | 0.3423 | 0.3933 | 0.9922 | 1.0000 |
| 8250 | 0.3697 | 0.3453 | 0.3940 | 0.9922 | 1.0000 |
| 8500 | 0.3700 | 0.3447 | 0.3953 | 0.9922 | 1.0000 |
| 8750 | 0.3705 | 0.3453 | 0.3957 | 0.9935 | 1.0000 |
| 9000 | 0.3703 | 0.3440 | 0.3967 | 0.9935 | 1.0000 |
| 9250 | 0.3707 | 0.3453 | 0.3960 | 0.9922 | 1.0000 |
| 9500 | 0.3710 | 0.3467 | 0.3953 | 0.9922 | 1.0000 |
| 9750 | 0.3735 | 0.3480 | 0.3990 | 0.9922 | 1.0000 |
| 10000 | 0.3725 | 0.3493 | 0.3957 | 0.9922 | 1.0000 |
| 10250 | 0.3735 | 0.3483 | 0.3987 | 0.9922 | 1.0000 |
| 10500 | 0.3730 | 0.3483 | 0.3977 | 0.9922 | 1.0000 |
| 10750 | 0.3740 | 0.3490 | 0.3990 | 0.9922 | 1.0000 |
| 11000 | 0.3742 | 0.3487 | 0.3997 | 0.9922 | 1.0000 |
| 11250 | 0.3743 | 0.3490 | 0.3997 | 0.9935 | 1.0000 |
| 11500 | 0.3740 | 0.3487 | 0.3993 | 0.9922 | 1.0000 |
| 11750 | 0.3742 | 0.3490 | 0.3993 | 0.9922 | 1.0000 |
| 12000 | 0.3740 | 0.3487 | 0.3993 | 0.9935 | 1.0000 |
