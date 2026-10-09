# D-041 results: whistle + wrong-lyrics, round 2

Copied from `/lambda/nfs/hum2song-data/results/d041/` on 2026-10-09. One chain (`run_chain.sh`) scored all four checkpoints on identical sets. Every number below is printed by `python summarize.py` from the JSON files in this folder.

**Labels.** **[synthetic]** = rendered queries, not recordings of people. **[real]** = recordings of people. For MLEnd, only the **held-out songs** row (Hakuna + Potter, 90 whistles, 26 test whistlers) is a clean real number for D-041: D-041 trained on MLEnd train-performer whistle→hum pairs of the other 6 songs, so its **seen songs** row measures song familiarity (as in D-016) and is **not clean**. E2b and D-040 never trained on MLEnd.

**Run.** `configs/train_contour_qsynth_d041.yaml`, init E2b, 1 seed. Attempt 1 hit CUDA OOM at step ~3,900 of 12,000 (`attempt1_oom/`, best.pt step 3,750 archived on Lambda). Attempt 2 was relaunched from scratch at 02:32 ET on 2026-10-09 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, same config and code; it early-stopped at step 11,500 (8 validations without a better synthetic-val score), best step 9,500. `train exit=0`, `ALLDONE` (`chain.log`, times in UTC).

## Comparison
| Synthetic hold-out [synthetic], top-1 / top-10 / MRR | D-041 best.pt (step 9500, selected on synthetic val) | D-041 last.pt (step 11500, early stop) | D-040 last.pt (step 3000) | E2b (live, step 800) |
| --- | --- | --- | --- | --- |
| Synthetic whistle (4,263 q) | 0.253 / 0.378 / 0.297 | 0.254 / 0.379 / 0.298 | 0.226 / 0.342 / 0.267 | 0.180 / 0.291 / 0.220 |
| Synthetic wrong-lyrics (4,263 q) | 0.269 / 0.401 / 0.314 | 0.269 / 0.403 / 0.315 | 0.238 / 0.365 / 0.283 | 0.193 / 0.314 / 0.235 |
| Both styles (8,526 q) | 0.261 / 0.389 / 0.305 | 0.262 / 0.391 / 0.306 | 0.232 / 0.354 / 0.275 | 0.186 / 0.303 / 0.227 |

| Real hums vs MIDI [real], top-1 / top-10 / MRR | D-041 best.pt (step 9500, selected on synthetic val) | D-041 last.pt (step 11500, early stop) | D-040 last.pt (step 3000) | E2b (live, step 800) |
| --- | --- | --- | --- | --- |
| HumTrans test (769) | 0.993 / 1.000 / 0.997 | 0.993 / 1.000 / 0.996 | 0.991 / 1.000 / 0.995 | 0.996 / 1.000 / 0.998 |
| HumTrans test +7 st (769) | 0.995 / 1.000 / 0.997 | 0.995 / 1.000 / 0.997 | 0.995 / 1.000 / 0.997 | 0.996 / 1.000 / 0.998 |
| MIR-QBSH 48 anywhere (4431) | 0.949 / 0.991 / 0.965 | 0.947 / 0.991 / 0.964 | 0.944 / 0.991 / 0.963 | 0.955 / 0.993 / 0.970 |

| MLEnd test people, whistle_peak->hum/centroid [real], top-1 / top-3 / MRR | D-041 best.pt (step 9500, selected on synthetic val) | D-041 last.pt (step 11500, early stop) | D-040 last.pt (step 3000) | E2b (live, step 800) |
| --- | --- | --- | --- | --- |
| MLEnd held-out songs (Hakuna+Potter) [real, clean] | 0.456 / 0.700 / 0.620 | 0.456 / 0.678 / 0.619 | 0.656 / 0.789 / 0.741 | 0.589 / 0.711 / 0.698 |
| MLEnd seen songs (other 6) [real, NOT clean for D-041] | 0.707 / 0.863 / 0.803 | 0.707 / 0.867 / 0.804 | 0.594 / 0.779 / 0.720 | 0.647 / 0.791 / 0.742 |
| MLEnd all 8 [real, mixed] | 0.640 / 0.820 / 0.754 | 0.640 / 0.817 / 0.755 | 0.611 / 0.782 / 0.725 | 0.631 / 0.770 / 0.731 |

**Reading.**
- **Synthetic [synthetic]:** D-041 best beats E2b by +0.073 whistle and +0.076 wrong-lyrics top-1, and D-040 by +0.027 / +0.031. Same generator as training (new songs), so this is fit to the generator.
- **MLEnd held-out songs [real, clean]:** D-041 best 0.456 vs E2b 0.589 vs D-040 0.656 top-1 (41 vs 53 vs 59 of 90 whistles). D-041 is **worse** than E2b on the only clean real whistle number. It is the same value D-016 got with MLEnd whistle pairs and the same two songs held out (0.456).
- **MLEnd seen songs [real, not clean]:** 0.707 vs E2b 0.647. This gain is on songs D-041 trained on (other performers), so it is song familiarity, not evidence of whistle robustness.
- **Hums [real]:** HumTrans test top-1 0.993 vs E2b 0.996; MIR-QBSH 48 0.949 vs 0.955. Small drops, within the 1-point guard.
- **Decision:** D-041 does not beat E2b on real held-out whistles. E2b stays live; no swap.
- The D-040 last.pt and E2b rows equal the committed `../d040/` files exactly (re-scored in this chain).

## Checkpoint steps

- D-041 best.pt (step 9500, selected on synthetic val): step 9500
- D-041 last.pt (step 11500, early stop): step 11500
- D-040 last.pt (step 3000): step 3000
- E2b (live, step 800): step 800

## Selection curve (synthetic val, ~1.1k val songs; not comparable to the 11k-song test numbers)

Guard: HumTrans val +7 st top-1 at most 1 point below step 0 (0.9948); `guard_ok` was 1 at every validation.

| step | val/select | val/qsynth_whistle_top1 | val/qsynth_wrong_lyrics_top1 | val/humtrans_val_shift+7_top1 |
| --- | --- | --- | --- | --- |
| 0 | 0.2877 | 0.2817 | 0.2937 | 0.9948 |
| 250 | 0.3022 | 0.2860 | 0.3183 | 0.9948 |
| 500 | 0.3105 | 0.2937 | 0.3273 | 0.9935 |
| 750 | 0.3127 | 0.2963 | 0.3290 | 0.9935 |
| 1000 | 0.3198 | 0.3027 | 0.3370 | 0.9922 |
| 1250 | 0.3290 | 0.3113 | 0.3467 | 0.9935 |
| 1500 | 0.3257 | 0.3077 | 0.3437 | 0.9935 |
| 1750 | 0.3327 | 0.3137 | 0.3517 | 0.9935 |
| 2000 | 0.3353 | 0.3140 | 0.3567 | 0.9948 |
| 2250 | 0.3360 | 0.3143 | 0.3577 | 0.9935 |
| 2500 | 0.3388 | 0.3173 | 0.3603 | 0.9935 |
| 2750 | 0.3467 | 0.3253 | 0.3680 | 0.9935 |
| 3000 | 0.3412 | 0.3203 | 0.3620 | 0.9935 |
| 3250 | 0.3432 | 0.3200 | 0.3663 | 0.9922 |
| 3500 | 0.3468 | 0.3240 | 0.3697 | 0.9935 |
| 3750 | 0.3517 | 0.3280 | 0.3753 | 0.9922 |
| 4000 | 0.3497 | 0.3277 | 0.3717 | 0.9922 |
| 4250 | 0.3535 | 0.3310 | 0.3760 | 0.9922 |
| 4500 | 0.3543 | 0.3337 | 0.3750 | 0.9935 |
| 4750 | 0.3570 | 0.3360 | 0.3780 | 0.9935 |
| 5000 | 0.3577 | 0.3380 | 0.3773 | 0.9935 |
| 5250 | 0.3580 | 0.3343 | 0.3817 | 0.9922 |
| 5500 | 0.3608 | 0.3430 | 0.3787 | 0.9935 |
| 5750 | 0.3635 | 0.3403 | 0.3867 | 0.9935 |
| 6000 | 0.3612 | 0.3373 | 0.3850 | 0.9935 |
| 6250 | 0.3663 | 0.3423 | 0.3903 | 0.9922 |
| 6500 | 0.3662 | 0.3413 | 0.3910 | 0.9922 |
| 6750 | 0.3693 | 0.3453 | 0.3933 | 0.9908 |
| 7000 | 0.3683 | 0.3420 | 0.3947 | 0.9922 |
| 7250 | 0.3693 | 0.3437 | 0.3950 | 0.9935 |
| 7500 | 0.3678 | 0.3447 | 0.3910 | 0.9935 |
| 7750 | 0.3710 | 0.3483 | 0.3937 | 0.9922 |
| 8000 | 0.3698 | 0.3483 | 0.3913 | 0.9922 |
| 8250 | 0.3678 | 0.3457 | 0.3900 | 0.9935 |
| 8500 | 0.3722 | 0.3467 | 0.3977 | 0.9935 |
| 8750 | 0.3732 | 0.3477 | 0.3987 | 0.9922 |
| 9000 | 0.3728 | 0.3483 | 0.3973 | 0.9922 |
| 9250 | 0.3717 | 0.3463 | 0.3970 | 0.9922 |
| 9500 | 0.3742 | 0.3510 | 0.3973 | 0.9935 |
| 9750 | 0.3722 | 0.3470 | 0.3973 | 0.9935 |
| 10000 | 0.3738 | 0.3487 | 0.3990 | 0.9935 |
| 10250 | 0.3728 | 0.3477 | 0.3980 | 0.9935 |
| 10500 | 0.3733 | 0.3487 | 0.3980 | 0.9935 |
| 10750 | 0.3723 | 0.3477 | 0.3970 | 0.9922 |
| 11000 | 0.3735 | 0.3483 | 0.3987 | 0.9908 |
| 11250 | 0.3728 | 0.3477 | 0.3980 | 0.9922 |
| 11500 | 0.3720 | 0.3467 | 0.3973 | 0.9922 |


## Files
| File | What |
|---|---|
| `run_chain.sh`, `chain.log` | train + all evals (attempt 2) |
| `train.log` | attempt 2 training log |
| `attempt1_oom/chain.log`, `attempt1_oom/train.log` | attempt 1 (CUDA OOM at step ~3,900) |
| `d041_code.patch` | code diff used for the run (applied on main) |
| `eval_<tag>.json` | synthetic hold-out (both styles) + hum baseline; blocked real sets listed |
| `eval_style_<tag>.json` | synthetic hold-out per style |
| `eval_mlend_<tag>.json` | MLEnd test people (`--split test --song-holdout`) |
| `summarize.py` | prints every table above |

Tags: `d041_best`, `d041_last`, `d040_last`, `e2b`.
