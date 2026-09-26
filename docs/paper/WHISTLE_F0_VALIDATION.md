# Whistle F0 validation on all MLEnd whistles (2026-09-26; downstream re-run with the fixed DTW, D-008)

Script: `docs/paper/scripts/validate_whistle_f0.py` (stages `extract`, `analyze`, `downstream`, `synth`, `humpeak`); contour
cleaning, spectral-peak track and DTW are shared with the contour figure through `docs/paper/scripts/contour_pipeline.py`.
Outputs, all in `docs/paper/results/` unless noted: `whistle_f0_validation.json` (dataset stats), `whistle_f0_validation_clips.csv`
(one row per clip), `docs/paper/figures/fig_whistle_f0_validation.{png,pdf}`, `whistle_f0_downstream{,_StarWars,_Hakuna}.json`
(contour-retrieval check, 3 songs) with per-pair `whistle_f0_downstream_pairs{,_StarWars,_Hakuna}.csv`, `whistle_f0_synth.json`
(known-F0 tones), `hum_spectral_peak_check.json` (hums: which harmonic is the spectral peak).
The first downstream run (before the DTW fix, D-008) is kept in `results/superseded/whistle_f0_downstream*_v1.json`.

## What was compared (per 10 ms frame)
- **RMVPE, half speed (fix):** clip loaded at 32 kHz and fed to RMVPE as if it were 16 kHz (i.e. played at half speed), F0 x 2, every 2nd frame kept. Voiced = salience >= 0.3 and 50 < F0 < 4200 Hz (same as `contour_figure.py`).
- **RMVPE, normal speed (no fix).**
- **Spectral peak:** STFT argmax in 200-6000 Hz (32 kHz, 64 ms Hann window zero-padded to 8192, parabolic interpolation on dB). Voicing gate: tonal ratio >= 6 dB (energy within +-50 cents of the peak vs the rest of 200-6000 Hz).
There is no ground truth in MLEnd: the numbers are agreement between two independent estimators. `synth` checks all three against known tones.

## Headline numbers (1,797 whistles listed, all 1,797 processed; 1,702 analyzed)
95 clips were excluded because RMVPE-fix voiced < 100 frames (1 s). Their median spectral peak is 3.74 kHz (73% above 2 kHz), and 39 of them have < 5% peak-voiced frames. So the exclusion removes mostly very high or barely whistled clips and **flatters** RMVPE-fix.

| per clip, voiced frames of RMVPE-fix | median | IQR | % clips > 90% |
|---|---|---|---|
| within 0.5 st of spectral peak, **with fix** | 98.2% | 96.7-99.1% | 91.1% |
| within 1 st, with fix | 99.5% | 98.4-99.8% | 92.1% |
| within 0.5 st, **no fix** (same frames) | 6.8% | 1.7-16.7% | 0.3% |
| octave-error rate, with fix | 0.0% | 0.0-0.34% (mean 3.4%) | |
| octave-error rate, no fix (same frames) | 55.4% | 37.6-66.4% | |

Pooled over 1,412,000 voiced frames: 95.9% within 0.5 st with the fix vs 14.2% without. Octave errors (|d - 12n| < 1 st, n != 0): 2.1% vs 52.9%.
Without the fix RMVPE also voices far fewer frames (median 16.6% of frames vs 52.8% with the fix).
Median whistle F0: 1,338 Hz (RMVPE-fix), 1,359 Hz (spectral peak).

## Failures (clips with < 90% agreement): 152 of 1,702 (8.9%)
- **Whistle pitch is the main driver, not noise.** By median spectral-peak frequency: < 1 kHz 96% of clips > 90% agreement (n = 76); 1-1.5 kHz 98.7% (n = 1,162); 1.5-2 kHz 85.3% (n = 382); 2-2.5 kHz 5.7% (n = 70, median agreement 20.7%); > 2.5 kHz 0% (n = 12).
- 51% of failing clips have a median peak above 2 kHz. Heuristic labels (overlapping): residual octave error (> 10% of frames) 123, very high pitch (> 2 kHz) 78, low SNR (< 15 dB) 2, breathy/noisy (tonal ratio < 0 dB) 2, unexplained 26.
- In the octave-error frames of failing clips, the spectrum at the frequency RMVPE-fix reports (f_peak/2) is -55.2 dB relative to the peak, the same as in passing clips (-55.4 dB). There is no component where RMVPE puts F0. So these are a *second* RMVPE octave error, and the spectral peak is the F0.
- Frame-level disagreement (4.1% of frames): 2.06% at -12 st (RMVPE-fix an octave below), 0.29% near -19 st (peak = 3 x RMVPE; harmonic-rich clips, where the peak is plausibly wrong), 1.7% other.
- Performers 220, 111, 170, 134 account for 54 of the 152 failures. Performer 170's clips (e.g. 0236) look harmonic-rich on inspection (components near 0.9, 1.9 and 2.8 kHz), and there the spectral peak, not RMVPE, is the suspect estimate.
- Pass vs fail medians: SNR 38.9 vs 34.2 dB; RMVPE-fix voiced fraction 0.54 vs 0.32; RMVPE salience 0.78 vs 0.55. Spearman correlations with agreement: salience 0.44, voiced fraction 0.39, SNR 0.11.
- The worst clips are listed in `whistle_f0_validation.json` -> `failures.worst_25`. The first 12 are 2258, 3828, 5435, 5812, 0236, 2899, 6234, 3482, 5096, 3134, 3568 and 1565 (all with a median peak of 2.3-4.1 kHz).

## Synthetic tones (known F0, +-2 st glide, weak 2nd harmonic, pink-ish noise at 30/10/0 dB SNR)
- RMVPE, no fix: correct at 500-700 Hz, 82-87% at 1 kHz, <= 6% at 1.4 kHz, 0% at >= 2 kHz.
- RMVPE-fix: 100% up to 1.4 kHz, 74-85% at 2 kHz, <= 10% at >= 2.8 kHz (its output ceiling is 2 x 2005.5 = 4011 Hz).
- Spectral peak: 100% at every frequency when voiced. With the 6 dB tonal gate it voices nothing at 0 dB broadband SNR, while RMVPE-fix still voices <= 1.4 kHz tones at 0 dB.

## Spectral peak alone as the whistle F0
- Agreement with RMVPE-fix on frames both voice: median 99.7% within 0.5 st (IQR 99.1-99.9%). Voicing vs RMVPE-fix: precision 0.82, recall 0.93 (per-clip medians).
- Downstream (hum vs same-performer whistle, same song vs other songs, `contour_pipeline.py`, the same code as the figure).
  Re-run on 2026-09-26 with the fixed DTW (D-008: the whistle is resampled to the hum's length, then slope-constrained DTW;
  no fallback, 0 unaligned pairs for every method and song). AUC by MAE on the pairs every non-raw method can score
  (1,044 / 1,204 / 1,196 pairs; 112 / 118 / 116 performers), 95% CI from a performer-level bootstrap (1,000 resamples):

  | target song | spectral peak | RMVPE-fix | difference (95% CI) | v1 difference (fallback DTW) |
  |---|---|---|---|---|
  | Potter | 0.851 | 0.811 | +0.040 (+0.020 to +0.063) | +0.073 (+0.029 to +0.122) |
  | StarWars | 0.773 | 0.717 | +0.056 (+0.027 to +0.086) | +0.063 (+0.024 to +0.102) |
  | Hakuna | 0.823 | 0.795 | +0.028 (+0.009 to +0.048) | +0.029 (+0.005 to +0.054) |

  Each method on its own pairs (AUC by MAE, 95% performer-bootstrap CI; v1 value in brackets):

  | method | Potter | StarWars | Hakuna |
  |---|---|---|---|
  | spectral peak, tonal ≥ 6 dB | 0.828 (0.790–0.865) [0.813] | 0.757 (0.709–0.799) [0.759] | 0.812 (0.769–0.857) [0.820] |
  | RMVPE-fix (half speed ×2) | 0.791 (0.750–0.838) [0.750] | 0.713 (0.669–0.758) [0.708] | 0.789 (0.737–0.841) [0.788] |
  | RMVPE, normal speed | 0.491 (0.445–0.539) [0.490] | 0.534 (0.490–0.575) [0.524] | 0.503 (0.438–0.562) [0.566] |

  The fix roughly halves the peak's advantage on Potter (+0.073 → +0.040). Potter is where RMVPE-fix had the largest share of
  pairs on the old fallback (189 of 1,230, vs 154 of 1,222 on StarWars and 132 of 1,214 on Hakuna), and its own AUC rose from
  0.750 to 0.791 once they were constrained. The advantage still excludes 0 on all three songs.
- RMVPE-fix on the new DTW reproduces the figure pipeline exactly: r 0.824 / 0.589, AUC 0.791 in both
  `whistle_f0_downstream.json` and `fig_contours_potter_rmvpefix_stats.json`; the peak method gives r 0.854 / 0.589, AUC 0.828
  in both `whistle_f0_downstream.json` and `fig_contours_potter_stats.json`.
- **Controls** (paired vs RMVPE-fix, Potter / StarWars / Hakuna): peak F0 on RMVPE-fix's voiced frames +0.014 (+0.001 to +0.028) /
  +0.018 (+0.005 to +0.034) / +0.009 (−0.002 to +0.023); RMVPE-fix F0 on the peak's voiced frames −0.022 (−0.043 to −0.003) /
  −0.011 (−0.033 to +0.009) / −0.003 (−0.020 to +0.014). With the fixed DTW the peak's F0 values alone give a small gain (CI above 0 on
  Potter and StarWars), and the peak's voicing alone does not help. Most of the gain still needs both together
  (v1 said neither alone mattered; that no longer holds for the F0 values).
- **Voicing threshold matters:** with prom ≥ 60 dB (too strict) the peak method is worse than RMVPE-fix on Potter
  (−0.064, CI −0.100 to −0.029; v1 −0.069). Stricter voicing gives shorter sequences: 515 of 1,048 Potter pairs have a
  voiced-length ratio outside [1/2, 2] with prom ≥ 60 dB vs 116 of 1,288 with the 6 dB tonal gate. Since D-008 these pairs are
  length-normalized and slope-constrained instead of aligned without a slope limit.
- RMVPE without the fix is at chance downstream (AUC 0.49 / 0.53 / 0.50). Paired vs RMVPE-fix on the pairs both score:
  −0.336 / −0.239 / −0.351, CIs far below 0. Under v1 its same-song and other-song r were both ≈ 0.80 (fallback on 824 of 922 Potter
  pairs); with the constrained DTW they drop to 0.52 / 0.53, which is what unconstrained warping hid.

## Recommendation
Use the **spectral peak with the 6 dB tonal-ratio voicing gate** as the whistle F0 in the paper (keep RMVPE for hums), and report RMVPE-fix agreement as a cross-check. See `docs/DECISIONS.md` D-007 in the repo.

## Compute note
The CPU on the Lambda box was saturated by the training data loaders (load average 105-134 on 30 cores). At nice 10 each worker got ~5% of a core, a projected 12+ hours. So 1,651 clips were computed on a separate idle CPU machine (2 processes x 3 threads, the same `spectral_track`/`rmvpe_track` code as the `extract` stage) and 146 on Lambda (nice 10, 8 processes, CPU only). A clip computed on both machines matched to < 0.0001 st. The GPU was not touched and no process was killed.
The downstream re-run with the fixed DTW (D-008) and the regenerated contour figure ran on that CPU machine (about 4 minutes for
the three songs in parallel); Lambda was not used.
