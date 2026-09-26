# Figure: hum vs whistle share one melody contour (MLEnd "Potter")

Files (repo paths):
- Figure: `docs/paper/figures/fig_contours_potter.png` (300 dpi), `docs/paper/figures/fig_contours_potter.pdf` (vector, fonts embedded).
- Numbers: `docs/paper/results/fig_contours_potter_stats.json`, `docs/paper/results/fig_contours_potter_pairs.json` (all 440 same-song hum–whistle pairs).
- Cross-check with RMVPE-fix whistles (same pipeline, same DTW): `docs/paper/results/fig_contours_potter_rmvpefix_stats.json`.
- Superseded first version (RMVPE-fix whistles, DTW with silent fallback): `docs/paper/results/superseded/fig_contours_potter_stats_v1.json`.
- Code: `docs/paper/scripts/contour_figure.py`, shared pipeline `docs/paper/scripts/contour_pipeline.py`, patched RVC `rmvpe_rvc.py` (CUDA-graph import stubbed out).

Regenerate (CPU; caches from `contour_figure.py extract` for hums and `validate_whistle_f0.py extract` for whistles):
```
python contour_figure.py plot --cache f0cache_potter --whistle-cache wcache --whistle-f0 peak \
    --outdir . --tag potter --null-songs StarWars,Hakuna --null-cache-pattern "f0cache_{song}" --interpreter 213
```
Your own recordings: `contour_figure.py extract --custom-dir DIR --out CACHE` then `plot --custom-dir DIR --cache CACHE ...`
(files named `*hum*`, `*whistle*`, `*sing*`; three lines drawn when sung clips exist).

## Methods, in brief
- Hum pitch: RMVPE (Wei et al., 2023), RVC `rmvpe.pt` checkpoint, run on CPU, 10 ms hop. Voiced = salience peak ≥ 0.3,
  50 Hz < F0 < 4.2 kHz.
- Whistle pitch: STFT spectral peak in 200–6000 Hz (32 kHz, 64 ms Hann window zero-padded to 8192, parabolic
  interpolation), voiced when the tonal ratio (energy within ±50 cents of the peak over the rest of the band) is ≥ 6 dB
  (D-007; validation in `docs/paper/WHISTLE_F0_VALIDATION.md`). RMVPE at half speed ×2 ("RMVPE-fix") is kept as a cross-check.
- Cleaning, both types: voiced runs < 50 ms dropped, isolated octave jumps (> 9 st from a 0.5 s running median) folded
  back, 5-frame median filter.
- Key normalization: MIDI semitones minus the clip's own median. Voiced frames only, averaged to 20 ms frames.
- DTW (D-008): the whistle sequence is linearly resampled to the length of the reference hum, then aligned with cost
  |Δsemitone|, slope-constrained steps (1,1),(1,2),(2,1) (local tempo ratio 1/2–2) inside a 25 % Sakoe–Chiba band.
  Because both sequences then have the same length, a slope-constrained path always exists; there is no fallback. A pair
  without a valid path would be counted as unaligned and left out; none occurred (0 of 5,586 pairs in this figure's stats).
  The first version silently fell back to unconstrained steps when the voiced lengths differed by more than 2×
  (189 of 1,230 hum–whistle pairs with RMVPE-fix whistles). Unconstrained DTW was rejected because it also fits contours from
  *different* songs: in the first version, RMVPE at normal speed (89 % of pairs on the fallback) scored median r 0.80
  same-song vs 0.79 other-song (`results/superseded/whistle_f0_downstream_v1.json`); with the constrained DTW it is
  0.52 vs 0.53 (`results/whistle_f0_downstream.json`).

## Numbers (MLEnd Potter; 114 performers, 440 same-performer hum–whistle pairs)
| comparison (after key normalization + DTW) | median MAE (st) | median Pearson r |
|---|---|---|
| **Best case:** featured performer 213, whistle 0008 vs reference hum 0491 (panel b) | 0.44 | 0.98 |
| Featured performer 213, all 4 hum×whistle pairs | 0.44–0.54 | 0.976–0.982 |
| All performers: hum vs whistle, same song (440 pairs) | 1.42 (mean 1.55) | 0.85 |
| Ceiling: hum vs hum, same performer, same song (654 pairs) | 0.73 | 0.95 |
| Null: hum (Potter) vs same performer's whistle of another song (StarWars/Hakuna, 848 pairs) | 2.35 | 0.59 |
| Null: hum vs same performer's hum of another song (3,644 pairs) | 2.16 | 0.66 |

AUC (same-song vs other-song whistle, using MAE) = 0.828, 95 % performer-bootstrap CI 0.790–0.865
(`results/whistle_f0_downstream.json`, method `peak_tonal_db6`, which reproduces the figure's pairs exactly).
Transposition offset, whistle minus hum (median of per-clip medians): featured performer +28.2 st; all pairs
median +35.8 st (IQR +32.1 to +38.6), i.e. about 2.5–3 octaves, NOT about +12. MLEnd hums have a median of about 180 Hz, and
whistles a median of about 1.36 kHz (spectral peak).

Performer 213 is a **best case**: it ranks 1st of 114 performers by mean hum–whistle MAE (`featured_rank` in the stats JSON).
It was first chosen for clean recordings *and* good hum/whistle agreement, so panel (b) is illustrative, not typical.
Panel (c) performers were chosen by pitch-track jitter only (with spectral-peak whistles the set changed: 213, 170, 90, 97,
136, 52, 49, 207). Performer 170 has harmonic-rich whistles where the spectral peak can sit on the 3rd harmonic (SIGNALS §S3).
Across the dataset, whistles match the hum less well than a second hum does, and whistlers often compress large leaps
(panel c: the whistle median undershoots the +8 st peak).

### Old vs new (what changed and why)
| | v1: RMVPE-fix, DTW with fallback | RMVPE-fix, fixed DTW | **v2 (current): spectral peak, fixed DTW** |
|---|---|---|---|
| performers / same-song pairs / other-song pairs | 111 / 424 / 806 | 111 / 424 / 806 | 114 / 440 / 848 |
| same-song median MAE / r | 1.63 / 0.80 | 1.54 / 0.82 | 1.42 / 0.85 |
| other-song median MAE / r | 2.34 / 0.57 | 2.37 / 0.59 | 2.35 / 0.59 |
| AUC by MAE | 0.750 | 0.791 | 0.828 |
| ceiling hum–hum MAE / r (654 pairs) | 0.75 / 0.95 | 0.73 / 0.95 | 0.73 / 0.95 |
| null hum–hum MAE / r (3,644 pairs) | 2.23 / 0.64 | 2.16 / 0.66 | 2.16 / 0.66 |
| pairs with voiced-length ratio outside 1/2–2 | 189 of 1,230 aligned without slope limit | 189 of 1,230, all slope-constrained | 116 of 1,288, all slope-constrained |
| featured 213 (panel b clip) MAE / r / offset | 0681: 0.51 / 0.99 / +27.9 st | 0008: 0.43 / 0.99 / +27.9 st | 0008: 0.44 / 0.98 / +28.2 st |
| whistle − hum offset, median (IQR) | +35.2 (+31.2 to +37.3) st | same | +35.8 (+32.1 to +38.6) st |

## Draft caption
**Figure X.** Hummed and whistled renditions of the same melody (MLEnd "Potter" theme) have the same pitch contour
but sit in very different registers. (a) F0 of one performer's hum (RMVPE) and whistle (spectral peak), voiced frames only,
log-frequency axis. The whistle is about 28 semitones (≈2.3 octaves) higher. (b) After subtracting each clip's median pitch
(removing transposition) and aligning the whistle to the hum with slope-constrained DTW, the two contours overlap
(MAE 0.44 st, r = 0.98; this performer is the best of 114 and shown as a best case). The grey band is the per-frame min–max over
this performer's two hums and two whistles. (c) The same normalization and alignment for 16 hums and 16 whistles from
8 performers (thin lines) with per-type medians (thick). Across all 114 performers, same-song hum–whistle pairs agree far
better than pairs from different songs (median r 0.85 vs 0.59; AUC 0.83).

## References (all links checked on 2026-09-26 unless marked)

### (1) Melody identity = relative pitch / contour, transposition-invariant
1. Dowling, W. J., & Fujitani, D. S. (1971). Contour, interval, and pitch recognition in memory for melodies. *JASA*, 49(2B), 524–531. https://doi.org/10.1121/1.1912382
   Supports: listeners recognise transposed melodies mainly by their contour, and familiar tunes stay recognisable when contour plus relative interval sizes are kept.
2. Dowling, W. J. (1978). Scale and contour: Two components of a theory of memory for melodies. *Psychological Review*, 85(4), 341–354. https://doi.org/10.1037/0033-295X.85.4.341
   Supports: melodies are stored as a contour mapped onto a scale. Exact transpositions (and same-contour tonal answers) get confused with the standard.
3. Attneave, F., & Olson, R. K. (1971). Pitch as a medium: A new approach to psychophysical scaling. *American Journal of Psychology*, 84(2), 147–166. https://doi.org/10.2307/1421351
   Supports: people transpose melodic patterns along a log-frequency (musical) scale, which justifies measuring in semitones and subtracting the median. Transposition breaks down above about 5 kHz.
4. Deutsch, D. (1972). Octave generalization and tune recognition. *Perception & Psychophysics*, 11(6), 411–412. https://doi.org/10.3758/BF03206280 (PDF: https://deutsch.ucsd.edu/pdf/PandP-1972_11_411-412-.pdf)
   Supports: "Yankee Doodle" is recognised in any single octave (whole-melody transposition keeps its identity), but not when its notes are scattered across octaves, so the contour and intervals carry the identity.

### (2) QBH systems use key/transposition-invariant pitch contours
5. Ghias, A., Logan, J., Chamberlin, D., & Smith, B. C. (1995). Query by humming: Musical information retrieval in an audio database. *Proc. ACM Multimedia '95*, 231–236. https://doi.org/10.1145/217279.215273
   Supports: the first QBH system represents melodies as relative pitch changes (U/D/S contour strings), which are key-invariant by construction.
6. Hu, N., & Dannenberg, R. B. (2002). A comparison of melodic database retrieval techniques using sung queries. *Proc. JCDL 2002*, 301–307. https://doi.org/10.1145/544220.544292
   Supports: sung queries can be transposed by any interval. Systems make them invariant with relative pitch or by searching over 12 (or 24 quarter-tone) transpositions, ignoring octaves (verified in the PDF, §3.5).
7. Dannenberg, R. B., Birmingham, W. P., Pardo, B., Hu, N., Meek, C., & Tzanetakis, G. (2007). A comparative evaluation of search techniques for query-by-humming using the MUSART testbed. *JASIST*, 58(5), 687–701. https://doi.org/10.1002/asi.20532
   Supports: "relative pitch is transposition invariant"; the matchers use relative pitch or search 24 transpositions (verified in the PDF).
8. Jang, J.-S. R., & Lee, H.-R. (2008). A general framework of progressive filtering and its application to query by singing/humming. *IEEE TASLP*, 16(2), 350–358. https://doi.org/10.1109/TASL.2007.913035
   Supports: a pitch-vector QBSH system (linear scaling + DTW) evaluated on the MIR-QBSH corpus. [PARTIALLY VERIFIED: citation and DOI confirmed; the key-transposition handling comes from secondary summaries, not the full text.] MIR-QBSH corpus: http://mirlab.org/dataset/public/ (dataset, not a paper) [link not checked].
9. Salamon, J., Serrà, J., & Gómez, E. (2013). Tonal representations for music retrieval: From version identification to query-by-humming. *IJMIR*, 2, 45–58. https://doi.org/10.1007/s13735-012-0026-0
   Supports: key-invariant melody (pitch-contour) representations for QBH, matched against real recordings. [Claim based on abstract and summary.]
10. Amatov, A., Lamanov, D., Titov, M., Vovk, I., Makarov, I., & Kudinov, M. (2023). A semi-supervised deep learning approach to dataset collection for query-by-humming task (CHAD). arXiv:2312.01092. https://arxiv.org/abs/2312.01092
   Supports: a modern QbH system on CREPE f0 activations trimmed to 3 octaves around the mean pitch (pitch-centred), with ±4 st pitch-shift augmentation, matched by DTW/correlation (verified in the full text §3, §5.1).

### (3) Hums/whistles are transposed relative to the original; whistle register is far above the voice
11. Levitin, D. J. (1994). Absolute memory for musical pitch: Evidence from the production of learned melodies. *Perception & Psychophysics*, 56(4), 414–423. https://doi.org/10.3758/BF03206733
   Supports: when non-musicians sing well-known songs from memory, most of them don't reproduce the original key (only 12 % hit it on both trials), so queries are usually transposed.
12. Frieler, K., Fischinger, T., Schlemmer, K., Lothwesen, K., Jakubowski, K., & Müllensiefen, D. (2013). Absolute memory for pitch: A comparative replication of Levitin's 1994 study in six European labs. *Musicae Scientiae*, 17(3), 334–349. https://doi.org/10.1177/1029864913493802
   Supports: the replication (N = 277) finds even less original-key reproduction (25 % one song, 4 % both), so transposition is the norm.
13. Nilsson, M., Bartůněk, J. S., Nordberg, J., & Claesson, I. (2008). Human whistle detection and frequency estimation. *Proc. CISP 2008*, vol. 5, 737–741. https://doi.org/10.1109/CISP.2008.415 (full text: https://www.diva-portal.org/smash/get/diva2:836227/FULLTEXT01.pdf)
   Supports: human whistling is a near-pure tone, typically 500–5000 Hz (20 subjects), which is well above the speaking/humming F0. Whistles therefore sit roughly 2–3+ octaves above the voice, consistent with our +35 st median.
14. Mori, M., & Fukuda, S. (2020). Frequency response of the vocal tract considering the glottis opening area during human whistling. *Electronics and Communications in Japan*, 103(8). https://doi.org/10.1002/ecj.12239
   Supports: whistling register is about 500 Hz–5 kHz with no significant male/female difference (unlike the voice). Whistle pitch is set by vocal-tract resonance, not vocal folds.
15. Shen, H.-C., & Lee, C. (2007). Whistle for music: Using melody transcription and approximate string matching for content-based query over a MIDI database. *Multimedia Tools and Applications*, 35(3), 259–283. https://doi.org/10.1007/s11042-007-0128-5
   Supports: query-by-whistling is a real QBH setting that needs its own transcription front-end. [PARTIALLY VERIFIED: bibliographic data confirmed; I couldn't read the full text (ACM page timed out), so I can't cite its specific claims about register or transposition.]
   NOTE: I found NO paper that directly measures "whistled query vs original recording key offset". The "about one octave or more higher" claim rests on 13–14 (whistle range) plus 11–12 (queries don't keep the original key). Our own data gives about +35 st whistle-over-hum.

### (4) Google Hum to Search / Now Playing
16. Frank, C. (2020, Nov 12). The Machine Learning Behind Hum to Search. Google Research Blog. https://research.google/blog/the-machine-learning-behind-hum-to-search/
   Supports: in hummed queries "the pitch, key, tempo or rhythm may vary slightly or even significantly". The production system learns melody embeddings from spectrograms, with pitch/tempo augmentation and synthetic hummed/whistled training audio. (Blog post, not peer reviewed.)
17. Agüera y Arcas, B., Gfeller, B., Guo, R., Kilgour, K., Kumar, S., Lyon, J., Odell, J., Ritter, M., Roblek, D., Sharifi, M., & Velimirović, M. (2017). Now Playing: Continuous low-power music recognition. arXiv:1711.10958. https://arxiv.org/abs/1711.10958
   Supports: only context (on-device recognition of *recorded* music by fingerprinting, the predecessor of Hum to Search). It does NOT support contour/transposition claims.

### Tools / data
18. Wei, H., Cao, X., Dan, T., & Chen, Y. (2023). RMVPE: A robust model for vocal pitch estimation in polyphonic music. *Interspeech 2023*. arXiv:2306.15412. https://arxiv.org/abs/2306.15412 (DOI 10.21437/Interspeech.2023-528 per dblp/search, not opened).
19. MLEnd Hums and Whistles dataset, Queen Mary University of London. https://mlenddatasets.github.io/hums_whistles/ (no dedicated paper found).
