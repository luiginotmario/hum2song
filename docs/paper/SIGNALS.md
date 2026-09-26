# Signal physics and pitch conventions (defined once, referenced everywhere)

This section fixes the terms, symbols and measured numbers that the rest of the paper uses. Other sections refer to it as §S1–§S8 and do not redefine anything.

Tags used below:
- **[V]** means the source was read and checked. It comes with a DOI or arXiv ID.
- **[PV]** means partly verified. The bibliographic data was checked, but the specific claim was taken from the abstract, a secondary source, or our earlier notes.
- **[ours]** means we measured it on MLEnd. The script and output file are named, and every number is copied from that output.

Paths like `figures/...` refer to the paper-figure workspace (`contour_figure.py`, `validate_whistle_f0.py` and their JSON/CSV outputs). It is not part of this repository yet.

---

## S1. Frequency, fundamental frequency, harmonics, pitch

- **Frequency** $f$ (Hz): the rate of a sinusoidal component.
- **Fundamental frequency** $F_0$: for a periodic signal with period $T$, $F_0 = 1/T$. A periodic signal can contain energy only at integer multiples $kF_0$ ($k = 1, 2, \dots$). These are the **harmonics**, and the $k$-th harmonic is $h_k = kF_0$. $F_0$ is a physical property of the waveform and is defined even when the component at $F_0$ itself is weak or absent.
- **Pitch** is the perceived height of a sound. For the harmonic sounds in this paper we treat pitch as $F_0$ on a log scale (§S5). A listener hears pitch at $F_0$ even when the $F_0$ component is missing ("missing fundamental"). de Cheveigné (2005) reviews this. **[PV]** (the chapter's bibliographic data was checked; we did not re-read the full text).
- **F0 track** (or contour): $F_0$ estimated every 10 ms, with a voicing flag per frame. **Voiced frame**: a frame the estimator marks as containing a pitched sound. The voicing rules we use are in §S6 and §S8.
- **Spectral peak** $f_{\text{pk}}$: the frequency of the largest STFT magnitude in a search band, per frame. It is an estimate of the *strongest component*, which is not necessarily $F_0$ (§S3).

## S2. Two kinds of source: harmonic voice and near-sinusoidal whistle

**Voice (speech, singing, humming): a harmonic source.** The vocal folds chop the airflow from the lungs into quasi-periodic glottal pulses. A pulse train with period $T$ has a line spectrum at $kF_0$. The vocal tract then shapes the amplitudes of those lines (formants) but does not move their frequencies. This is the source-filter model of speech production (Fant, 1960/1971). **[PV]** (the book's bibliographic data was checked via its De Gruyter DOI; it is a standard textbook result and we did not re-read it.)

**Hum: a voiced sound with the lips closed.** The oral tract is sealed and sound leaves through the nose (a nasal murmur). Titze (2001) describes nasal-murmur energy as mainly low-frequency (about 200–300 Hz). **[PV]** (bibliographic data checked; the claim is from an excerpt of the article shown on the publisher's page, and we did not read the full text.)

We use the hum's harmonic structure, measured on our own data, rather than a citation:
- On 607 MLEnd *Potter* hums (752,679 voiced frames), the spectral peak in 50–6000 Hz is on the 1st harmonic ($F_0$ itself) in 79.4% of frames. It is on the 2nd harmonic in 12.7%, the 3rd in 3.3%, and on no harmonic in 3.1%.
- Per clip, the peak equals $F_0$ in a median 88.4% of frames (IQR 68.6–96.0%).

**[ours]** (`validate_whistle_f0.py humpeak`, `figures/hum_spectral_peak_check.json`; $F_0$ from RMVPE at normal speed, salience ≥ 0.3.) So hums are dominated by low harmonics, but not so completely that the spectral peak can stand in for $F_0$.

**Whistle: a near-sinusoidal tone from an airflow-driven resonance.** Air forced through the lip orifice excites a resonance of the mouth cavity. Rayleigh and Wilson et al. (1971) identified it as close to a Helmholtz resonance. Shigetomi & Mori (2016) show with vocal-tract models that an air-column resonance also contributes when the tract is long. They note that the sound source is turbulence at the mouth orifice. They also write that "because the amplitude and period of the human whistling sound fluctuate to a minor extent, the sound is close to a (sine wave) pure tone", and they measure whistle $F_0$ as the peak of the FFT power spectrum. **[V]** Shigetomi & Mori (full text read). **[PV]** Wilson et al. (1971) is cited here as summarized by Shigetomi & Mori; we did not read the original. Nilsson et al. (2008) describe human whistling as "typically single frequency dominated signals ... although harmonics might occur". **[V]**

On MLEnd, the 2nd-harmonic level of whistles is a median −45.0 dB relative to the peak (IQR −49.5 to −40.6 dB) over the 1,702 analyzed whistles. **[ours]** (`figures/whistle_f0_validation_clips.csv`, column `median_h2_db`, voiced frames of RMVPE-fix.)

## S3. Why the spectral peak equals F0 for whistles but not for voices

A whistle has one dominant component (median 2nd harmonic −45 dB, §S2), so the largest spectral line *is* the fundamental. That gives $f_{\text{pk}} = F_0$ up to the frequency resolution.

A voice has many harmonics, and the vocal-tract filter can make a higher harmonic stronger than $h_1$. In band-limited audio, $h_1$ can also be missing entirely. In both cases $f_{\text{pk}} = kF_0$ with $k > 1$, and the listener still hears $F_0$ (§S1).

Measured:
- On MLEnd whistles, the spectral peak and RMVPE-fix agree within 0.5 semitone on a median 99.7% of the frames both call voiced (IQR 99.1–99.9%, 1,702 clips).
- For hums the peak is off $F_0$ in 20.6% of voiced frames (§S2).

**[ours]** (`figures/whistle_f0_validation.json`, `peak_only.per_clip_within05_on_both_voiced`; `figures/hum_spectral_peak_check.json`.)

**Exception.** Some MLEnd "whistles" are harmonic-rich. In 0.29% of voiced frames the peak sits about 19 st (a factor of 3) above RMVPE-fix. In a clip we inspected by eye (0236, performer 170), components appear near 0.9, 1.9 and 2.8 kHz. There the peak is plausibly on the 3rd harmonic, and RMVPE, not the peak, is right (§S8).

## S4. Typical frequency ranges

| Source | Range | Status |
|---|---|---|
| Speaking $F_0$, adults and children | Vowel-averaged $F_0$ in Peterson & Barney (1952, Table II, 76 speakers): men 124–141 Hz, women 210–235 Hz, children 251–276 Hz (range across the 10 vowels) | **[V]** (table read in the full text) |
| Singing $F_0$ | Conventional classical voice-type ranges run from about E2 (82 Hz, bass) to C6 (1,047 Hz, soprano) | **[PV]** secondary summary of voice-range literature, not a primary source. The RMVPE authors state that their output range, 32.7–1,975.5 Hz, "fully covers the range of most melodic instruments including the singing voice" (Wei et al., 2023, §2.2) **[V]** |
| Hum $F_0$ | Produced by the same vocal folds as speech and song, so it lies in the voice range above | inference, not a citation |
| Whistle $F_0$ | "typically located in the range of 500–5000 Hz" (20 subjects asked to whistle as high and as low as possible; "some people might exceed these limits, such as trained whistlers") (Nilsson et al., 2008) | **[V]** full text read |

**MLEnd, measured [ours]:**
- **Hum:** median $F_0$ 178.7 Hz. This is the median over 607 *Potter* hums of each clip's median voiced $F_0$ (RMVPE at normal speed). `figures/hum_spectral_peak_check.json`.
- **Whistle:** median 1,338 Hz (RMVPE-fix) or 1,359 Hz (spectral peak), taken as the median of per-clip medians over the 1,702 analyzed whistles of all 8 songs. The 5th–95th percentile of per-clip median peak frequency is 1,024–1,959 Hz. The 95 excluded clips (§S8) are mostly higher, with a median of 3.74 kHz. `figures/whistle_f0_validation.json`.
- **Whistle minus hum, same performer, *Potter*:** median offset +35.2 st (IQR +31.2 to +37.3 st, 424 pairs), i.e. about 2.5–3 octaves. The offset is each clip's median semitone value, whistle minus hum. `figures/fig_contours_potter_stats.json`. As a consistency check, $12\log_2(1338/178.7) = 34.9$ st.

## S5. Semitones, cents, key normalization, transposition invariance

**Log-frequency units.** For a frequency $f$ and a reference $f_{\text{ref}}$:
$$\text{st}(f) = 12\log_2\frac{f}{f_{\text{ref}}}, \qquad \text{cents}(f) = 1200\log_2\frac{f}{f_{\text{ref}}} = 100\,\text{st}(f).$$
An octave (a factor of 2) is 12 st, or 1200 cents. Our code uses MIDI numbers, $m(f) = 69 + 12\log_2(f/440\ \text{Hz})$, which is st with $f_{\text{ref}} = 440$ Hz, offset so that A4 = 69. RMVPE uses cents with $f_{\text{ref}} = 10$ Hz (Wei et al., 2023, Eq. 1). **[V]**

**Key normalization.** For a clip with voiced semitone values $s(t)$, the normalized contour is
$$\tilde s(t) = s(t) - \operatorname{median}_{t' \in \text{voiced}} s(t').$$

**Transposition invariance (exact).** Suppose a rendition is the same melody transposed by a factor $\alpha$, so that $f'(t) = \alpha f(t)$. Then $s'(t) = s(t) + 12\log_2\alpha$ is a constant shift. The median shifts by the same constant, so $\tilde s'(t) = \tilde s(t)$ exactly. This removes the key difference between a hum and a whistle (about +35 st, §S4), and between any query and the original recording.

**Two consequences we rely on:**
1. An octave error that holds for the *entire* clip is also a constant shift, so key normalization removes it.
2. *Intermittent* octave errors, where the estimate jumps between $f$ and $f/2$, are not constant and are not removed. They are what damage contour matching (§S6, §S8).

Our contour cleaning (in `contour_figure.py`) does three things:
- It drops voiced runs shorter than 50 ms.
- It folds isolated jumps of more than 9 st from a 0.5 s running median back by a multiple of 12 st.
- It applies a 5-frame median filter.

## S6. The octave error, why RMVPE makes it on whistles, and the half-speed fix

**Definition.** Let $d = 12\log_2(\hat f / f)$ be the error of an estimate $\hat f$ against a reference $f$, in st. We count a frame as an **octave error** when $|d - 12n| < 1$ st for some integer $n \neq 0$. $n = -1$ is "one octave too low", and so on. This is stricter than the broader "subharmonic error" of de Cheveigné & Kawahara (2002), who note that such errors are "sometimes called 'octave error' (improperly because not necessarily in a power of 2 ratio with the correct value)". **[V]** Errors near $d \approx -19.02$ st (a factor of 3) are therefore *not* counted as octave errors here and are reported separately.

**RMVPE's output range.**
- *Paper:* RMVPE outputs a 360-bin salience vector with 20-cent bins from C1 (32.7 Hz) to B6 (1,975.5 Hz) and decodes $F_0$ as a local weighted average around the argmax bin (Wei et al., 2023, §2.2, Eqs. 2–3). **[V]**
- *Code:* in the RVC implementation we run (`rmvpe_rvc.py`, `cents_mapping = 20*arange(360) + 1997.38`), the bin centres span 31.7–2,005.5 Hz. The decoded $F_0$ can never exceed the top bin centre.
- The paper trains and evaluates on MIR-1K, MDB-stem-synth, MIR_ST500 and Cmedia, which are singing voice and melodic resynthesis (§3.1). **[V]** We could not find documentation of the exact training data of the RVC `rmvpe.pt` checkpoint we use. **[unverified]**

**What we observe.** On whistles, RMVPE at normal speed mostly reports a lower octave. Per clip, a median 55.4% of RMVPE-fix-voiced frames are octave errors (IQR 37.6–66.4%). Pooled over all voiced frames it is 52.9%, almost all one octave down. There are also non-octave errors near −19 st and −24 st. **[ours]** On synthetic whistle-like tones with known $F_0$, normal-speed RMVPE is correct at 500–700 Hz, 82–87% correct at 1 kHz, at most 6% at 1.4 kHz and 0% at ≥ 2 kHz (§S8). The errors start well inside its nominal 2 kHz range, so the output ceiling alone does not explain them.

**Why (hypothesis, not established).** Sung $F_0$ is rarely above about 1 kHz (§S4), so the model has probably seen few near-pure tones in the whistle register. A weak sinusoid at 1–2 kHz may also look to it like an upper harmonic of a lower voice. We have not tested this mechanism.

**The half-speed fix.** We play the whistle to RMVPE at half speed and double the answer.
- Let $x(t) = \sum_k a_k \cos(2\pi f_k t + \phi_k)$. Playing it at half speed gives $y(t) = x(t/2) = \sum_k a_k \cos\!\big(2\pi (f_k/2)\, t + \phi_k\big)$.
- Every component moves from $f_k$ to $f_k/2$, i.e. by $1200\log_2\frac{f_k/2}{f_k} = -1200$ cents, **exactly and independently of $f_k$**. Relative pitch (the contour) is unchanged.
- Multiplying the estimate by 2 adds +1200 cents back. If RMVPE estimates $F_0/2$ correctly on $y$, the corrected value is exactly $F_0$.
- The transform itself is exact. Whether the correction is right depends only on whether RMVPE is right on the slowed signal.

**Implementation.**
- The clip is resampled to 32 kHz, which preserves frequencies in Hz. The samples are then given to RMVPE, which assumes 16 kHz. Every frequency therefore appears at $f \cdot 16000/32000 = f/2$.
- RMVPE's 160-sample hop becomes 5 ms of real time, so every 2nd frame is kept to return to a 10 ms grid. Its 1,024-sample analysis window covers 32 ms of real time instead of 64 ms.
- The output ceiling becomes $2 \times 2005.5 = 4011$ Hz.

**Where the fix stops working.** It moves the whistle down by one octave only, so whistles above about 2 kHz land above about 1 kHz after slowing, which is again the region where RMVPE fails. On synthetic tones, RMVPE-fix is 100% correct up to 1.4 kHz, 74–85% at 2 kHz and at most 10% at ≥ 2.8 kHz. On MLEnd, clips whose median spectral peak is in 2–2.5 kHz have a median agreement of only 20.7% (§S8).

## S7. Contour comparison: DTW with a slope constraint

We compare two key-normalized contours (§S5) with dynamic time warping (DTW).

**Setup** (as implemented in `contour_figure.py`):
- **Sequences:** voiced frames only, averaged to 20 ms frames.
- **Local cost:** $|\tilde s_{\text{ref}}(i) - \tilde s_{\text{qry}}(j)|$ in st.
- **Steps:** $(1,1)$, $(1,2)$, $(2,1)$ with multiplicative weights 1, 1.5, 1.5, inside a Sakoe–Chiba band of radius 25% of the sequence length.
- **Scoring:** the query is mapped onto the reference's frame grid (mean of matched frames), then we report MAE (st) and Pearson $r$.

The step set limits the local tempo ratio to between 1/2 and 2. This is the idea of the slope constraint of Sakoe & Chiba (1978), who restrict the warping-path slope "so as to improve discrimination between words in different categories". **[V]** (abstract). Their P = 1 condition allows the same 1/2–2 slope range. **[PV]** (from the dtw-python documentation of their Table I, not re-derived.) Our step weights are our own.

**Why unconstrained DTW was dropped.** With standard steps $(1,0)$, $(0,1)$, $(1,1)$ and no slope limit, DTW could also fit contours of *different* songs well (median $r \approx 0.88$–$0.91$). A high $r$ therefore did not show that two renditions shared a melody. This is from an earlier internal run, recorded in `figures/REFERENCES_and_CAPTION.md`. Its per-pair output is not in the stats JSON, and we did not re-run it here.

**Caveat (measured).** When the two voiced-sequence lengths differ by more than a factor of 2, no path with slope in [1/2, 2] exists. The code then falls back to standard steps, which have no slope limit, inside the band. In the *Potter* hum-vs-whistle analysis (RMVPE-fix whistles), this happens for 189 of 1,230 pairs: 424 same-song plus 806 other-song. **[ours]** (`figures/whistle_f0_downstream.json`.) The published figure's statistics include those pairs. A whistle F0 method that voices fewer frames makes sequences shorter and triggers more fallbacks.

## S8. Validation of the whistle F0 (all MLEnd whistles)

All numbers in this section are **[ours]**, from `validate_whistle_f0.py` (outputs: `figures/whistle_f0_validation.json`, `figures/whistle_f0_validation_clips.csv`, `figures/fig_whistle_f0_validation.pdf`, `figures/whistle_f0_downstream*.json`, `figures/whistle_f0_synth.json`). MLEnd has no ground-truth F0, so we measure agreement between two independent estimators: RMVPE (a neural network) and the plain STFT spectral peak. A known-F0 synthetic check backs this up.

**Estimators and voicing rules:**
- **RMVPE-fix:** half speed, ×2 (§S6). Voiced when salience ≥ 0.3 and 50 < $\hat f$ < 4,200 Hz.
- **Spectral peak:** 200–6,000 Hz, 64 ms Hann window zero-padded to 8,192 points at 32 kHz, parabolic interpolation. Voiced when the tonal ratio is ≥ 6 dB, where the tonal ratio is the energy within ±50 cents of the peak over the rest of the band.

**Coverage.** MLEnd lists 1,797 whistles and all were processed. 95 clips were excluded because RMVPE-fix voiced fewer than 100 frames (1 s), leaving 1,702 analyzed. The excluded clips have a median spectral peak of 3.74 kHz (73% above 2 kHz), and 39 of them have under 5% peak-voiced frames. **The exclusion removes mostly clips where RMVPE-fix fails, so the agreement numbers below are optimistic for RMVPE-fix.**

**Agreement, RMVPE-fix vs spectral peak** (voiced frames of RMVPE-fix; per-clip statistics):

| | median | IQR | clips > 90% |
|---|---|---|---|
| frames within 0.5 st, with half-speed fix | 98.2% | 96.7–99.1% | 91.1% |
| frames within 1 st, with fix | 99.5% | 98.4–99.8% | 92.1% |
| frames within 0.5 st, normal speed (same frames) | 6.8% | 1.7–16.7% | 0.3% |
| octave-error rate, with fix | 0.0% | 0.0–0.34% (mean 3.4%) | |
| octave-error rate, normal speed (same frames) | 55.4% | 37.6–66.4% | |

- **Pooled over 1,412,000 voiced frames:** 95.9% within 0.5 st with the fix vs 14.2% without. Octave errors: 2.1% with the fix vs 52.9% without.
- **Voicing:** at normal speed RMVPE voices a median 16.6% of frames, compared with 52.8% with the fix.
- **Spot check vs full dataset:** the earlier 12-clip spot check (88–100% within 0.5 st) is consistent with the dataset-wide median but hides the high-pitch failures below.

**Failures** (clips under 90% agreement): 152 of 1,702 (8.9%).
- **Pitch is the main cause.** By median spectral-peak frequency, the share of clips above 90% agreement is:

  | median spectral peak | clips | share > 90% agreement | median agreement |
  |---|---|---|---|
  | < 1 kHz | 76 | 96.1% | |
  | 1–1.5 kHz | 1,162 | 98.7% | |
  | 1.5–2 kHz | 382 | 85.3% | |
  | 2–2.5 kHz | 70 | 5.7% | 20.7% |
  | > 2.5 kHz | 12 | 0% | |

  51% of failing clips have a median peak above 2 kHz.
- **These are a second octave error by RMVPE-fix, not a peak mistake.** In the octave-error frames of failing clips, the spectrum at RMVPE-fix's reported frequency ($f_{\text{pk}}/2$) is −55.2 dB relative to the peak, no different from passing clips (−55.4 dB). There is no component where RMVPE places $F_0$.
- **Noise explains little.** Heuristic labels (overlapping): residual octave error 123, very high pitch (> 2 kHz) 78, low SNR (< 15 dB) 2, breathy/noisy (tonal ratio < 0 dB) 2, unexplained 26. Median SNR is 38.9 dB in passing clips and 34.2 dB in failing ones. The Spearman correlation of agreement with SNR is only 0.11, against 0.44 with RMVPE salience and 0.39 with voiced fraction.
- **Harmonic-rich clips** (peak ≈ 3× RMVPE, 0.29% of frames) are the one pattern where the spectral peak, not RMVPE, is the likely culprit (§S3).
- **Worst clips:** 2258, 3828, 5435, 5812, 0236, 2899, 6234, 3482, 5096, 3134, 3568, 1565 (median peak 2.3–4.1 kHz). Four performers (220, 111, 170, 134) account for 54 of the 152 failures. The full list is `failures.worst_25` in the JSON.

**Synthetic tones with known $F_0$** (±2 st glide, 2nd harmonic at −26 dB, pink-ish noise at 30/10/0 dB SNR):

| | < 1 kHz | 1 kHz | 1.4 kHz | 2 kHz | ≥ 2.8 kHz |
|---|---|---|---|---|---|
| RMVPE, normal speed | 100% (500–700 Hz) | 82–87% | ≤ 6% | 0% | 0% |
| RMVPE-fix | 100% | 100% | 100% | 74–85% | ≤ 10% |
| spectral peak, when voiced | 100% | 100% | 100% | 100% | 100% (up to 4 kHz) |

With the 6 dB tonal gate the spectral peak voices nothing at 0 dB broadband SNR, where RMVPE-fix still voices tones up to 1.4 kHz. The synthetic tones are easier than real whistles (one clean component), so they bound behaviour but do not stand in for it.

**Spectral peak alone as the whistle F0.** Against RMVPE-fix:
- Voicing: per-clip median precision 0.82, recall 0.93.
- Pitch: median 99.7% of jointly voiced frames within 0.5 st.

We then ran the contour-retrieval test behind the paper's figure: each performer's hum against their own whistle of the same song vs their whistles of two other songs, scored by AUC of MAE. On the pairs every method can score, with a 95% CI from a performer-level bootstrap:

| target song | spectral peak | RMVPE-fix | difference (95% CI) |
|---|---|---|---|
| *Potter* | 0.840 | 0.766 | +0.073 (+0.029 to +0.122) |
| *StarWars* | 0.773 | 0.711 | +0.063 (+0.024 to +0.102) |
| *Hakuna* | 0.827 | 0.798 | +0.029 (+0.005 to +0.054) |

- **Controls:** swapping in only the peak's $F_0$ values (RMVPE's voicing) changes AUC by +0.009, +0.009 and −0.001. Swapping in only the peak's voicing (RMVPE's $F_0$) changes it by +0.013, −0.003 and −0.001. Neither is significant. The gain comes from the peak's $F_0$ and its own voicing used together.
- **Threshold sensitivity:** a too-strict peak gate (prominence ≥ 60 dB) is *worse* than RMVPE-fix on *Potter* (−0.069), partly through more DTW fallbacks (§S7).
- **No fix:** RMVPE at normal speed is at chance (AUC 0.49, 0.52, 0.57).
- **Choice of rule:** the 6 dB gate was one of four pre-listed rules. It was not tuned on the retrieval test, but it was chosen after seeing its voicing agreement with RMVPE on the same whistles.

**Chosen method: see DECISIONS D-007.**

**Caveats that apply to every number in the paper's contour figure:**
- The figure's featured performer (213; whistle 0681 vs reference hum 0491: MAE 0.51 st, $r$ = 0.99) was picked for clean recordings *and* good hum/whistle agreement. It is a best case and illustrative only.
- The honest numbers are dataset-wide, over 111 *Potter* performers and 424 same-performer hum–whistle pairs:
  - median $r$ 0.80 same song vs 0.57 different song (806 pairs)
  - median MAE 1.63 vs 2.34 st
  - AUC 0.75
  - ceiling (two hums of the same performer, same song): median $r$ 0.95, MAE 0.75 st

  Sources: `figures/fig_contours_potter_stats.json`, `figures/REFERENCES_and_CAPTION.md`. Re-running the pipeline with RMVPE-fix whistles reproduces $r$ 0.7976 / 0.5656 and AUC 0.7496 exactly (`figures/whistle_f0_downstream.json`).
- Those figure numbers use RMVPE-fix whistles, so they inherit the high-pitch failures and exclusions described above and the DTW fallback in §S7.

---

## References (verification status as of 2026-09-26)

1. de Cheveigné, A. (2005). Pitch perception models. In *Pitch: Neural Coding and Perception* (Springer Handbook of Auditory Research), 169–233. doi:10.1007/0-387-28958-5_6. **[PV]** bibliographic data only.
2. de Cheveigné, A., & Kawahara, H. (2002). YIN, a fundamental frequency estimator for speech and music. *JASA* 111(4), 1917–1930. doi:10.1121/1.1458024. **[V]** full text, Step 4 ("octave error" wording) and the evaluation section (too-low/too-high gross errors).
3. Fant, G. (1960; reprint 1971). *Acoustic Theory of Speech Production*. De Gruyter Mouton. doi:10.1515/9783110873429. **[PV]** bibliographic data only.
4. Nilsson, M., Bartůněk, J. S., Nordberg, J., & Claesson, I. (2008). Human whistle detection and frequency estimation. *Proc. CISP 2008*, 737–741. doi:10.1109/CISP.2008.415. **[V]** full text (DiVA copy), §2 and §7.
5. Peterson, G. E., & Barney, H. L. (1952). Control methods used in a study of the vowels. *JASA* 24(2), 175–184. doi:10.1121/1.1906875. **[V]** full text, Table II.
6. Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition. *IEEE TASSP* 26(1), 43–49. doi:10.1109/TASSP.1978.1163055. **[V]** abstract and "P = 1 is optimum" passage. **[PV]** the P = 1 slope range (1/2 to 2) is from secondary documentation.
7. Shigetomi, T., & Mori, M. (2016). Principles of sound resonance in human whistling using physical models of human vocal tract. *Acoust. Sci. & Tech.* 37(2), 83–86. doi:10.1250/ast.37.83. **[V]** full text.
8. Titze, I. R. (2001). Acoustic interpretation of resonant voice. *J. Voice* 15(4), 519–528. doi:10.1016/S0892-1997(01)00052-2. **[PV]** abstract page only.
9. Wei, H., Cao, X., Dan, T., & Chen, Y. (2023). RMVPE: A robust model for vocal pitch estimation in polyphonic music. *Interspeech 2023*, 5421–5425. arXiv:2306.15412, doi:10.21437/Interspeech.2023-528. **[V]** full text (arXiv), §2.2 and §3.1.
10. Wilson, T. A., Beavers, G. S., DeCoster, M. A., Holger, D. K., & Regenfuss, M. D. (1971). Experiments on the fluid mechanics of whistling. *JASA* 50(1B), 366–372. doi:10.1121/1.1912641. **[PV]** cited via Shigetomi & Mori (2016); original not read.

Singing-range figures (E2–C6) come from a secondary summary of the voice-range literature and are **[PV]**. For a primary source, use a voice-range-profile study read in full (candidate: Lamarche, Ternström & Pabon, 2010, *J. Voice* 24(4), 410–426, doi:10.1016/j.jvoice.2008.12.008, whose bibliographic data was checked but full text not read).
