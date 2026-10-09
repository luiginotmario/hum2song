# Finding a song from a whistle or from singing with the wrong words: query-by-melody with physics-based synthetic queries and a key-invariant contour encoder

**Draft, 2026-10-09.** Luigi Rivolta (hum2song). Every number below points to a file in this repository. Synthetic results are marked **[synthetic]**, results on real recordings of people are marked **[real]**, and hand-picked examples are marked **[best case]**. D-041 (round 2) numbers were added on 2026-10-09 from `docs/paper/results/d041/` (printed by `summarize.py` there).

---

## Abstract

People look up songs in three ways: humming, whistling, and singing a half-remembered tune with the wrong words or with nonsense syllables. Published query-by-humming (QbH) work and products focus on humming. Google's Hum to Search reports that a model trained mostly on sung clips "worked well enough for people singing, but not for people humming or whistling" [1], and the open QbH literature (CHAD [3], ByteHum [2]) evaluates hummed and sung queries, not whistles. We study whistled and wrong-lyrics queries, and keep humming as a baseline.

Our system matches the melody only. A query is converted to a pitch track (F0), key-normalized by subtracting its median pitch, and embedded by a 5.2 M-parameter convolution + transformer contour encoder (the E2b model, fine-tuned from a HumTrans-trained encoder on cover → original pairs). Whistles are tracked with a spectral-peak tracker because the standard vocal pitch model, RMVPE, often reports whistles an octave low (a median 55.4% of voiced frames per clip) (SIGNALS §S6, §S8). Songs are indexed by the vocal melody of their recordings (vocal stem → RMVPE).

The new contribution is a synthetic query pipeline (D-040). From the vocal melody of 11,083 open songs we render audio of (a) a whistle, a near-sinusoid moved into 500–4,000 Hz with a weak second harmonic, breath noise and a room, and (b) a source-filter voice singing random syllables from a fixed list (la, na, ma, du, lu, ni, me, lo), so the words can never be the song's words. Human errors (a wrong note, a dropped note, drift, tempo warp, octave slips) are applied first, and each rendering is re-tracked with the tracker a real query would get. The tracked contours are used to fine-tune E2b against vocal-melody windows of the same song with the CLEWS loss [7].

On held-out songs, with 8,526 synthetic queries against 11,050 songs, the fine-tuned model raises top-1 from 0.180 to 0.226 on synthetic whistles and from 0.193 to 0.238 on synthetic wrong-lyrics singing **[synthetic]**. Hum retrieval is almost unchanged (HumTrans test top-1 0.996 → 0.991, MIR-QBSH 48-song top-1 0.955 → 0.944) **[real]**. On real whistles we have only one test, MLEnd whistle → other people's hums, 8 songs. There the result is mixed: top-1 0.631 → 0.611 over all 8 songs, and 0.589 → 0.656 on two songs (90 queries), which is within sampling noise **[real]**. We could not yet score real whistles or real wrong-lyrics singing against the actual song recordings, because the target recordings are not available to us (§9). The training run also exposed a checkpoint-selection bug: selection used a hum metric that was already saturated, so the "best" checkpoint was the starting model (§8.3). A second round (D-041) fixed selection (synthetic val with a hum guard), trained 4× longer with per-epoch query augmentation, and added real MLEnd whistle → hum pairs from training performers of 6 of the 8 songs. It raises synthetic top-1 further, to 0.253 (whistle) and 0.269 (wrong lyrics) **[synthetic]**, with hums nearly flat (HumTrans 0.993, MIR-QBSH 0.949) **[real]**. But on the two MLEnd songs it never trained on, real whistle top-1 falls to 0.456 (E2b 0.589), while it rises on the 6 trained songs: the real-data gain is song familiarity, not whistle robustness (§7.3). We therefore make no claim that the method beats E2b on real queries, and E2b stays the live model.

---

## 1. Introduction

**The use case.** When a tune is stuck in your head, you often don't know the words. You whistle it, or sing it with whatever words come to mind ("la la la", or the wrong lyrics). The matching problem is the same in every case: find the recording whose melody matches the query's melody, regardless of timbre, words, key, and tempo.

**Why whistles are hard.** A whistle is a different sound source from the voice (SIGNALS §S2). It is close to a pure tone: on 1,702 MLEnd whistles the median second harmonic is −45.0 dB below the fundamental. A voice has many harmonics. Whistles also sit far higher: the MLEnd median whistle is about 1.34–1.36 kHz, against a median hum of 178.7 Hz, a same-performer offset of a median +35.8 semitones (about 3 octaves; SIGNALS §S4). Two consequences matter for retrieval:
- A vocal pitch model can fail. RMVPE at normal speed reports a lower octave on a median 55.4% of a whistle's voiced frames, and whistle-to-hum retrieval with it is at chance (AUC 0.49–0.53; SIGNALS §S8). A spectral-peak tracker agrees with half-speed RMVPE on a median 99.7% of jointly voiced frames and gives the best retrieval (D-007).
- People whistle a compressed version of the tune. On MLEnd, the 5–95% pitch range of a whistle is 6.5 semitones, against 9.7 for a hum of the same songs (D-013).

**Why wrong lyrics matter.** A spectrogram model can learn to use lyrics and timbre, which a wrong-lyrics query does not share with the recording. A melody-only model cannot use them, by design. Our encoder sees only the key-normalized F0 track (SIGNALS §S5), so lyrics are irrelevant to it at inference time. What remains hard is that real singing with made-up words changes the voiced/unvoiced pattern and adds consonant breaks that a pitch tracker turns into gaps and glitches.

**What is missing.** There is no public dataset of whistled or wrong-lyrics queries paired with the song recordings. MLEnd Hums and Whistles [10] has whistles of 8 songs but ships no song recordings. MTG-QBH [11] has 118 sung queries but not the collection audio. CHAD [3] has hums, with originals as YouTube IDs. We found no public wrong-lyrics QbH set (D-040).

**Contributions.**
1. A physics-based synthetic query generator for whistles and nonsense-syllable singing, locked frame-by-frame to the vocal melody of real songs and re-tracked with the tracker a real query would get (§5.2, D-040). Its physics is the one measured and defined in `SIGNALS.md`.
2. A melody-only retrieval system built on a key-invariant contour encoder, with a separate whistle tracker validated on all 1,702 usable MLEnd whistles (SIGNALS §S8, D-007), a windowed song index, and a re-ranking pipeline (D-025, D-027).
3. A measured comparison with a music foundation model (MERT [8]) under the same hum protocol, which motivates the contour encoder (§7.4).
4. An honest account of what worked and what did not: D-039 (hum fine-tune: no gain), D-040 (synthetic gain, real whistle result mixed), the checkpoint-selection failure and its fix (§8), and D-041 (larger synthetic gain, worse on real held-out whistles).

---

## 2. Related work

Each entry says what we checked. "Abstract only" means we did not read the full paper.

**Google Hum to Search** [1] (blog post, not peer reviewed, no public accuracy numbers). It embeds a spectrogram of the query and of recording segments, trained with metric learning on (sung or hummed, recorded) pairs. The first training set was mostly sung, with pitch and tempo augmentation; that model worked for singing but "not for people humming or whistling". They then generated synthetic hums from SPICE [12] pitch, first as discrete tones and later with "a neural network that generates audio resembling an actual hummed or whistled tune", and added a batch-level confidence loss to a triplet loss. The production database had "over half a million songs" at the time. *Read in full.* Our whistle synthesis follows their first step (tones rendered from extracted pitch) and adds the measured whistle physics and re-tracking. Their neural generator is unpublished.

**ByteHum** [2] (Du et al., ICASSP 2024, pp. 1111–1115). A CNN on raw audio, trained with weak supervision from a source-separated cover-song identification dataset, with unsupervised domain adaptation toward hums; the authors also annotate original recordings for three existing QbH benchmarks. *Abstract only*; the full paper is not open to us, so we report none of its numbers.

**CHAD** [3] (Amatov et al., arXiv:2312.01092). A semi-supervised pipeline that collects time-aligned cover fragments, plus crowdsourced hums (5,164 humming fragments, 31,630 cover songs). The model is ResNet18 on CREPE F0 activations or CQT, trained with NT-Xent (τ = 0.05). With CREPE features it reports top-10 0.921 on MIR-QBSH (Jang) against about 2,600 MIDI references, 0.868 on "Jang Real" and 0.883 on MTG-QBH with 1,886 imposter songs (Table 1). The authors note that "the pure search-by-humming setup requires adding a portion of real humming data". *Read in full.* We use its hums, cover pairs and song split (D-019, D-028).

**Recent contour QbH.** Kung, Chen and Ding (2026) [4] propose a Conformer-based QbH network with a harmonic-aware mechanism. *Bibliographic record only; venue and numbers not verified.* D-011 lists it as an example of convolution + self-attention over the melody, which is the same family as our encoder.

**Datasets.**
- **HumTrans** [5] (Liu et al., arXiv:2309.09623): 500 compositions, 1,000 segments, about 56.22 hours of humming with MIDI labels and official splits. Our hum training data (13,080 training hums; D-011).
- **MIR-QBSH** (Jang [6], [19]): 4,431 hummed/sung queries for 48 MIDI songs. Fully held out in our work (D-005). It is the standard MIREX QBSH set used by CHAD.
- **MLEnd Hums and Whistles** [10] (QMUL): 4,804 hums and 1,797 whistles of 8 songs from about 200 people (D-013). No song recordings.
- **MTG-QBH** [11] (Salamon, Serrà and Gómez, 2013): 118 sung queries of 81 songs, with or without lyrics; collection audio is not distributed.

**Representations and losses.**
- **CLEWS** [7] (Serrà et al., ICML 2025): contrastive learning from weakly labeled audio segments for musical version matching. We use its segment-level loss (best of several reference windows as the positive, closest window of each other song as the negative; D-026).
- **MERT** [8] (Li et al., ICLR 2024): a self-supervised music audio model (95M and 330M parameters). We fine-tuned MERT-v1-95M as our first query encoder (D-010, D-011) and report it as a baseline.
- **Pitch trackers.** RMVPE [9] (Wei et al., Interspeech 2023) estimates vocal pitch directly from mixtures, with an output range of 32.7–1,975.5 Hz in the paper (SIGNALS §S6). SPICE [12] (Gfeller et al., 2020) is the self-supervised pitch model used in Hum to Search's synthetic data. CREPE [13] is used by CHAD.

**Whistle transcription and query-by-whistling.** Shen and Lee [14] built "Whistle for Music", a query-by-whistling system with melody transcription and approximate string matching over MIDI (*bibliographic data only*). Dias, Ventura and Gaspar [15] compare four pitch detectors (time and frequency autocorrelation, AMDF, harmonic product spectrum) for transcribing whistles and apply them to melody retrieval (*abstract and conclusions read*). Nilsson et al. [16] describe whistling as "typically single frequency dominated" with F0 typically in 500–5000 Hz (*read in full*; SIGNALS §S2, §S4). These systems transcribe whistles into notes and match strings or DTW. None of them learns a joint embedding with real song recordings.

**Singing voice synthesis (future work).** SoulX-Singer [17] (arXiv:2602.07803) is an open singing synthesizer trained on more than 42,000 hours, with a melody-control mode that takes "the target lyrics together with a continuous F0 contour". Vevo2 [18] (arXiv:2508.16332) unifies speech and singing generation and lists humming-to-singing among its tasks. Both could replace our source-filter singer with realistic voices; neither was used here (§10).

---

## 3. Signals: definitions used throughout

All signal physics, symbols and measured numbers are defined once in [`SIGNALS.md`](SIGNALS.md) and referenced by section number. We do not restate them. The parts this paper uses:

| Topic | Where | Used for |
|---|---|---|
| F0, harmonics, pitch, F0 track, spectral peak | §S1 | all inputs |
| Voice = harmonic source through a vocal-tract filter; hum = nasal voiced sound; whistle = near-sinusoid (MLEnd median 2nd harmonic −45.0 dB) | §S2 | synthetic singer (§5.2.2), synthetic whistle (§5.2.1) |
| Why the spectral peak equals F0 for whistles but not for voices | §S3 | whistle tracker |
| Ranges: whistle 500–5,000 Hz (Nilsson et al.); MLEnd hum median 178.7 Hz, whistle median about 1.34–1.36 kHz, whistle − hum +35.8 st | §S4 | whistle register in the generator |
| Semitones, key normalization, exact transposition invariance; whole-clip octave errors removed, intermittent ones not | §S5 | encoder input |
| RMVPE's octave error on whistles and the half-speed fix | §S6 | tracker choice |
| Slope-constrained DTW | §S7 | contour figure, re-ranking |
| Whistle F0 validation on all MLEnd whistles | §S8 | tracker choice, limits above 2 kHz |

---

## 4. Method

### 4.1 Melody front ends

- **Hums and singing:** RMVPE [9] at 10 ms, voiced where salience ≥ 0.3, then contour cleaning (short runs dropped, isolated octave jumps folded, median filter; D-011, SIGNALS §S5).
- **Whistles:** the spectral peak in 200–6,000 Hz with a 6 dB tonal-ratio voicing gate, the same cleaning (D-007; validation in SIGNALS §S8). RMVPE at half speed ("RMVPE-fix") is kept as a cross-check.
- **Songs:** htdemucs vocal stem → RMVPE, with RMVPE on the mixture as a fallback when the stem has no voiced window (D-014, D-017). On MIR-1K this costs 0.03–0.06 top-10 against annotated melody (D-014).

### 4.2 Contour encoder

Input: semitones minus the crop's median voiced pitch, plus a voicing flag, at 20 ms (key invariance is exact; SIGNALS §S5). A convolutional front end (halving the frame rate), a 6-layer pre-norm transformer (d = 256, 4 heads), a masked attention pool and an MLP give one L2-normalized 256-d vector; about 5.2 M parameters, one tower for queries and songs (D-011). Training starts from the D-012 HumTrans model and the E2b cover fine-tune (D-028): symmetric InfoNCE on HumTrans hum ↔ MIDI pairs plus the CLEWS loss on cover → original windows.

### 4.3 Search

Songs are cut into 10 s chunks and 5 s windows (1 s hop) and stored in pgvector. A query's 5 s windows vote over all song windows; the union of the top 200 songs by chunk score and by window vote is re-ranked by a weighted sum of chunk score, window vote, an order-consistent window path score and slope-constrained DTW, with weights chosen on CHAD val songs only (D-023, D-025, D-027). The D-040 experiments below use a simpler in-memory protocol (§6.3).

---

## 5. Synthetic whistle and wrong-lyrics queries (D-040)

### 5.1 Design principle

D-031 tried whistle robustness by editing contours directly (interval compression, octave folding, gaps over 7,545 melodies). It gave +0.033 top-1 on two held-out MLEnd songs, nothing on all 8, and hurt hums (MTG-QBH −4.2 top-1). Its revisit note was that contour edits never pass through a tracker. D-040 therefore renders **audio** and re-tracks it, so the training contours carry the errors a real tracker makes on that kind of sound. Every rendering is locked frame-by-frame to a real song's vocal F0, so the positive pair (query, song window) is known exactly.

### 5.2 Generator

Source melody: the cached vocal-stem RMVPE track of each indexed open song (`youtube_charts_v1` plus every FMA library; evaluation-target titles dropped), 11,083 songs. Four windows per song, each rendered in both styles (`generate_query_synth.py --per-song 4 --sing-tracker rmvpe`). Code: `training/src/hum2song/query_synth/`.

**Human errors, both styles, before rendering** (`errors.py`): one note moved by ±1 or ±2 semitones (p = 0.5), one note dropped (p = 0.4), then the existing contour augmentation and humanization (glides, vibrato). Ranges as in D-039: tempo stretch 0.5–1.9, local warp 0.3, interval scale 0.25, drift 1 st, jitter 0.25 st, voicing dropout 0.6, octave slips 0.15.

#### 5.2.1 Whistle (`whistle_audio.py`)
- **Register.** Add 2 or 3 octaves, then shift by whole octaves until the median sits in 500–4,000 Hz. The lower edge is Nilsson et al.'s range (SIGNALS §S4). The upper edge is where the spectral peak stayed correct on synthetic tones (SIGNALS §S8). After key normalization the register itself does not reach the encoder (SIGNALS §S5); what changes is how the tracker behaves at that pitch.
- **Tone.** One sinusoid at F0 plus a second harmonic drawn uniformly in −50 to −40 dB, centered on the MLEnd median of −45.0 dB (SIGNALS §S2). 15 ms attack/release.
- **Breath.** High-passed noise 30–40 dB below the tone while voiced, 50 dB below in gaps.
- **Room and microphone.** One reflection (12–35 ms, gain 0.04–0.18) plus high-passed noise at 18–36 dB SNR (`signal.py`). This is a minimal room, not a measured impulse response.
- **Re-track:** spectral peak (D-007).

#### 5.2.2 Wrong-lyrics singing (`singing.py`)
- **Source.** 16 harmonics of F0 with a 1/k tilt: the harmonic source of SIGNALS §S2.
- **Filter.** Three resonances per vowel (a, e, i, o, u), with separate settings for nasals (m, n) and liquids (l), and a silent burst for "d". The resonance values are fixed constants of this synthesizer, close to textbook adult vowel formants; they are not a new measurement and we do not claim them as a checked table.
- **Words.** Syllables of 180–280 ms drawn at random from la, na, ma, du, lu, ni, me, lo. No lyric text is an input, so the words cannot be the song's words.
- **Room** as for the whistle. **Re-track:** RMVPE, the tracker real sung queries get.

**Smoke check [best case]** (`docs/paper/results/d040/smoke.json`, one invented tune, best lag within ±20 frames): steady whistle probe 2nd harmonic −46.21 dB at 1,318.5 Hz; rendered whistle median 1,558 Hz, re-tracked median absolute error 0.045 st; rendered singing median 390 Hz, 0.106 st. This checks the renderer on one clip; it is not a dataset error rate.

**Output.** 86,110 query contours were written (`docs/paper/results/d040/gen.log`). Songs are split by `crc32(song_id) mod 10`: 0 test, 1 val, else train. The training set has 68,714 synthetic pairs (`train.log`).

### 5.3 Training

From E2b (`ckpt/contour_e2b_s0/best.pt`, 85/85 tensors loaded). Loss = CLEWS (weight 1.0) on synthetic query ↔ 4 vocal-melody windows of the same song within ±3 s (64 songs per batch, γ = 5) + 0.15 × InfoNCE on HumTrans hums ↔ MIDI (batch 256). The 0.15 was fixed before the run, not tuned. LR 3e-5, 100 warmup steps, 3,000 steps, one seed, about 26 minutes on one A100 (`configs/train_contour_qsynth.yaml`, `docs/paper/results/d040/train.log`). Checkpoint selection used HumTrans val top-1 with a +7 semitone shift; §8.3 explains why that failed.

---

## 6. Data and evaluation protocols

### 6.1 Song library
Open full-length tracks: FMA (`fma_full`, several disjoint seeded draws) plus YouTube chart songs used only as features (D-017, D-020, D-033, D-038). The live index after D-038 has 13,060 songs, 11,467 searchable. The D-040 synthetic gallery has 11,050 songs (the songs with rendered queries and usable tracks).

### 6.2 Query sets
| Set | Origin | Queries | References | Used for |
|---|---|---|---|---|
| D-040 synthetic hold-out | **synthetic** | 8,526 (4,263 whistle, 4,263 wrong-lyrics) | 11,050 songs × up to 24 windows of 8 s | primary synthetic metric |
| HumTrans test | real hums | 769 | MIDI | hum baseline |
| MIR-QBSH (`mir48_anywhere`) | real hums/singing | 4,431 | 48 MIDI songs, no distractors, windows anywhere in the song | hum baseline |
| MLEnd test people | real hums and whistles | 339 whistles, 972 hums (45 people, 26 whistlers) | other people's clips of the same 8 songs | real whistle check |
| CHAD test (live) | real hums | 300 | 11,467 searchable songs | hum baseline vs real recordings |
| MTG-QBH (live) | real singing | 110 | same library | sung baseline vs real recordings |

### 6.3 Protocols
- **Synthetic hold-out** (`eval_query_synth.py`, in memory): each test-split query against up to 24 windows of 8 s of every song; a song's score is its best window. Test songs were never trained on; training songs remain in the gallery as distractors.
- **MLEnd query-by-example** (`eval_mlend.py --split test --song-holdout`, D-013, D-016): each test person's whistle is scored against the centroid of other people's hums of each of the 8 songs; the query's own performer is excluded. It is an 8-way closed set (chance top-1 0.125) and compares query types; it is **not** whistle-against-song retrieval. Hakuna and Potter are the songs held out from whistle-pair training in D-016. Neither E2b nor D-040 trained on MLEnd, so for these two models the split only names a subset. D-041 trained on MLEnd **train-performer** whistle → hum pairs of the other 6 songs (test performers excluded), so for D-041 only Hakuna + Potter is a clean test; its other-6 numbers measure song familiarity (as in D-016).
- **Live search** (`eval_live_search.py`, D-027, D-038): real hums through the full served pipeline against the 11.5k-song index.

Metrics: top-1, top-10 (top-3 for 8-way MLEnd), MRR.

---

## 7. Results

### 7.1 Synthetic whistle and wrong-lyrics queries **[synthetic]**

Held-out songs, 4,263 queries per style, 11,050-song gallery. Source: `docs/paper/results/d040/eval_style_e2b.json`, `eval_style_last.json`, `eval_e2b.json`, `eval_last.json` (print with `python docs/paper/results/d040/summarize.py`).

| Query style | E2b top-1 / top-10 / MRR | D-040 step 3000 top-1 / top-10 / MRR | Δ top-1 |
|---|---|---|---|
| Synthetic whistle | 0.180 / 0.291 / 0.220 | **0.226 / 0.342 / 0.267** | +0.046 |
| Synthetic wrong-lyrics singing | 0.193 / 0.314 / 0.235 | **0.238 / 0.365 / 0.283** | +0.045 |
| Both (8,526) | 0.186 / 0.303 / 0.227 | **0.232 / 0.354 / 0.275** | +0.046 |

These queries come from the same generator the model was trained on (different songs), so this table measures fit to the generator on new melodies, not robustness to real people. One seed. With 4,263 queries per style the binomial standard error of a top-1 near 0.2 is about 0.006, so the gain is far outside query-sampling noise; seed-to-seed variance is not measured.

**D-041 [synthetic].** Same protocol and queries, all four checkpoints re-scored in one chain. Source: `docs/paper/results/d041/eval_style_*.json`, `eval_*.json`. D-041 best.pt is step 9,500, chosen on the synthetic **val** split (mean whistle / wrong-lyrics top-1, ~1.1k val songs, HumTrans guard ≤ 1 point), not on these test numbers; D-041 last.pt (step 11,500) is within 0.003 of it on every row, so the selection does not flatter the result.

| Query style | E2b top-1 / top-10 / MRR | D-040 step 3000 | D-041 best (step 9500) | D-041 last (step 11500) | Δ top-1 D-041 best vs E2b |
|---|---|---|---|---|---|
| Synthetic whistle | 0.180 / 0.291 / 0.220 | 0.226 / 0.342 / 0.267 | **0.253 / 0.378 / 0.297** | 0.254 / 0.379 / 0.298 | +0.073 |
| Synthetic wrong-lyrics singing | 0.193 / 0.314 / 0.235 | 0.238 / 0.365 / 0.283 | **0.269 / 0.401 / 0.314** | 0.269 / 0.403 / 0.315 | +0.076 |
| Both (8,526) | 0.186 / 0.303 / 0.227 | 0.232 / 0.354 / 0.275 | **0.261 / 0.389 / 0.305** | 0.262 / 0.391 / 0.306 | +0.075 |

The D-040 and E2b numbers re-scored in the D-041 chain equal the D-040 files exactly. Over D-040, D-041 adds +0.027 (whistle) and +0.031 (wrong lyrics) top-1. The same caveat applies: same generator as training, one seed, and four changes at once (selection, length, augmentation, MLEnd pairs), so the gain is not attributed to any one of them.

### 7.2 Hum baseline **[real]**

Source: `hum_baseline` in `docs/paper/results/d040/eval_e2b.json` and `eval_last.json`. MIDI references.

| Set | E2b top-1 / top-10 | D-040 step 3000 top-1 / top-10 |
|---|---|---|
| HumTrans test (769) | 0.996 / 1.000 | 0.991 / 1.000 |
| HumTrans test, query +7 st | 0.996 / 1.000 | 0.995 / 1.000 |
| MIR-QBSH, 48 songs, `anywhere`, no distractors (4,431) | 0.955 / 0.993 | 0.944 / 0.991 |

The small hum term (weight 0.15) kept hums nearly intact: −0.005 HumTrans top-1, −0.011 MIR-QBSH top-1. We did not rerun MIR-QBSH with the 2,000 Essen distractors for D-040; E2b scores 0.867 / 0.958 there (`anywhere`, D-028).

D-041 (`docs/paper/results/d041/eval_d041_best.json`, `eval_d041_last.json`), top-1 / top-10:

| Set | E2b | D-041 best (step 9500) | D-041 last (step 11500) |
|---|---|---|---|
| HumTrans test (769) | 0.996 / 1.000 | 0.993 / 1.000 | 0.993 / 1.000 |
| HumTrans test, query +7 st | 0.996 / 1.000 | 0.995 / 1.000 | 0.995 / 1.000 |
| MIR-QBSH, 48 songs, `anywhere` (4,431) | 0.955 / 0.993 | 0.949 / 0.991 | 0.947 / 0.991 |

D-041 costs hums −0.003 HumTrans and −0.006 MIR-QBSH top-1 against E2b, slightly less than D-040.

### 7.3 Real whistles on MLEnd **[real]**

Whistle (spectral peak) → other people's hums, centroid, test people. Source: `docs/paper/results/d040/eval_mlend_e2b.json`, `eval_mlend_last.json`.

| Songs | Whistles | E2b top-1 / top-3 / MRR | D-040 step 3000 top-1 / top-3 / MRR |
|---|---|---|---|
| All 8 | 339 | 0.631 / 0.770 / 0.731 | 0.611 / 0.782 / 0.725 |
| Hakuna + Potter | 90 | 0.589 / 0.711 / 0.698 | 0.656 / 0.789 / 0.741 |
| Other 6 | 249 | 0.647 / 0.791 / 0.742 | 0.594 / 0.779 / 0.720 |

Other rows from the same files (all 8 songs, top-1): whistle via RMVPE-fix → hums 0.584 → 0.599; whistle → whistles 0.608 → 0.617; hum → hums 0.878 → 0.874.

**Reading.** There is no clear real-whistle gain. On Hakuna + Potter, D-040 gets 6 more of 90 whistles right at top-1 (binomial standard error about 0.05, so within noise). On the other 6 songs it gets about 13 fewer of 249 right. Over all 8 songs top-1 falls by 0.020 and top-3 rises by 0.012. Earlier whistle work shows the same pattern of small, song-dependent changes (D-016, D-031). This test also matches whistles to hums, not to the song recording, which is the task we care about (§9).

**D-041.** D-041 trained on MLEnd **train-performer** whistle → hum pairs of the 6 songs other than Hakuna and Potter (test performers excluded). Its Hakuna + Potter row is the clean real number; its other-6 row is **not clean** (the model has learned those tunes from other people). Source: `docs/paper/results/d041/eval_mlend_*.json`; same protocol and key as above.

| Songs | Whistles | E2b top-1 / top-3 / MRR | D-040 step 3000 | D-041 best (step 9500) | D-041 last (step 11500) |
|---|---|---|---|---|---|
| Hakuna + Potter **[real, clean]** | 90 | 0.589 / 0.711 / 0.698 | 0.656 / 0.789 / 0.741 | **0.456 / 0.700 / 0.620** | 0.456 / 0.678 / 0.619 |
| Other 6 **[real, not clean for D-041]** | 249 | 0.647 / 0.791 / 0.742 | 0.594 / 0.779 / 0.720 | 0.707 / 0.863 / 0.803 | 0.707 / 0.867 / 0.804 |
| All 8 **[real, mixed]** | 339 | 0.631 / 0.770 / 0.731 | 0.611 / 0.782 / 0.725 | 0.640 / 0.820 / 0.754 | 0.640 / 0.817 / 0.755 |

**Reading.** On the clean songs D-041 gets 41 of 90 whistles right at top-1, against 53 for E2b and 59 for D-040: worse than both, beyond the ~0.05 standard error against E2b. On the 6 trained songs it gains 0.060 top-1 over E2b. This is the D-015 → D-016 pattern again (held-out songs 0.630 → 0.456 there): real whistle pairs from 6 songs teach those tunes, not whistling in general, and here they also undo D-040's held-out gain. The all-8 number (0.640) mixes the two and should not be read as a whistle gain.

### 7.4 MERT baseline vs the contour encoder (hums) **[real]**

The same hum protocols, MIR-QBSH 48 songs with no distractors and the first 10 s of each reference (`start10`).

| Model | HumTrans test top-1 / top-10 | MIR-QBSH 48 `start10` top-1 / top-10 | Source |
|---|---|---|---|
| MERT-v1-95M, Stage A2 rerun, step 8,000 | 0.947 / 0.999 | 0.169 / 0.602 | `results/d039/eval_step8000.json` |
| Contour encoder E2b | 0.996 / 1.000 | 0.949 / 0.990 | `results/d040/eval_e2b.json`; `start10` from the step-0 validation line of `results/d040/train.log` (E2b weights) |

The MERT run continued Stage A2 from `ckpt/stage_a2/last.pt` with the D-010 recipe for 8,000 more steps (`results/d039/train_mert.yaml`). During training, HumTrans val top-1 went from 0.936 at the first validation to a best of 0.952, the +7 semitone shifted top-1 from 0.822 to 0.860, and MIR-QBSH 48 top-1 never exceeded 0.179 (top-10 at most 0.595) (`results/d039/train_mert.log`). MERT learns HumTrans but does not transfer to MIR-QBSH; the F0 encoder does. The two evaluation scripts differ (`eval.py` for MERT, `eval_contour.py` for the contour encoder), so the HumTrans rows are close but not guaranteed to be the identical protocol; the MIR-QBSH `start10` rows use the same 48 targets and first-10-s crop. A MERT library index (11,973 FMA songs) was built but cannot be scored on real hums because the CHAD and MTG-QBH target audio is gone; that result file is marked invalid (`results/d039/INVALID_README.txt`).

### 7.5 Humming against real recordings, and published systems **[real]**

| System | Query → reference | Protocol | Top-1 / top-10 | Source |
|---|---|---|---|---|
| Ours, E2b, live pipeline | CHAD test hums → full recordings | 300 queries, 11,467 searchable songs | 0.483 / 0.657 | D-038 |
| Ours, E2b, live pipeline | MTG-QBH sung → full recordings | 110 queries, same library | 0.655 / 0.791 | D-038 |
| Ours, E2b | MIR-QBSH → MIDI | 48 targets + 2,000 Essen MIDI distractors, `anywhere` | 0.867 / 0.958 | D-028 |
| CHAD (CREPE) | MIR-QBSH → MIDI | 48 targets in about 2,600 MIDIs | – / 0.921 | [3], Table 1 |
| CHAD (CREPE) | MTG-QBH → recordings | 118 queries, 1,886 imposters | – / 0.883 | [3], Table 1 |
| Google Hum to Search | hums/whistles/singing → recordings | >500k songs | not published | [1] |
| ByteHum | hums → recordings | three benchmarks re-annotated with originals | not verified (paper closed to us) | [2] |

Protocols differ in distractors, references and vocal extraction, so this table places our numbers next to published ones; it is not a ranking. D-040 has not been run through the live pipeline (§9).

D-041 was not re-indexed and not run through the live pipeline: it fails the real held-out whistle check (§7.3), so E2b stays live and no CHAD / MTG-QBH rows exist for it.

---

## 8. Ablations and lessons

### 8.1 What the earlier whistle attempts showed
| Decision | Change | Real-whistle effect | Hum effect |
|---|---|---|---|
| D-013 | No whistle training; spectral-peak tracker vs RMVPE | whistle → hums top-1 0.625 (peak) vs 0.156 (plain RMVPE), all MLEnd clips, 3 seeds | – |
| D-015 | MLEnd whistle pairs on all 8 songs | test people 0.540 → 0.740 | MIR +2000 `anywhere` top-1 −0.024 |
| D-016 | Same, with Hakuna + Potter held out | held-out songs 0.630 → 0.456 (worse): the D-015 gain was song familiarity | – |
| D-031 | Contour-level synthetic whistles, 7,545 melodies | held-out 0.589 → 0.622; all 8 unchanged at 0.631 | MTG-QBH −4.2 top-1 |
| D-040 | Audio-level synthetic whistle + wrong-lyrics singing, re-tracked | held-out 0.589 → 0.656; all 8 0.631 → 0.611 | HumTrans −0.005, MIR-48 −0.011 top-1 |
| D-041 | D-040 + synthetic-val selection, 4× longer, per-epoch query augmentation, MLEnd whistle pairs on 6 songs | held-out 0.589 → 0.456 (worse); other 6 (trained tunes, not clean) 0.647 → 0.707 | HumTrans −0.003, MIR-48 −0.006 top-1 |

Rendering and re-tracking (D-040) costs hums far less than contour editing (D-031), and gives a clear gain on synthetic queries. Neither has yet shown a reliable gain on real whistles. Adding real MLEnd whistle pairs (D-015/D-016, D-041) repeatedly helps the trained tunes and hurts or does not help the held-out ones.

### 8.2 D-039: hum fine-tune with self-match positives
D-039 fine-tuned E2b with synthetic hums made by distorting each song's own vocal F0, plus CHAD train hums and HumTrans. The first run early-stopped at step 500 with no improvement. A rerun with early stopping off for 3,000 steps reached CHAD val top-1 0.4359 at step 2,200 against 0.4341 at step 0 (`results/d039/train_long.log`): no real gain, not adopted. Our reading: positives that are distorted crops of the song's own F0 teach the model little that the CHAD-val task (real hums against vocal-stem melody) needs, because the reference-side errors (stem drop-outs, octave errors, harmonies; D-022) are not in the positives.

### 8.3 The checkpoint-selection bug
D-040 selected checkpoints on HumTrans val top-1 with a +7 semitone shift. That metric was 0.9948 at step 0 (the E2b weights) and never exceeded it; it drifted to 0.9922 by step 900 and stayed there (`results/d040/train.log`). So:
- Run 1 early-stopped at step 500 with "best step 0" (`results/d040/train_run1_earlystop.log`).
- Run 2 (patience off) also selected step 0. Its `best.pt` is the E2b weights, and every number in `results/d040/eval.json` and `eval_mlend.json` equals the E2b files exactly.
- The step-3,000 `last.pt` is what we report as D-040. It was not chosen by looking at test numbers, but it was also not chosen by any validation rule, so it is the end of a fixed schedule.

The same problem appeared in D-012, where HumTrans val was saturated and `last.pt` had to be reported. The lesson: a selection metric must be able to move on the thing being trained. D-041 selects on the synthetic **val** split (mean whistle and wrong-lyrics top-1) with HumTrans as a guard.

**D-041 selection** (`docs/paper/results/d041/train.log`; full curve in `results/d041/README.md`). The new metric moves: synthetic val select (mean whistle / wrong-lyrics top-1) rose from 0.288 at step 0 to 0.335 at step 2,000, 0.358 at 5,000, and a best of 0.374 at step 9,500 (whistle 0.351, wrong lyrics 0.397), then plateaued. Early stopping fired at step 11,500 after 8 validations without a better score. The HumTrans guard (val +7 st top-1 ≥ 0.9848) held at every validation (range 0.9908–0.9948). The first attempt ran out of GPU memory at step ~3,900 and was restarted from scratch with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, same config and seed (D-041). The val gallery is ~1.1k songs, so these values are higher than the 11k-song test numbers and are not comparable to them. The fix worked as a selection rule, but the metric it selects on is synthetic; it does not see the real held-out whistle drop in §7.3.

### 8.4 Other negative results that shaped the method
- Training from scratch on 116 songs of real CHAD hum pairs lost 6 top-1 points on CHAD test (D-019); fine-tuning gently gave +1.0, inside seed noise (D-021).
- Soft salience input built from hard F0 lost on every set (D-030).
- Octave correction of the song melody gave no gain (D-024).
- The largest real-hum gains came from matching, not training: multi-window voting and re-ranking raised CHAD test top-1 from 0.303 to 0.512 (D-023, D-025), and cover → original pairs (E2b) added +2.1 on top (D-028).

---

## 9. Limitations

1. **No real whistle or wrong-lyrics query has been scored against the actual song recording.** MLEnd ships whistles but not the 8 songs; MTG-QBH ships sung queries but not the 81 recordings; the earlier feature-only copies of those recordings were deleted after melody extraction, and downloads from the GPU host are blocked. Our real-whistle evidence is whistle → other people's hums (§7.3), and our sung evidence is MTG-QBH through E2b, not D-040.
2. **MLEnd has only 8 songs** and 26 whistling test people. Per-song effects dominate (§7.3), and the "held-out" pair is 2 songs, 90 whistles.
3. **The main gain is on synthetic queries** generated by the same renderer used in training. The renderer is simple: one-reflection room, no learned singer timbre, fixed formants, a fixed 8-syllable inventory.
4. **One seed** for D-039, D-040 and D-041. Seed spread was 0.297–0.314 CHAD top-1 for D-012 (D-019), so differences of a point or two are not resolved.
5. **No checkpoint selection** for D-040 (§8.3); `last.pt` is reported. D-041 selects on synthetic val only; no real whistle val exists that is free of the trained tunes.
6. **Song side.** Melody comes from the vocal stem. Instrumental hooks and sections with no sung melody are unreachable; on CHAD test, 19.4% of all queries were misses whose reference section was under 30% voiced in the vocal stem (D-022).
7. **Whistles above about 2 kHz** are tracked poorly by RMVPE-fix, and the spectral peak voices nothing at 0 dB SNR with the 6 dB gate (SIGNALS §S8, D-007).
8. **No direct comparison with Google or ByteHum numbers**: Google publishes none, and we could not read ByteHum's results.

---

## 10. Future work

- **Score real whistles and wrong-lyrics singing against recordings.** Needs the 8 MLEnd recordings and MTG-QBH collection audio placed as owned files (`owned/`, D-040 LAUNCH), or a new recorded set of whistled and nonsense-sung queries of open FMA songs.
- **Select on something that moves**: synthetic val with a hum guard (D-041), then real whistle val when it exists.
- **Realistic voices.** Drive SoulX-Singer [17] (melody-control mode takes a continuous F0 and lyrics) or Vevo2 [18] with our F0 and random lyrics, and compare their re-tracked error with the source-filter singer (0.106 st on the smoke clip).
- **Whistle tracker above 2 kHz**: a per-clip speed factor for RMVPE, or a self-supervised pitch model adapted to whistles (D-007, D-031).
- **Run D-040/D-041 through the live pipeline** on CHAD and MTG-QBH (cached song contours), with the D-039 adopt gate (both sets above 0.483 / 0.657 and 0.655 / 0.791).
- **More seeds** for any change smaller than about 2 points.

- **Separate the D-041 changes**: rerun D-041 without the MLEnd pairs (keep the selection fix and query augmentation) to test whether synthetic-only training keeps D-040's held-out MLEnd number while gaining D-041's synthetic number.

---

## 11. Conclusion

A melody-only contour encoder with a whistle-specific tracker is a practical base for whistle and wrong-lyrics search: lyrics and timbre cannot mislead it, and key changes are removed exactly. Rendering synthetic whistles and nonsense-syllable singing from real song melodies, then re-tracking them, improves retrieval of such synthetic queries on new songs by about 4.6 top-1 points and costs hums about half a point. A second round with a working selection rule, longer training and per-epoch augmentation (D-041) raises the synthetic gain to about 7.5 points, still at about half a point of hum cost. Whether that carries over to real people whistling and singing against real recordings is still open: the one real whistle test we can run is mixed for D-040, and for D-041, which also trained on real whistles of 6 of the 8 test songs, it is worse on the 2 songs it never saw (0.456 vs 0.589 for E2b). E2b remains the live model. The target recordings needed for the decisive test are not available yet.

---

## References

Tags: **[V]** read in full or the cited section read; **[A]** abstract or bibliographic record only; **[B]** blog post.

1. Frank, C. (2020, Nov 12). The machine learning behind Hum to Search. Google Research Blog. https://research.google/blog/the-machine-learning-behind-hum-to-search/ **[B][V]**
2. Du, X., Zou, P., Liu, M., Liang, X., Chu, M., & Zhu, B. (2024). ByteHum: Fast and accurate query-by-humming in the wild. *Proc. IEEE ICASSP 2024*, 1111–1115. doi:10.1109/ICASSP48485.2024.10448117 **[A]**
3. Amatov, A., Lamanov, D., Titov, M., Vovk, I., Makarov, I., & Kudinov, M. (2023). A semi-supervised deep learning approach to dataset collection for query-by-humming task. arXiv:2312.01092 **[V]**
4. Kung, Y.-H., Chen, K.-Y., & Ding, J.-J. (2026). Improved query by humming using Conformer-based network with harmonic-aware mechanism. (Venue not verified.) **[A]**
5. Liu, S., Li, X., Li, D., & Shan, Y. (2023). HumTrans: A novel open-source dataset for humming melody transcription and beyond. arXiv:2309.09623 **[A]**
6. Jang, J.-S. R., & Lee, H.-R. (2008). A general framework of progressive filtering and its application to query by singing/humming. *IEEE TASLP* 16(2), 350–358. doi:10.1109/TASL.2007.913035 **[A]**
7. Serrà, J., Araz, R. O., Bogdanov, D., & Mitsufuji, Y. (2025). Supervised contrastive learning from weakly-labeled audio segments for musical version matching (CLEWS). *Proc. ICML 2025*, PMLR 267, 53923–53939. arXiv:2502.16936 **[A]** (loss form taken from the paper as implemented in D-026)
8. Li, Y., Yuan, R., Zhang, G., et al. (2024). MERT: Acoustic music understanding model with large-scale self-supervised training. *ICLR 2024*. arXiv:2306.00107 **[A]**
9. Wei, H., Cao, X., Dan, T., & Chen, Y. (2023). RMVPE: A robust model for vocal pitch estimation in polyphonic music. *Interspeech 2023*, 5421–5425. arXiv:2306.15412 **[V]**
10. MLEnd Hums and Whistles dataset. Queen Mary University of London. https://mlenddatasets.github.io/hums_whistles/ **[A]** (no dedicated paper found)
11. Salamon, J., Serrà, J., & Gómez, E. (2013). Tonal representations for music retrieval: From version identification to query-by-humming. *IJMIR* 2, 45–58. doi:10.1007/s13735-012-0026-0. MTG-QBH: Zenodo record 1290712. **[A]**
12. Gfeller, B., Frank, C., Roblek, D., Sharifi, M., Tagliasacchi, M., & Velimirović, M. (2020). SPICE: Self-supervised pitch estimation. *IEEE/ACM TASLP* 28. arXiv:1910.11664 **[A]**
13. Kim, J. W., Salamon, J., Li, P., & Bello, J. P. (2018). CREPE: A convolutional representation for pitch estimation. *ICASSP 2018*. arXiv:1802.06182 **[A]**
14. Shen, H.-C., & Lee, C. (2007). Whistle for music: Using melody transcription and approximate string matching for content-based query over a MIDI database. *Multimedia Tools and Applications* 35(3), 259–283. doi:10.1007/s11042-007-0128-5 **[A]**
15. Dias, B., Ventura, R., & Gaspar, J. (2008). Automatic transcription of musical-whistling: Comparing pitch detection methods. Instituto Superior Técnico / ISR Lisbon. https://isr.tecnico.ulisboa.pt/publications/automatic-transcription-of-musical-whistling-comparing-pitch-detection-methods/ **[A]** (abstract and conclusions read; venue listed by the authors as JETC 2008)
16. Nilsson, M., Bartůněk, J. S., Nordberg, J., & Claesson, I. (2008). Human whistle detection and frequency estimation. *Proc. CISP 2008*, 737–741. doi:10.1109/CISP.2008.415 **[V]**
17. Qian, J., Meng, H., et al. (2026). SoulX-Singer: Towards high-quality zero-shot singing voice synthesis. arXiv:2602.07803 **[V]** (method sections read)
18. Zhang, X., Zhang, J., Wang, Y., Wang, C., Chen, Y., Jia, D., Chen, Z., & Wu, Z. (2025). Vevo2: Bridging controllable speech and singing voice generation via unified prosody learning. arXiv:2508.16332 **[A]**
19. Jang, J.-S. R. MIR-QBSH corpus. http://mirlab.org/dataset/public/ (dataset) **[A]**

Signal-physics references (Fant, Titze, Shigetomi & Mori, Wilson et al., Peterson & Barney, de Cheveigné, Sakoe & Chiba) are listed with their verification status in [`SIGNALS.md`](SIGNALS.md).

---

## Appendix A. Reproducing the D-040 numbers

Files in `docs/paper/results/d040/` (copied from `/lambda/nfs/hum2song-data/results/d040/`):

| File | What |
|---|---|
| `LAUNCH.md` | commands |
| `run_gen.sh`, `gen.log` | query generation (11,083 songs, 86,110 queries) |
| `run_train.sh`, `train_run1_earlystop.log`, `train.log` | run 1 (early stop at 500) and run 2 (3,000 steps) |
| `run_eval.sh`, `run_all_evals.sh`, `per_style_eval.py` | evaluation |
| `eval_e2b.json`, `eval.json` (best.pt = step 0), `eval_last.json` | synthetic hold-out + hum baseline |
| `eval_style_e2b.json`, `eval_style_last.json` | per-style synthetic split |
| `eval_mlend_e2b.json`, `eval_mlend.json`, `eval_mlend_last.json` | MLEnd test people |
| `smoke.json` | one-clip renderer check |
| `summarize.py` | prints every D-040 number in this paper |

D-041 files are in `docs/paper/results/d041/` (`README.md` lists them; `summarize.py` prints every D-041 number in this paper).

D-039 files in `docs/paper/results/d039/`: `train_long.log` (contour hum fine-tune), `train_mert.yaml`, `train_mert.log`, `eval_step8000.json` (MERT), and the invalid library eval with its README.
