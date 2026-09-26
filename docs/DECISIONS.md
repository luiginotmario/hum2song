# Decision log

A running record of the choices made while building hum2song, the alternatives considered, and the trade-offs. Every entry states what we chose, what we gave up, and when we would revisit it.

---

## D-001 · Pitch extractor for synthetic queries: RMVPE (server), SwiftF0 (on-device option)
**Date:** 2026-09-26 (revised same day)

**Context.** Stage 2 builds synthetic hum/whistle queries from real songs by extracting the vocal melody's pitch contour (F0) and resynthesizing it. That needs a pitch extractor that works on singing, ideally straight from the full mix.

**How the decision evolved.** The first pick was CREPE, contrasted with Google's SPICE (the pitch model behind Hum to Search). We then questioned it: CREPE is from 2018, so we re-checked the 2023 to 2026 literature before locking anything in.

**Options (numbers are as reported by each source; protocols differ, so they are indicative, not directly comparable).**
| Model | Year | Params | Works on full mixes? | Reported accuracy | License |
|---|---|---|---|---|---|
| CREPE | 2018 | 22.2M | No (needs separation; Spleeter+CREPE 91.05 RPA on MIR-1K mixes) | MIR-1K 97.8 RPA (clean) | MIT |
| SPICE | 2019 | 2.38M | Partially | MIR-1K 90.6 RPA | TF Hub |
| FCNF0++ / penn | 2023 | 8.9M | No (20.9 RPA at 0 dB music) | MDB 99.6 RPA | MIT |
| **RMVPE** | 2023 | ~90M | **Yes, designed for it** | MIR-1K mixes 95.42 RPA; clean 97.27 | Apache-2.0 / MIT |
| PESTO v2 | 2025 | 130k | Only with accompaniment training | MIR-1K 97.7 RPA | LGPL-3.0 |
| FCPE | 2025 | 10.6M | No (monophonic) | MIR-1K 96.79 RPA; robust to noise | MIT |
| **SwiftF0** | 2025 | ~14k | No (monophonic) | Pitch F1 0.781, tied top with RMVPE (pitch-benchmark v2) | MIT |

Sources: CREPE arXiv 1802.06182; SPICE arXiv 1910.11664 and [Google blog](https://research.google/blog/spice-self-supervised-pitch-estimation/); penn arXiv 2301.12258; RMVPE arXiv 2306.15412; PESTO arXiv 2309.02265 and 2508.01488; FCPE arXiv 2509.15140; SwiftF0 arXiv 2508.18440; github.com/lars76/pitch-benchmark (note: maintained by the SwiftF0 author).

**Decision.**
- **Server (synthetic data generation): RMVPE.** It is the only open, permissively licensed model built and evaluated for vocal F0 directly from mixes, and it is top-tier on clean singing. Size and speed do not matter on a server.
- Optional label cleaning: also run RMVPE on separated vocal stems and keep only high-confidence frames where both contours agree (unmeasured; to be tested).
- **On-device option: SwiftF0** (tiny, MIT, ONNX, fast). Humming is monophonic, so its weakness on mixes does not matter for live queries. PESTO v2 is the alternative if sub-10 ms streaming latency is needed and LGPL is acceptable.

**Why RMVPE over CREPE.** CREPE (2018) was the original default and was seriously considered: it is accurate on clean, isolated vocals (MIR-1K 97.8 RPA), MIT licensed, and widely used as a baseline. We moved to the newer RMVPE (2023) because:
- **It works on full songs directly.** CREPE needs the vocals separated first; with Spleeter+CREPE it drops to 91.05 RPA on MIR-1K mixes, 81.61 on MIR_ST500 and 74.61 on Cmedia. RMVPE on the raw mix gets 95.42, 89.32 and 83.57 on the same sets. Our input is real songs, so this matters most.
- **Fewer failure points.** Skipping or double-checking the separation step removes a source of artifacts that would otherwise leak into the synthetic hums.
- **Equal on clean vocals.** On clean singing RMVPE is on par with CREPE (97.27 vs 97.8 RPA on MIR-1K), so we lose nothing where CREPE is strongest.
- **Faster.** About 4.9 GFLOPs per second of audio vs CREPE's 141, so labeling a large catalog is much cheaper.
- **More current.** It ties for first in the 2026 pitch benchmark; CREPE is a 2018 model that newer work consistently outperforms.
CREPE stays in the codebase as a swappable baseline and will be reported in the paper's ablation.

**Trade-off / what we gave up.** A fully on-device pipeline. If everything had to run locally on an iPhone (Apple A-series / Neural Engine), a SPICE-style tiny model (today SwiftF0 or PESTO) would be chosen, trading some robustness for size and speed. We prioritized accuracy because inference runs on a server.

**Caveats.** MIR-1K appears in the training data of several of these models, so their MIR-1K scores are optimistic.

**Revisit when.** We pursue on-device inference, or an ablation shows a lighter extractor produces equally good training data. Plan: keep the extractor swappable and report an RMVPE vs CREPE vs SwiftF0 ablation in the paper.

---

## D-002 · Where inference runs: server-first
**Date:** 2026-09-26

**Context.** We considered running the model on the phone (Apple Neural Engine) for faster results.

**Analysis.** Speed barely changes: uploading a ~10 s clip and getting a response takes a few hundred ms, about the same as running MERT-95M on-device. Battery cost per search is small (one or two seconds of Neural Engine compute). The real blocker is the catalog: matching needs the full vector index of song chunks (gigabytes, constantly growing), which can't ship inside an app, so search must hit a server regardless.

**Decision.** Server-first: embedding and search both run server-side.

**Trade-off / what we gave up.** Privacy (raw audio leaves the phone), lower server cost, and partial offline support, which a hybrid design (embed on phone, search on server) would provide.

**Revisit when.** Server cost or privacy becomes a priority; then move embedding on-device (Core ML) and keep search server-side.

---

## D-003 · Sung MIR-QBSH clips train; matching ignores lyrics
**Date:** 2026-09-26

**Context.** Queries are humming, whistling, or singing. Sung queries may use the right lyrics, the wrong words, or nonsense. The match is the melody. The public MIR-QBSH archive does not label hum versus sing on each wav. Holding every one of the 48 songs out of training left Stage A with no sung clips.

**Decision.** `query_type` is required on every manifest row (`hum`, `whistle`, `sing`), duplicated in the SPEC field `qtype`. `waveFile/year2006a` (`2006a-MIR補錄英文歌`, songs `00001`–`00010`) is `sing`. Other MIR-QBSH clips are `hum`. A song that has a sung clip is hash-split instead of forced into test, and every clip of that song shares the split, so the sung recordings train without leaking into the 38 held-out songs. Eval reports top-1, top-3, top-10, and MRR for each query type. Stage 2 reserves `augment_lyric_agnostic` for resynthesis that keeps the melody and replaces the words.

**Trade-off / what we gave up.** The full 48-song Jang holdout. Ten English-song ids can appear in train. Unlabeled course recordings of those same songs are called `hum` and follow the song into that split. Clips outside `year2006a` that were actually sung stay labeled `hum` until a real per-clip label exists.

**Revisit when.** A hum/sing label file is available, or a sung corpus with song-disjoint audio (for example separated vocals) replaces the `year2006a` heuristic.

---

## D-004 · Optimization method: backpropagation, not predictive coding
**Date:** 2026-09-26

**Context.** We asked whether predictive coding, where each layer learns from local prediction errors instead of a single global error passed backward through the whole network, could replace backprop for fine-tuning MERT.

**Options.**
- **Backpropagation + AdamW**: the standard, proven method for fine-tuning transformers; what every baseline we compare against (CHAD, MERT) uses.
- **Predictive coding**: biologically inspired local learning; an active research area.

**Decision.** Backpropagation with AdamW.

**Why not predictive coding.** It needs many iterative settling steps per example, which makes it slow on current GPUs; demonstrated results are on small networks; and nobody has shown it matching backprop on a transformer the size of MERT-95M. For a paper we want to publish and benchmark against CHAD, it would add risk without payoff.

**Trade-off / what we gave up.** Exploring a biologically plausible, local learning rule that could, in principle, avoid a full backward pass.

**Practical alternative for "don't update the whole network."** Freeze the lower MERT layers or train LoRA adapters with backprop. Both are candidates for an ablation.

**Revisit when.** Predictive coding is shown to match backprop on transformer-scale models, or as a separate follow-up experiment.

---

## D-005 · All 48 MIR-QBSH songs held out for a valid CHAD comparison (supersedes part of D-003)
**Date:** 2026-09-26

**Context.** D-003 let the ten English songs with sung clips (`00001`–`00010`) be hash-split, so some could land in train. CHAD's 0.921 top-10 on MIR-QBSH is measured with all 48 songs as queries plus at least 2,000 distractor songs. If any of the 48 appear in training, our number is not comparable, and the eval now reports `comparable_to_chad: false` in that case.

**Options.**
- **Keep D-003:** about 130 more training clips, including the only real sung MIR-QBSH clips, but no valid CHAD comparison.
- **Hold out all of MIR-QBSH:** every MIR-QBSH clip goes to test.

**Decision.** Every MIR-QBSH song and clip is in test. None are in train or val.

**Trade-off / what we gave up.** About 130 training clips (under 1% of roughly 13,000), including 64 real sung clips. Real sung training data drops to almost nothing until Stage 2.

**Revisit when.** Stage 2 adds sung queries from separated vocals, or we report a separate non-CHAD experiment that trains on the sung clips.

---

## D-006 · MERT layerdrop disabled during fine-tuning
**Date:** 2026-09-26

**Context.** MERT-v1-95M ships with `layerdrop: 0.05`. In train mode, each of its 12 transformer layers (attention plus feed-forward block) is skipped with 5% probability on every forward pass. It is a regularizer used in pretraining. Our encoder captures all 12 layer outputs with forward hooks and mixes them with learned weights, because pitch and timbre live in lower layers and structure in higher ones. When the top 6 layers unfroze (step 1,000 in Stage A), the encoder switched to train mode and layerdrop began skipping layers. A pass then returned 11 outputs instead of 12 and training crashed (`hooks captured 11 layers`). Across 12 layers, about 46% of passes skip at least one, since \(1 - 0.95^{12} \approx 0.46\). A GPU smoke test with the unfreeze moved to step 30 caught it before the full run.

**Options.**
- **Set layerdrop to 0:** every layer always runs, so the layer mix is always complete and deterministic.
- **Keep layerdrop and tolerate missing layers:** fill in or renormalize the skipped slots. This adds complexity and noise to the learned layer weights, and it changes what the model sees between train and eval.
- **Use only the last layer:** avoids the crash but throws away the lower layers, where most of the pitch information lives.

**Decision.** `config.layerdrop = 0.0`, set alongside the other pretraining-time masking we already disable in `_disable_spec_augment`. A unit test unfreezes the top layers, runs the encoder in train mode, and asserts that all 12 are captured.

**Trade-off / what we gave up.** One source of regularization during fine-tuning. We judge the cost small because fine-tuning uses a low learning rate, only the top 6 layers train, and dropout plus audio augmentation (pitch shift, noise, crops) still regularize.

**Revisit when.** Stage A shows overfitting (val loss rising while train loss falls). Then we try a small nonzero layerdrop with a layer-mix that handles missing layers, as an ablation.

---

## D-007 · Whistle F0: spectral peak with a tonal-ratio voicing gate, not RMVPE at half speed
**Date:** 2026-09-26

**Context.** The paper's contour figure extracts whistle F0 with RMVPE. At normal speed RMVPE reports mostly the lower octave on whistles, so the figure feeds whistles at half speed and doubles the result ("RMVPE-fix"; definitions and math in `docs/paper/SIGNALS.md` §S6). A 12-clip spot check looked good. We then ran the check on every MLEnd whistle: 1,797 listed, 1,702 analyzed, 95 excluded because RMVPE-fix voiced under 1 s. Details and all numbers are in SIGNALS §S8, from `validate_whistle_f0.py`.

**Options.**
- **RMVPE, normal speed.**
  - A median 55.4% of voiced frames per clip are octave errors, and only 6.8% are within 0.5 st of the spectral peak.
  - In the contour-retrieval test (hum vs same-performer whistle, same song vs other songs) it is at chance: AUC 0.49 / 0.52 / 0.57 on Potter / StarWars / Hakuna.
- **RMVPE, half speed, ×2 (the current method).**
  - Per clip, a median 98.2% of frames are within 0.5 st of the spectral peak (IQR 96.7–99.1%), and 91.1% of clips exceed 90%.
  - It still fails on whistles above about 2 kHz. Clips with a median peak of 2–2.5 kHz have a median agreement of 20.7%, and most of the 95 excluded clips are higher still (median 3.74 kHz). The spectrum shows no energy where RMVPE-fix puts F0, so these are a second octave error.
  - On synthetic tones it is correct up to 1.4 kHz, 74–85% at 2 kHz and ≤ 10% at ≥ 2.8 kHz.
  - Retrieval AUC is 0.766 / 0.711 / 0.798 on common pairs.
- **Spectral peak with a voicing gate.** F0 is the STFT argmax in 200–6000 Hz with parabolic interpolation. A frame is voiced when ≥ 6 dB more energy lies within ±50 cents of the peak than in the rest of the band.
  - It agrees with RMVPE-fix on a median 99.7% of jointly voiced frames and has no octave ceiling.
  - Retrieval AUC is 0.840 / 0.773 / 0.827: +0.073, +0.063 and +0.029 over RMVPE-fix, with 95% performer-bootstrap CIs excluding 0 on all three songs.
- **RMVPE with a per-clip speed factor** (for example ×4 for whistles above about 2 kHz). This is plausible but untested.

**Decision.** Use the spectral peak with the 6 dB tonal-ratio gate as the whistle F0 in the paper, followed by the same contour cleaning as for hums. Keep RMVPE (normal speed) for hums, where the spectral peak is not F0 in 20.6% of voiced frames (SIGNALS §S2–S3). Report RMVPE-fix agreement as a cross-check.

**Trade-off / what we gave up.**
- **One estimator for all query types.** Whistles and hums now go through different front ends, so the paper must describe both.
- **Low-SNR robustness.** With the 6 dB gate the peak voices nothing on synthetic tones at 0 dB broadband SNR, where RMVPE-fix still voices tones up to 1.4 kHz.
- **Harmonic-rich "whistles."** In about 0.3% of frames the peak sits on the 3rd harmonic, where RMVPE is plausibly right. This happens mostly for one performer.
- **Stable background tones** can capture the peak.
- **Threshold sensitivity.** A too-strict gate (prominence ≥ 60 dB) is worse than RMVPE-fix. The 6 dB rule was one of four pre-listed rules, not tuned on the retrieval test, but chosen after seeing its voicing agreement on the same whistles.
- **Figure consistency.** The existing figure (median r 0.80 vs 0.57, AUC 0.75) was made with RMVPE-fix. It must be regenerated with the peak method or labeled as RMVPE-fix.

**Revisit when.**
- A whistle corpus with reference F0 exists, so we can measure accuracy instead of agreement.
- We test the per-clip speed-factor variant of RMVPE.
- Low-SNR or phone-microphone whistles (our target use) show the 6 dB gate dropping too many frames.
