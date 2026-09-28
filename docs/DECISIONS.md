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
  - In the contour-retrieval test (hum vs same-performer whistle, same song vs other songs) it is at chance: AUC 0.49 / 0.53 / 0.50 on Potter / StarWars / Hakuna (0.49 / 0.52 / 0.57 before the D-008 DTW fix).
- **RMVPE, half speed, ×2 (the current method).**
  - Per clip, a median 98.2% of frames are within 0.5 st of the spectral peak (IQR 96.7–99.1%), and 91.1% of clips exceed 90%.
  - It still fails on whistles above about 2 kHz. Clips with a median peak of 2–2.5 kHz have a median agreement of 20.7%, and most of the 95 excluded clips are higher still (median 3.74 kHz). The spectrum shows no energy where RMVPE-fix puts F0, so these are a second octave error.
  - On synthetic tones it is correct up to 1.4 kHz, 74–85% at 2 kHz and ≤ 10% at ≥ 2.8 kHz.
  - Retrieval AUC is 0.811 / 0.717 / 0.795 on common pairs (0.766 / 0.711 / 0.798 before D-008).
- **Spectral peak with a voicing gate.** F0 is the STFT argmax in 200–6000 Hz with parabolic interpolation. A frame is voiced when ≥ 6 dB more energy lies within ±50 cents of the peak than in the rest of the band.
  - It agrees with RMVPE-fix on a median 99.7% of jointly voiced frames and has no octave ceiling.
  - Retrieval AUC is 0.851 / 0.773 / 0.823: +0.040, +0.056 and +0.028 over RMVPE-fix, with 95% performer-bootstrap CIs excluding 0 on all three songs (+0.020 to +0.063, +0.027 to +0.086, +0.009 to +0.048). Before the D-008 DTW fix these were 0.840 / 0.773 / 0.827 and +0.073 / +0.063 / +0.029; the Potter gap roughly halved but the ranking held.
- **RMVPE with a per-clip speed factor** (for example ×4 for whistles above about 2 kHz). This is plausible but untested.

**Decision.** Use the spectral peak with the 6 dB tonal-ratio gate as the whistle F0 in the paper, followed by the same contour cleaning as for hums. Keep RMVPE (normal speed) for hums, where the spectral peak is not F0 in 20.6% of voiced frames (SIGNALS §S2–S3). Report RMVPE-fix agreement as a cross-check.

**Trade-off / what we gave up.**
- **One estimator for all query types.** Whistles and hums now go through different front ends, so the paper must describe both.
- **Low-SNR robustness.** With the 6 dB gate the peak voices nothing on synthetic tones at 0 dB broadband SNR, where RMVPE-fix still voices tones up to 1.4 kHz.
- **Harmonic-rich "whistles."** In about 0.3% of frames the peak sits on the 3rd harmonic, where RMVPE is plausibly right. This happens mostly for one performer.
- **Stable background tones** can capture the peak.
- **Threshold sensitivity.** A too-strict gate (prominence ≥ 60 dB) is worse than RMVPE-fix. The 6 dB rule was one of four pre-listed rules, not tuned on the retrieval test, but chosen after seeing its voicing agreement on the same whistles.
- **Figure consistency.** The first figure (median r 0.80 vs 0.57, AUC 0.75) was made with RMVPE-fix. It has been regenerated with the peak method and the D-008 DTW: median r 0.85 vs 0.59, AUC 0.828 (`docs/paper/results/fig_contours_potter_stats.json`).

**Revisit when.**
- A whistle corpus with reference F0 exists, so we can measure accuracy instead of agreement.
- We test the per-clip speed-factor variant of RMVPE.
- Low-SNR or phone-microphone whistles (our target use) show the 6 dB gate dropping too many frames.

---

## D-008 · Contour DTW: length-normalize, then always slope-constrain (no silent fallback)
**Date:** 2026-09-26

**Context.** The paper's contour comparison (SIGNALS §S7) aligns two key-normalized contours with DTW whose steps (1,1),(1,2),(2,1) limit the local tempo ratio to 1/2–2. A slope-limited path cannot exist when the two voiced sequences differ in length by more than 2×. The first implementation then caught the failure and silently re-ran DTW with standard steps (1,0),(0,1),(1,1), which have no slope limit. In the Potter hum-vs-whistle statistics this affected 189 of 1,230 pairs with RMVPE-fix whistles, and 824 of 922 with normal-speed RMVPE. Those pairs were scored under a different, looser alignment than the rest, and the figure's numbers did not say so. The contour figure and the whistle-F0 validation each carried a copy of this logic.

**Options.**
- **Keep the fallback, report the count.** No code change, but two alignment rules stay mixed in one statistic. Unconstrained steps fit different songs well: normal-speed RMVPE, mostly on the fallback, scored median r 0.80 same-song vs 0.79 other-song.
- **Drop pairs outside the 1/2–2 length ratio.** Every scored pair is constrained, but it removes about 15% of pairs (189 of 1,230 with RMVPE-fix whistles) and the removal depends on the F0 method's voicing, so methods would be compared on different pairs.
- **Resample only the pairs outside 1/2–2.** Fixes the failures but still treats two groups of pairs differently.
- **Always resample the query to the reference length, then slope-constrained DTW** (linear scaling + DTW, as in pitch-vector QBSH systems). Every pair gets the same rule, and with equal lengths a constrained path always exists.
- **Subsequence DTW** (query may match any part of the reference). Handles partial renditions, but adds free endpoints that also help wrong songs match; untested.

**Decision.** Always resample the query to the reference length, then run slope-constrained DTW (steps (1,1),(1,2),(2,1), weights 1/1.5/1.5, 25% Sakoe–Chiba band). There is no fallback: a pair without a valid path is returned as unaligned, counted (`n_unaligned`) and left out. One shared function (`score_pair` in `docs/paper/scripts/contour_pipeline.py`) is used by both `contour_figure.py` and `validate_whistle_f0.py downstream`.

Measured result: 0 unaligned pairs in every run (contour figure: 5,586 pairs; downstream: 8 whistle-F0 variants on Potter, 6 each on StarWars and Hakuna). Pairs that used to fall back and are now constrained: 189 of 1,230 (Potter, RMVPE-fix), 116 of 1,288 (Potter, spectral peak), 154 of 1,222 and 132 of 1,214 (StarWars, Hakuna, RMVPE-fix). Effect, Potter, RMVPE-fix whistles: AUC 0.750 → 0.791, median r same song 0.80 → 0.82, other song 0.57 → 0.59. Normal-speed RMVPE: r 0.80 / 0.79 → 0.52 / 0.53, so the fallback had been hiding its failure. The spectral peak's AUC advantage over RMVPE-fix (D-007) shrank from +0.073 to +0.040 on Potter and stayed significant on all three songs. Sources: `docs/paper/results/` (current) vs `docs/paper/results/superseded/*_v1.json` (before), details in `docs/paper/WHISTLE_F0_VALIDATION.md`.

**Trade-off / what we gave up.**
- **Global tempo as evidence.** Linear resampling removes the overall tempo and voiced-duration difference, so a whistle with far less voiced material than the hum is no longer penalized for it.
- **Partial renditions.** Linear scaling assumes both clips cover the same stretch of melody. A clip covering only part of the tune gets stretched over the whole reference. We have not measured how often this happens in MLEnd.
- **Comparability with the first figure.** All contour numbers changed (see the old-vs-new table in `docs/paper/REFERENCES_and_CAPTION.md`); the first version is kept only as superseded files.

**Revisit when.**
- Any run reports `n_unaligned > 0`.
- Queries are partial or much shorter than the reference (e.g. real user hums against full songs); then test subsequence DTW with the same slope limit.
- The retrieval model replaces DTW scoring in the paper's main results, leaving this only for the figure.

---

## D-009 · Song library sourcing: open full-track audio for the paper, no yt-dlp
**Date:** 2026-09-26 (drafted), 2026-09-27 (adopted)

**Context.** hum2song has two jobs. It should retrieve the commercial songs people get stuck on (TikTok and pop, house and club, SoundCloud remixes). It must also produce reproducible numbers for the paper. SPEC §1 listed "yt-dlp personal downloads + FMA + MTG-Jamendo". The research note `research/song_library_sources.md` (not in this repo) found:
- YouTube's terms and API policies forbid downloading or storing audio.
- Spotify (Developer Policy III.13–14) and SoundCloud (API terms: no AI, no fingerprints) explicitly forbid this use.
- Apple iTunes previews are promotional only: "streamed only, and not downloaded, saved, cached".
- Deezer allows private or family use only and bans storing audio. New app registration is closed, but there is an approval route (§IX).
- Open full-track datasets are freely usable for research: FMA (106,574 tracks), MTG-Jamendo (55,701), JamendoMaxCaps (362k, instrumental) and IAMD (>34k h).
- Previews cover about 30 s of a roughly 3.5 min track (5 of about 41 chunks), usually from fixed default start points. Earworm studies find the chorus is the stuck part only about 33 to 39 % of the time.

**Options.**
- **A.** Scrape previews and embed them. Cheap, but it breaches the terms and covers only the preview.
- **B.** Keep yt-dlp full downloads. It breaches the terms, risks anti-circumvention claims, and cannot be published.
- **C.** Open data only. Clean and reproducible, but it has no commercial hits.
- **D.** Tiered. Open data for the paper, lawfully owned full tracks for the demo, and previews or licensed audio only with permission.
- **E.** A licensed catalogue (7digital, Songtradr). Clean and full-track, but it costs money and needs contracts.

**Decision: D.**
- **Paper results use Tier P:** FMA and MTG-Jamendo as targets, JamendoMaxCaps and IAMD as distractors.
- **The demo index adds Tier O:** audio Luigi owns lawfully.
- **Tier L waits for written permission:** Deezer or Apple previews, Beatport, label or DJ permissions.
- **Song lists** come from Apple RSS, Deezer charts (metadata storage is allowed), Last.fm, ListenBrainz/MusicBrainz and curated seeds.
- **yt-dlp is removed** from the pipeline and the SPEC.
- **Every song is tagged** with `coverage` (full or preview) and `source_tier`. Only code, the open-data index and aggregate metrics are published.

**Trade-off / what we gave up.**
- **No commercial breadth at first.** The demo starts with owned and open songs, without the breadth of TikTok, club and SoundCloud songs.
- **No commercial-scale claim** in the paper.
- **SoundCloud-only content** is out unless the uploader gives permission.
- **Time** spent waiting on permission replies.
- **Preview gaps.** Where previews are later allowed, sections outside the preview (bridges, later drops) will be missed for those songs.

**Revisit when.**
- Deezer, Apple or Beatport answers a permission request.
- Measured preview-only recall is known.
- A university affiliation makes EU DSM Art. 3 available.
- A licence budget appears.
- Platform terms change.
- Before camera-ready, when all terms are re-checked.

---

## D-010 · Fix the MIDI tempo map; train key- and tempo-invariant from the Stage A weights
**Date:** 2026-09-27

**Context.** A diagnosis of the Stage A checkpoint (`ckpt/stage_a/last.pt`, 20,000 steps) found three problems. Artifacts: `/lambda/nfs/hum2song-data/diag/` (`mir_experiments.json`, `humtrans_robustness.json`, the pitch-baseline and reference-inspection logs).
- **Weak MIR-QBSH retrieval.** Against the 48 MIR-QBSH targets only (no distractors), Stage A scored top-10 0.371. Chance is 0.208, and base MERT scored about 0.30. A pitch-only melody baseline on the same queries got top-1 0.888 and top-10 0.981. So the melody information is there, and the model is not using it.
- **The model learned "same pitch at the same time", not relative melody.** HumTrans hums were recorded in sync with their label MIDI, so every hum/reference pair shares key and timing. On the HumTrans test set, top-1 is 0.770 unshifted, 0.246 with the query shifted +7 semitones, and 0.229 an octave down. A 0.8× tempo change drops it to 0.492. Real users hum in any key and at any speed.
- **Renderer bug.** `midi_render.py` reset tempo to 120 bpm at the start of every track. Type-1 MIDI files keep their tempo in track 0 and notes in track 1, so their notes were timed at 120 bpm whatever the real tempo was. 17 of 48 MIR-QBSH songs and about 66% of HumTrans references were rendered at the wrong speed. Re-rendering at the right tempo alone raised MIR-QBSH top-10 from 0.37 to 0.48 with the same Stage A weights.
- **Eval reporting.** The MIR-QBSH eval used HumTrans renders as distractors. Those are the training distribution and share its synthesis. The docs also said references are scored by their best chunk. In fact eval embeds one clip per reference: its first 10 s.

HumTrans structure matters for the fix. All 1,000 segments (song × segment id) have 6–20 takes by different singers. In 79 segments every take's MIDI is byte-identical. In the other 921 the MIDIs have the same intervals and timing (duration ratio 0.995–1.004, 5th–95th percentile) but sit at different octaves per singer (−36 to +12 semitones against the first take).

**Options.**
- **Fix the tempo only, keep training as is.** Cheapest, and worth +0.11 top-10 on its own, but the model stays locked to absolute pitch and timing.
- **Retrain from base MERT with invariance augmentations.** Clean, but throws away 20,000 steps that did learn hum-vs-render features.
- **Continue from the Stage A weights with invariance augmentations** (weights only, fresh optimizer, short warmup).
- **Train on pitch contours instead of audio** (like the pitch baseline). Strong on MIR-QBSH, but it gives up the audio model's robustness to sung lyrics and noise, and it is a different project.

**Decision.** Continue from Stage A (run `stage_a2`, `configs/train_stage_a2.yaml`, checkpoints in `ckpt/stage_a2/`), after fixing the data.
1. **Tempo map.** `parse_midi` gathers tempo events from all tracks into one tempo map, then converts note ticks to seconds (`TempoMap`). No new dependency. It matches `pretty_midi` note times on all 48 MIR-QBSH MIDIs to within 1e-14 s. `training/scripts/rerender_catalog.py` moves the old `catalog/humtrans` and `catalog/mirqbsh` folders and both manifests to `backup/pre_tempo_fix_<timestamp>/`, renders again, and rewrites `song_dur_s` / `duration_s`.
2. **Query augmentation** (`augment.py`). Pitch and tempo are now independent. Transposition: 80% of queries, uniform in ±12 semitones, of which 25% are exactly ±1 octave. Time-stretch: 50% of queries, log-uniform 0.7–1.4× duration, pitch unchanged. Both run as one linear resample plus one numpy phase-vocoder pass (1024-point FFT, hop 256). Before this change, "pitch" and "time" were both resampling, so each changed the other.
3. **Pair construction** (`dataset.py`). The reference crop starts 0–2 s after the aligned start, on top of the ±1 s jitter, so hum and reference crops are offset. With probability 0.5 the positive is another singer's or take's reference for the same HumTrans segment. For the 921 octave-varied segments that is an octave-shifted positive with the same timing. For the 79 identical segments it changes nothing.
4. **Loader cost.** Data loading was the bottleneck. The query is cut to the longest window a crop can need (12 s / 0.7) before augmenting. `trim_silence` is vectorized. ffmpeg runs with `-threads 1`. Each dataloader worker uses one torch thread. The biggest cost turned out to be elsewhere. The GPU box's distro ffmpeg links OpenBLAS and libgomp, which start one spinning thread per core on every launch. Each codec call cost about 1.7 CPU-seconds, and `-threads 1` did not change that. Setting `OMP_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` for the subprocess cuts it to about 0.1 s. Measured at launch: 2.5 steps/s with the GPU at 99%. Stage A ran at 0.89 steps/s (20,000 steps in about 6 h 15 min), with heavier augmentation now included.
5. **Eval.** Every result has a `targets_only` line: the set's own references and nothing else (48 songs for MIR-QBSH). HumTrans songs are never distractors. No other song audio exists yet, so `targets_only` is the headline MIR-QBSH number (`headline` field). It is not comparable to CHAD's 0.921, which uses about 2,600 MIDI distractors. The docs now say eval embeds one clip per reference.
6. **Validation during training.** At step 0 and every 1,000 steps, the loop logs to W&B: MIR-QBSH top-1/top-10/MRR against the 48 targets only (all ~4,400 queries), HumTrans val, and HumTrans val with queries shifted +7 semitones (`val/humtrans_val_shift+7_*`), which tracks key robustness directly.
7. **Schedule.** 8,000 steps, batch 32, same InfoNCE with learned temperature, top 6 MERT layers trainable from step 0 (Stage A already unfroze them), layerdrop 0, 300 warmup steps, then cosine decay, same learning rates as Stage A. With validation every 1,000 steps that gives 9 points on the curve. Stage A's loss was still falling at 20,000 steps, but this run starts from trained features and only has to learn the invariances. If MIR 48-target top-10 is still rising at 8,000, the next run extends. If it peaks early, we keep the best `step_*.pt`, not `last.pt`.

**Trade-off / what we gave up.**
- **Peak HumTrans accuracy.** Unshifted HumTrans val top-1 will probably drop. That shortcut (absolute pitch and timing) was real signal on HumTrans but useless on real queries.
- **A clean ablation.** Tempo fix, augmentations, offsets, and cross-take positives change together, and the run starts from weights trained on wrongly timed references. We cannot say which change helped how much without extra runs. For the tempo fix alone we have the diagnosis number (0.37 → 0.48 top-10, same weights).
- **Augmentation fidelity.** The phase vocoder smears transients and sounds phasey at large factors: transposing up an octave plus a 1.4× stretch is a 2.8× vocoder stretch. Linear-interpolation resampling aliases slightly when shifting up. We chose speed over quality because the loader is the bottleneck.
- **A comparable headline number.** Without distractors, MIR-QBSH `targets_only` is a 48-way closed set and cannot be read next to CHAD.
- **Cross-take coverage.** Other takes only add octave variation. Key variation within an octave comes only from synthetic transposition.

**Revisit when.**
- We have non-HumTrans song audio (FMA / MTG-Jamendo or MIDI renders from another corpus) for at least 2,000 distractors. Then report the CHAD-comparable number next to `targets_only`.
- `val/humtrans_val_shift+7_top1` stays far below unshifted val top-1 after 8,000 steps. Then raise the transposition probability, or add hum-vs-hum positives across takes.
- MIR 48-target top-10 stays well below the pitch baseline (0.981). Then consider adding a pitch-contour branch, or training on contours.
- The loader is still the bottleneck. Then precompute stretched and transposed variants, or move augmentation to the GPU.

---

## D-011 · Key-invariant pitch-contour encoder replaces MERT for melody matching
**Date:** 2026-09-27

**Context.** Stage A2 (D-010, W&B run `5rst2r1c`) plateaued. Against the 48 MIR-QBSH targets it reached top-10 0.583 at best (step 5000), HumTrans val top-1 0.936, and 0.822 with queries shifted +7 semitones. A pitch-only baseline with no learning, using hand-labelled pitch, removing the key and trying 13 tempo scales, got top-1 0.888 / top-10 0.981 on the same songs. The melody is in the input, but MERT fine-tuned on about 1,000 HumTrans melodies does not pick it out, and key augmentation of the audio does not make it key-invariant.

Recent query-by-humming work points the same way. CHAD (Amatov et al., ISMIR 2023) trained the same metric-learning model on CREPE F0 and on CQT: F0 got top-10 0.921 on MIR-QBSH with about 2,600 MIDI distractors, and CQT got 0.840. Jeong's melody-embedding QbH (SK Telecom, `jdasam/qbh_project`) encodes a (pitch, voicing) contour, not audio. Hum to Search (Google, 2020) and ByteHum (ICASSP 2024) learn from spectrograms, but with far more paired data than our 13,000 HumTrans hums. A 2026 Conformer QbH model ("Improved Query by Humming Using Conformer-based Network with Harmonic-Aware Mechanism") uses convolution plus self-attention over the melody. None of these pass audio through a general music foundation model.

**Options.**
- **More MERT training or augmentation.** Stage A2's curve was flat from step 3,000 to 8,000, so more of the same looked unlikely to close a 0.40 top-10 gap.
- **Late fusion of MERT with a DTW pitch matcher.** Uses what exists, but DTW scoring costs O(songs × tempo scales) per query, and it does not scale to a real catalogue.
- **Learned contour encoder, alone or fused with MERT.** RMVPE F0 → semitones → minus the crop's median voiced pitch, plus a voicing flag, at 20 ms frames. A small conv + transformer turns that into one 256-d vector, trained with the same symmetric InfoNCE and learned temperature. Key invariance is exact by construction, not learned. Retrieval stays one dot product per reference window.

**Decision.** Build the contour encoder (`training/src/hum2song/contour/`) and test it alone first. Fusion with MERT waits until the contour branch is shown to need help.
- **Queries.** RMVPE (D-001) on the untrimmed query audio (`training/scripts/extract_f0.py`, cached in `$H2S_DATA/f0/`). Voiced where confidence ≥ 0.3. Voiced runs under 60 ms dropped, octave jumps folded, 3-frame median. On MIR-QBSH the queries go through RMVPE, not the hand-labelled `.pv` files.
- **References.** Rendered straight from the melody MIDI (highest note per frame, rests unvoiced). The references are MIDI today, so this is exact. Real song audio would need a vocal-melody extractor on the reference side.
- **Model.** Conv front end (the second conv halves the frame rate, padded frames masked), 6-layer pre-norm transformer, d = 256, 4 heads, masked attention pool, MLP to 256-d, L2 norm. About 5.2 M parameters, one tower for both sides.
- **Training.** HumTrans train only (13,080 hums). MIR-QBSH stays fully held out (D-005). Each pair is a random 3–12 s hum window and the MIDI window at the same time, with ±1 s start jitter, −1 to +3 s of extra length and a 0.8–1.25× stretch. Hum-side augmentation: 0.6–1.7× duration, ±20% local tempo warp, ±15% interval scaling, ±0.7 semitone slow drift, 0.15 semitone jitter, voicing gaps, and occasional one-octave tracker errors. Batch 256, AdamW lr 3e-4, wd 0.05, 300 warmup steps, cosine decay, bf16.
- **Checkpoint selection.** On HumTrans val shifted +7 top-1, never on MIR-QBSH.
- **MIR-QBSH reference protocols** (48 targets, no distractors, all 4,431 queries):
  - `start10`: the first 10 s from the first note. The same crop Stage A2's eval uses.
  - `start_multi`: the first 6/8/10/12/15 s, best window. This is a tempo search on the reference side, like the baseline's 13 scales. It relies on MIR-QBSH queries starting at the song start.
  - `anywhere`: 6/10/14 s windows every 1 s over the whole song. This makes no start assumption.

**Feasibility (3,000 steps, W&B `lugkvys8`, about 14 min of training plus 15 min of F0 extraction on the A100).** Selected checkpoint (step 1,500, chosen on HumTrans val):

| | MIR-48 top-1 | MIR-48 top-10 | HumTrans val top-1 | HumTrans val +7 top-1 |
|---|---|---|---|---|
| Stage A2 best step (MIR-selected, so best case) | – | 0.583 | 0.936 | 0.822 |
| Pitch baseline, hand-labelled `.pv` (13 scales, from song start) | 0.888 | 0.981 | – | – |
| Pitch baseline, RMVPE queries (same method) | 0.903 | 0.986 | – | – |
| Contour encoder, `start10` | 0.882 | 0.984 | 0.991 | 0.990 |
| Contour encoder, `start_multi` | 0.945 | 0.991 | | |
| Contour encoder, `anywhere` | 0.863 | 0.982 | | |

The best MIR value over all validation steps was `start_multi` top-1 0.951 / top-10 0.991. It is a best case because it was picked by looking at MIR. Under the same reference protocol (song start, tempo search), the encoder beats the RMVPE pitch baseline at top-1 (0.945 vs 0.903) and top-10 (0.991 vs 0.986). With a single 10 s window, as Stage A2 used, top-10 goes from 0.583 to 0.984. HumTrans no longer depends on key: shifted and unshifted scores match.

**Full run (6,000 steps, W&B `hbufzlsf`; selected step 2,500; HumTrans test, not val).** MIR-48 `start10` 0.884 / 0.985, `start_multi` 0.949 / 0.991, `anywhere` 0.869 / 0.983 (top-1 / top-10). HumTrans test top-1 0.988, +7 semitones 0.980. With 2,000 Essen distractors (D-012), `start_multi` top-10 is 0.940. That run and the feasibility run have a bug, found afterwards and fixed in D-012: the dataloader workers never saw the epoch counter, so every epoch replayed the same crops and augmentations. The three runs first launched in parallel (`plc45h43`, `51adgn6y`, `nootcoi5`) ran out of GPU memory in their first steps. They show as crashed in W&B and have no results.

**Trade-off / what we gave up.**
- **Timbre and lyrics.** The encoder sees only the F0 track, so it cannot use lyrics, timbre or accompaniment. It depends on RMVPE: noisy rooms, heavy reverb or breathy humming that RMVPE cannot track give it nothing. Whistles need the spectral-peak F0 (D-007), which is not wired in yet. MLEnd whistles have no references to test against.
- **Symbolic references.** References come from MIDI, not audio. Against real recordings, reference contours would come from a vocal-melody extractor with its own errors, and these numbers would drop.
- **Still no CHAD comparison.** Numbers are against the 48 targets only. `start_multi` also leans on MIR-QBSH's start-of-song queries. `anywhere` is the protocol to watch for real use.
- **Two towers of code.** The MERT pipeline stays for fusion and for the paper's ablation, so there are two training entry points.

**Revisit when.**
- We have ≥ 2,000 distractor melodies (e.g. the MIREX ~2,600 Essen/MIDI set). Then report the CHAD-comparable number.
- Queries come from noisy or real-world audio, or references from real songs. Then test fusion with MERT embeddings, and a melody extractor on the reference side.
- Whistle queries matter. Then feed the spectral-peak F0 (D-007) through the same encoder.

---

## D-012 · Contour encoder: fix epoch reseeding, add a MIREX-style distractor eval, keep training HumTrans-only
**Date:** 2026-09-27

**Context.** After D-011, the 48-target MIR-QBSH top-10 was already about 0.98 to 0.99 and no longer separated models. Two further problems came up.
1. **The training data repeated itself.** `ContourPairDataset` stored its epoch as a plain attribute. The dataloader uses persistent workers, and workers only get a copy of the dataset, so they never saw the new epoch. Every epoch replayed the same crops and augmentations. Loss fell to about 0.01, a sign of memorization.
2. **The eval had no distractors.** The 48-way closed set could not be read next to published QbH results. MIREX QBSH, and CHAD's 0.921 top-10, rank the MIR-QBSH queries against the 48 targets plus Essen folk-song MIDIs as distractors.

**Options.**
- **Epoch:** turn off persistent workers (respawns 16 workers every epoch), or keep the epoch in shared memory.
- **More melodies:** HumTrans has about 1,000 distinct segments. Candidates for extra melodies: MIDI-only synthetic pairs from Essen (Jeong's self-supervised stage), Lakh MIDI melody tracks, or POP909. POP909 and Lakh would need an overlap check against the MIR-QBSH pop songs.
- **Distractors:** the exact MIREX list (not available to us), or a documented stand-in of 2,000 Essen German songs.

**Decision.**
- **Epoch counter.** It now lives in shared memory (`EpochCounter`, a `multiprocessing.Value`). A test checks that persistent workers see a new epoch.
- **Distractor eval.** `eval_contour.py --distractors raw/essen_midi/deutschl --distractor-count 2000` adds 2,000 Essen German songs, spread evenly over the sorted files, to the MIR-QBSH reference side. The files come from `ccarh/essen-folksong-collection` via `essen_to_midi.py` (music21, 120 bpm). The metrics are named `mir48+2000_*`.
- **Synthetic pairs.** `SyntheticPairDataset` and `configs/train_contour_synth.yaml` build MIDI-only pairs: the query is the same melody window after `humanize` (glides and vibrato) and the usual augmentation. They use Essen German songs outside the 2,000 distractors plus Essen China, 5,605 melodies in all. An overlap check (`docs/paper/results/contour/essen_overlap.*`) compared each MIR-QBSH target with every training melody. MIR-QBSH 00040 was 0.47 semitones MAE from deut2282 ("Ich hab mich ergeben"). The next closest pair was 0.83, so deut2282 is excluded.
- **Headline config: `configs/train_contour.yaml`, HumTrans-only, reporting `last.pt`.** HumTrans val is saturated (shifted top-1 ≥ 0.99 from step 1,000), so it cannot rank checkpoints. Selecting on it picked step 1,500 for `contour_v2_s0` because of a tie. `last.pt` (end of cosine decay) is the checkpoint chosen before looking at results, so it is the one reported. Checkpoint choice never uses MIR-QBSH.

**Results.** HumTrans **test** and MIR-QBSH, all 4,431 queries, RMVPE F0 for every query. Values are top-1 / top-10. The three seeds of the headline config are W&B `f4086zb7`, `5wgj4pbq`, `3yx3d3nh` (mean ± sd). JSON files are in `docs/paper/results/contour/`.

| | MIR-48 start10 | MIR-48 start_multi | MIR-48 anywhere | +2000 start10 | +2000 start_multi | +2000 anywhere | HumTrans test top-1 | +7 st top-1 |
|---|---|---|---|---|---|---|---|---|
| Stage A2 (MERT, best MIR step, best case) | – / 0.583 | | | | | | 0.936 (val) | 0.822 (val) |
| Pitch baseline, RMVPE queries, 13 scales | | 0.903 / 0.986 | | | 0.722 / 0.899 | | | |
| D-011 run with the epoch bug (`hbufzlsf`, step 2,500) | 0.884 / 0.985 | 0.949 / 0.991 | 0.869 / 0.983 | 0.639 / 0.854 | 0.805 / 0.940 | 0.616 / 0.851 | 0.988 | 0.980 |
| **Epoch fix, HumTrans-only, `last.pt`, 3 seeds** | **0.961±0.005 / 0.994±0.000** | **0.977±0.001 / 0.995±0.000** | **0.959±0.004 / 0.993±0.001** | **0.853±0.008 / 0.955±0.002** | **0.922±0.003 / 0.974±0.002** | **0.866±0.009 / 0.958±0.003** | **0.996±0.002** | **0.993±0.001** |
| + Essen synthetic pairs (`ezzpz0ff`, `last.pt`) | 0.967 / 0.995 | 0.977 / 0.994 | 0.958 / 0.994 | 0.847 / 0.962 | 0.918 / 0.974 | 0.876 / 0.956 | 0.996 | 0.995 |

Synthetic pairs gave no clear gain (+0.007 on `start10` top-10, −0.002 on `anywhere` top-10, within seed noise). They also make the Essen distractors look like the training data. So the headline config stays HumTrans-only, and synthetic pairs remain an option.

**Trade-off / what we gave up.**
- **Not the official MIREX number.** The 2,000 distractors are a documented stand-in, not MIREX's list, and MIREX's collection has about 2,600 songs where ours has 2,048. The comparison with CHAD's 0.921 top-10 is close, not exact. Our closest protocol is `start_multi` (0.974), which assumes queries start at the song start, as MIR-QBSH queries do. `anywhere` (0.958) makes no such assumption.
- **No checkpoint selection.** We report `last.pt` because the validation set is saturated. A harder validation set (HumTrans val plus distractors) would be needed to select checkpoints or tune hyperparameters again.
- **Synthetic data unused.** We kept the code but not the data, so the headline model has seen only about 1,000 distinct melodies.

**Revisit when.**
- A harder validation set exists. Then select checkpoints and tune on it, never on MIR-QBSH.
- Queries are no longer clean monophonic hums (noisy phones, whistles, real songs as references). Then test MERT fusion, spectral-peak F0 for whistles (D-007), and a vocal-melody extractor on the reference side.
- The official MIREX distractor list turns up. Then rerun `eval_contour.py` with it.

---

## D-013 · Whistle queries: spectral-peak F0 into the contour encoder, measured on MLEnd without melody references
**Date:** 2026-09-27

**Context.** D-011 and D-012 list whistles as untested. D-007 found that RMVPE loses whistles, because they sit above its range (about 2 kHz and up), and that a spectral peak with a tonal-ratio gate tracks them. Whistles sit about +36 semitones above hums, and the encoder's median subtraction should absorb that offset. We needed whistle queries with something to match against:
- **MLEnd Hums and Whistles:** 8 songs, 4,804 hums and 1,797 whistles from about 200 people, but no reference melodies or song audio.
- **MTG-QBH:** 118 sung and hummed queries only. The song audio is not distributed.
- **CHAD:** hums only. Its originals are YouTube IDs.
- **MIR-QBSH and HumTrans:** hums and singing only.
We found no public whistle set with matching references.

**Options.**
- **Queries against MIDI we transcribe ourselves.** We would write the 8 MLEnd melodies by hand. That is subjective, and we might fit the transcription to the data.
- **Query-by-example on MLEnd.** Each song's reference is built from other people's clips. The query's own performer is always left out.
- **Synthesized whistles from HumTrans MIDI.** These test the F0 tracker, not real whistling.

**Decision.**
- **Protocol: query-by-example on MLEnd** (`contour/example_eval.py`, `scripts/eval_mlend.py`). It is an honest test built only from existing material, and it is labelled as a closed 8-way test, not QbH against melodies. Chance top-1 is 0.125.
- **Song scores.** A song's score is cosine to the mean embedding of its reference clips (`centroid`, primary) or the best single clip (`max`). We report both. No thresholds or settings were tuned on MLEnd. The model is the D-012 headline encoder (`last.pt`, 3 seeds), unchanged.
- **Whistle F0 trackers** (`contour/trackers.py`, `contour/whistle.py`, `extract_f0.py --method`):
  - `peak`: D-007's spectral peak, 200 to 6000 Hz, 6 dB tonal gate, 32 kHz.
  - `rmvpe_half`: RMVPE on audio played at half speed with F0 doubled, D-007's "RMVPE-fix".
  - `rmvpe`: plain RMVPE, shown for contrast.
  - Contours get the same cleanup as hums. `rmvpe_contour` now takes the upper F0 limit as a parameter.

**Results.** MLEnd, all clips, mean of seeds 0 to 2 (sd ≤ 0.015). JSON files are `docs/paper/results/contour/mlend_whistle_hum_s*.json`.

| Query → references (centroid) | top-1 | top-3 | MRR |
|---|---|---|---|
| hum (RMVPE) → others' hums | 0.873 | 0.955 | 0.918 |
| hum → others' whistles (peak) | 0.858 | 0.948 | 0.909 |
| **whistle (peak) → others' hums** | **0.625** | **0.805** | **0.740** |
| whistle (peak) → others' whistles | 0.652 | 0.826 | 0.761 |
| whistle (RMVPE-half) → others' hums | 0.570 | 0.765 | 0.699 |
| whistle (plain RMVPE) → others' hums | 0.156 | 0.424 | 0.372 |

With `max` song scores the numbers are within 0.03 of these (e.g. hum → hums 0.895, whistle (peak) → hums 0.642).

Diagnostics:
- **Short voiced tracks.** Clips with under 1 s voiced: 2 hums, 33 whistles (peak), 94 (RMVPE-half), 425 (plain RMVPE).
- **Pitch offset.** Median whistle pitch is 88.9 vs 54.9 for hums, +34 semitones, close to D-007's +36. Median subtraction removes it.
- **Compressed range.** The 5 to 95 % pitch range of a whistle is 6.5 semitones against 9.7 for a hum of the same songs. People whistle a compressed or octave-folded version of the tune. The encoder was trained on hums and has never seen that.

**Trade-off / what we gave up.**
- **Not QbH against melodies.** References are other people's performances, and the task is 8-way. The numbers compare query types with each other. They are not comparable to MIR-QBSH top-10.
- **Whistles are much weaker than hums.** Top-1 drops by 0.25 (0.873 → 0.625) on the same songs, same people and same model. Spectral peak beats RMVPE-half by 0.055 and plain RMVPE by 0.47, which confirms D-007. Part of the gap is the tracker, part is how people whistle (compressed range, more dropouts).
- **No whistle training data.** The encoder has never seen a whistle contour. Training on whistles (e.g. MLEnd with a people-disjoint split) would help, but it would use up the only whistle test set.

**Revisit when.**
- A whistle set with melody references appears, or CHAD-style originals become usable. Then rerun as real QbH.
- We add whistle-like augmentation (interval compression, octave folding, higher dropout) to training. Measure it on MLEnd without changing this protocol.
- We need whistle training data. Then split MLEnd by performer first and keep a fixed test half.

---

## D-014 · References from recorded songs: htdemucs vocals → RMVPE, measured against annotated melody
**Date:** 2026-09-27

**Context.** Every result so far uses MIDI references. A real catalogue has recordings, so the reference contour has to come from a melody extractor, and D-011 and D-012 expected accuracy to drop. We needed songs with both audio and ground-truth melody.
- **MIR-1K:** 1,000 karaoke clips from 110 songs, with vocals and accompaniment on separate channels and manual vocal pitch every 20 ms. mirlab.org did not respond, so we used a public mirror, `AnhP/Mir-1k-use-DJCM-training` on the Hugging Face Hub (download only).
- **ADC2004 and MIREX05** (LabROSA): 20 and 13 excerpts, not used to train any model here.
- **Not available:** MedleyDB needs an access request. HumTrans and MIR-QBSH have no song audio, and CHAD's originals are YouTube links.

Recent melody-extraction work reports OA 91.6 / 92.5 / 78.9 on ADC2004 / MIREX05 / MedleyDB (joint network, Jing et al., Interspeech 2025), next to SpectMamba, MTANet, TONet and FTANet. We found none with maintained public PyTorch weights we could run as-is. RMVPE is itself a vocal-pitch extractor built for polyphonic music.

**Options** (all in `contour/melody.py`):
- `rmvpe_mix`: RMVPE on the mixture.
- `rmvpe_vocals`: htdemucs vocal stem → RMVPE.
- `fcpe_vocals`: htdemucs vocals → FCPE (torchfcpe).
- `crepe_vocals`: htdemucs vocals → CREPE full (torchcrepe).
- `melodia_mix`: Melodia (essentia), the classic dedicated melody extractor.

Every method uses its library defaults and our existing RMVPE voicing threshold (0.3). Nothing was tuned on these sets.

**Decision.** For vocal songs, reference contours come from **htdemucs vocals → RMVPE** (`extract_song_melody.py`). RMVPE on the mixture is the fallback: it is 10× cheaper and better when the melody is instrumental.

The retrieval test (`contour/song_eval.py`, `eval_song_melody.py`) runs on MIR-1K. It ranks against all 1,000 clips and reports two targets:
- **`clip`:** the exact clip.
- **`song`:** any clip of the same song counts. This is the QbH question, because clips of one song repeat melodies.

Queries are cropped to 50 to 80 % of the voiced span, with a fixed seed. There are two query types:
- **Sung:** RMVPE on the isolated singing channel, transposed by 2 to 5 semitones and stretched 0.85 to 1.15×. It is the same performance as the reference, so the absolute level is optimistic.
- **Synthetic:** the annotation, humanized and augmented like training queries. It never touches the audio.

The model is the D-012 encoder (`last.pt`, 3 seeds), unchanged.

**Results.**

Frame accuracy (mir_eval, clip mean; unvoiced frames carry a pitch guess where the tracker gives one). Values are OA / RPA / VFA.

| Method | MIR-1K (1,000) | ADC2004 vocal (12) | ADC2004 all (20) | MIREX05 all (13) |
|---|---|---|---|---|
| clean vocal channel → RMVPE (ceiling) | 0.959 / 0.975 / 0.071 | | | |
| **htdemucs → RMVPE** | **0.927 / 0.953 / 0.108** | **0.924 / 0.944 / 0.086** | 0.632 / 0.662 / 0.052 | 0.756 / 0.669 / 0.066 |
| RMVPE on mixture | 0.904 / 0.945 / 0.108 | 0.847 / 0.904 / 0.238 | **0.706 / 0.825** / 0.159 | **0.777 / 0.779** / 0.123 |
| htdemucs → FCPE | 0.849 / 0.944 / 0.399 | 0.879 / 0.940 / 0.433 | 0.585 / 0.570 / 0.260 | 0.716 / 0.647 / 0.192 |
| htdemucs → CREPE | 0.819 / 0.944 / 0.458 | 0.853 / 0.934 / 0.502 | 0.703 / 0.802 / 0.383 | 0.677 / 0.727 / 0.363 |
| Melodia on mixture | 0.721 / 0.727 / 0.275 | 0.743 / 0.724 / 0.135 | 0.694 / 0.672 / 0.131 | 0.701 / 0.705 / 0.283 |

Retrieval on MIR-1K, `song` target, mean of 3 seeds (sd ≤ 0.02). Values are top-1 / top-10. `clip` numbers are in `docs/paper/results/contour/song_melody.json`.

| References | Sung queries | Drop vs annotation (top-10) | Synthetic queries | Drop vs annotation (top-10) |
|---|---|---|---|---|
| annotation (MIDI-like) | 0.673 / 0.924 | | 0.617 / 0.892 | |
| clean vocal channel → RMVPE | 0.678 / 0.929 | +0.005 | 0.575 / 0.871 | −0.021 |
| **htdemucs → RMVPE** | **0.626 / 0.891** | **−0.033** | **0.539 / 0.832** | **−0.060** |
| RMVPE on mixture | 0.584 / 0.878 | −0.046 | 0.514 / 0.820 | −0.072 |
| htdemucs → FCPE | 0.536 / 0.837 | −0.087 | 0.444 / 0.776 | −0.116 |
| htdemucs → CREPE | 0.453 / 0.777 | −0.147 | 0.374 / 0.723 | −0.169 |
| Melodia on mixture | 0.317 / 0.644 | −0.280 | 0.290 / 0.598 | −0.294 |

On the same queries and model, audio references cost about 0.03 to 0.06 top-10 and 0.05 to 0.08 top-1 against annotated melody when the extractor is htdemucs → RMVPE. Plain RMVPE on the mixture costs about 0.01 more.

Absolute levels are low because MIR-1K clips are 4 to 13 s, so queries are 2 to 10 s, ranked against 1,000 clips. They are not comparable to MIR-QBSH.

GPU cost on the A100 for MIR-1K's 133 minutes of audio, clip by clip: htdemucs 359 s (about 22× real time), RMVPE 32 s, CREPE 230 s, FCPE 5 s.

**Trade-off / what we gave up.**
- **Separation helps only when a voice carries the melody.** On ADC2004 and MIREX05 with instrumental melodies (jazz sax, MIDI), vocals → RMVPE loses to RMVPE on the mixture (OA 0.632 vs 0.706 on all of ADC2004). A catalogue with instrumentals needs a per-song choice. One candidate rule: use the vocal stem when it has enough energy and voiced frames, otherwise the mixture. It is not implemented yet.
- **Possible MIR-1K contamination.** RMVPE's authors trained on MIR-1K splits, and the RVC weights we use may include it, so RMVPE's MIR-1K numbers are likely optimistic. ADC2004 vocal (OA 0.924, close to the joint network's published 91.6 on its ADC2004 protocol) and MIREX05 are uncontaminated checks with the same ordering among RMVPE variants for vocal material. The published figure is not measured under our exact protocol.
- **Voicing not tuned for other trackers.** CREPE and FCPE get library-default voicing on separated vocals, which still hold accompaniment bleed, and their false alarms are high (VFA 0.40 to 0.50). Tuning would narrow the gap but needs a validation split we do not have.
- **Sung queries share the reference performance.** That flatters the absolute level. The synthetic queries do not, and they give the same ordering.
- **htdemucs is the costly step.** For a large catalogue it is about 20× real time per clip without batching.

**Revisit when.**
- A catalogue with real recordings is built. Then add the vocal-or-mixture rule and batch htdemucs.
- CHAD originals are allowed (they need YouTube downloads and the user's approval). Then measure real hums against audio references end to end.
- A dedicated extractor with usable weights appears (the joint network or SpectMamba). Then add it as one more method in `melody.py`.
- We train on audio-derived references, e.g. MIR-1K annotation vs extracted contour pairs, to close the gap. Keep ADC2004 and MIREX05 out of training.

---

## D-015 · Whistle training: MLEnd train-performer whistle pairs plus whistle-like hum augmentation
**Date:** 2026-09-27

**Context.** In D-013, whistles scored 0.25 top-1 below hums on MLEnd (0.625 vs 0.873), with a model that had never seen a whistle. D-013 found two reasons:
- **Compressed range.** Whistles span about 6.5 semitones where hums of the same songs span 9.7.
- **More dropouts.** Whistles lose more frames to the tonal gate.

The only whistle data is MLEnd: 8 songs, 226 performers. Training on it without burning the test set needs a performer split.

**Options.**
- **Fine-tune or retrain.** Fine-tune the D-012 model, or retrain from scratch with the new data.
- **What whistle data to use:**
  - whistle-like augmentation of HumTrans hums only (no MLEnd in training);
  - MLEnd whistle pairs only;
  - both.
- **Checkpoint choice:** HumTrans val alone (saturated, D-012), or a mean that includes MLEnd val people.

**Decision.**
- **Fixed performer split** (`contour/mlend.py`, `scripts/make_mlend_split.py`, committed as `training/splits/mlend_performers.json` in `83a9ec9` before any training). Seed 20260927, 60/20/20 over sorted performer ids. The script refuses to overwrite the manifest.

  | Split | People | Hums | Whistles |
  |---|---|---|---|
  | train | 136 | 2,812 | 1,148 |
  | val | 45 | 1,020 | 310 |
  | test | 45 | 972 | 339 |

  Only 26 of the 45 test people whistled.
- **Retrain from scratch** with `configs/train_contour_whistle.yaml`: the D-012 headline config plus:
  - **MLEnd pairs.** A train-split whistle (spectral-peak F0, D-007), cropped to 60 to 100 % and given the usual augmentation, is paired with a whole hum of the same song by another train-split person. Each whistle appears twice per epoch, 2,296 of 15,376 pairs. The loss masks same-song clips instead of using them as negatives.
  - **Whistle-like augmentation** (`augment.WhistleAugment`). With probability 0.3, a HumTrans hum query's intervals are scaled by 0.55 to 0.85 and it gets extra gaps up to 0.6 s. These values were set before any MLEnd result.
  - **Checkpoint choice.** The mean top-1 of HumTrans val (+7 st), MLEnd val-people whistles → hums and MLEnd val-people hums → hums. MIR-QBSH and MLEnd test people are never used. `select_metric` now takes a comma list.
- **Eval.** `eval_mlend.py --split test` runs the D-013 protocol (centroid, the query's performer excluded) with queries and references restricted to the 45 test people. MIR-QBSH and HumTrans use `eval_contour.py` as in D-012.
- **W&B runs:** 3 seeds `llj5eq5r`, `2obdri9m`, `wf9o1t5d`; ablations (seed 0) augmentation-only `y49hl1f0` and pairs-only `jhmg6sx4`. Two first attempts run in parallel (`4fnvy3wp`, `fpdvbu1t`) ran out of GPU memory and have no results. One run takes about 35 GB.

**Results.** Top-1, mean ± sd over 3 seeds. The whistle model is the val-selected checkpoint (steps 4,000 / 6,000 / 6,000). `last.pt` is within 0.005 everywhere. Files: `docs/paper/results/contour/d015/` (`summary.json`, `summarize.py`, per-run JSON).

MLEnd, test people only (8-way, chance 0.125):

| Query → references (centroid) | Old model (D-012) | **Whistle model** | Aug only (s0) | Pairs only (s0) |
|---|---|---|---|---|
| hum → hums | 0.859±0.007 | **0.939±0.003** | 0.856 | 0.943 |
| **whistle (peak) → hums** | 0.540±0.010 | **0.740±0.006** | 0.566 | 0.729 |
| whistle (peak) → whistles | 0.559±0.003 | **0.739±0.005** | 0.611 | 0.735 |
| hum → whistles | 0.829±0.010 | **0.940±0.003** | 0.829 | 0.943 |
| whistle (RMVPE-half) → hums | 0.511±0.011 | **0.742±0.010** | 0.549 | 0.717 |

Regression check on data with no whistles (top-1 / top-10):

| | Old model (D-012) | Whistle model | Aug only (s0) | Pairs only (s0) |
|---|---|---|---|---|
| MIR-48 `anywhere` | 0.959 / 0.993 | 0.954 / 0.994 | 0.959 / 0.994 | 0.961 / 0.993 |
| +2000 `start_multi` | 0.922 / 0.974 | 0.908±0.007 / 0.969±0.001 | 0.934 / 0.975 | 0.922 / 0.975 |
| **+2000 `anywhere`** | **0.866±0.007 / 0.958±0.002** | **0.842±0.011 / 0.949±0.004** | 0.879 / 0.959 | 0.863 / 0.959 |
| HumTrans test top-1 (+7 st) | 0.996 (0.993) | 0.992 (0.988±0.005) | 0.995 (0.993) | 0.995 (0.992) |

Selection scores on val people (whistle / hum top-1): full model 0.913 / 0.959, 0.906 / 0.961, 0.910 / 0.966; pairs-only 0.923 / 0.963; augmentation-only 0.803 / 0.889.

**Takeaways.**
- **Whistles on the test people: 0.540 → 0.740 top-1** (+0.20, all 3 seeds). The gap to hums on the same people narrows from 0.32 to 0.20. RMVPE-half whistles now match spectral-peak ones (0.742).
- **MLEnd hums also improve** (0.859 → 0.939). The model now sees MLEnd recording conditions and these 8 songs.
- **The pairs carry the gain.** Pairs-only reaches 0.729 whistle top-1 on one seed. Augmentation-only adds just +0.026 whistle top-1: squeezed hums alone do not teach the model what whistles look like.
- **MIR-QBSH regresses slightly with the full config.** With 2,000 distractors, `anywhere` drops 0.024 top-1 (0.866 → 0.842) and 0.009 top-10 (0.958 → 0.949). `start_multi` drops 0.014 / 0.005. HumTrans test drops about 0.004. The single-seed ablations show no regression for either part alone (pairs-only 0.863 / 0.959, augmentation-only 0.879 / 0.959). So the cost may come from combining the two, or from seed noise. One seed cannot tell.

**Trade-off / what we gave up.**
- **Test people are new, the songs are not.** All 8 songs appear in training through train-split people, so the MLEnd test gain is partly song familiarity, not only whistle robustness. Whistles on songs never seen in training remain untested. That needs a leave-songs-out split, e.g. train pairs on 6 songs and test whistles of test people on the other 2.
- **A small cost on hums without whistles.** For MIR-QBSH-style use the D-012 model is still slightly better (+2000 `anywhere` top-1 0.866 vs 0.842). We keep both checkpoints and report both.
- **Val and test people differ a lot.** Whistle top-1 is about 0.91 on val people and 0.74 on test people for the same models. With 26 whistling test people, performer variance is large. Selection only chose between steps (4,000 vs 6,000), and `last.pt` gives the same numbers, so this gap is not selection bias.
- **Ablations are single-seed.**

**Revisit when.**
- **Pairs-only confirmation.** Run two more seeds of pairs-only. If val confirms it is at least as good as the full config (it is ahead on val on seed 0: 0.960 vs 0.955 mean selection score), switch to it. That choice uses val only; its MIR-QBSH numbers stay a check, not the reason.
- **Leave-songs-out.** Run the leave-songs-out whistle test before claiming whistle QbH on unseen songs.
- **More whistle data.** A whistle set with more songs or melody references appears. Then train on it and use MLEnd test people only for evaluation.

---

## D-016 · Whistle training does not transfer to unseen songs; keep D-012 as the default model
**Date:** 2026-09-27

**Context.** D-015 raised whistle top-1 on MLEnd test people from 0.540 to 0.740. However, all 8 songs were in training through train-split people, so the gain could be song familiarity rather than whistle robustness. D-015 also left open whether pairs-only should replace the combined config, and it showed a small MIR-QBSH dip against D-012.

**Options.**
- Adopt D-015 (combined) as the default model.
- Adopt pairs-only as the default model.
- Keep D-012 as the default model.

**Decision.**
- **Unseen-song test** (`make_mlend_song_holdout.py`, `training/splits/mlend_heldout_songs.json`, commit `7241f20`, fixed before training). Seed 20260928 held out **Hakuna** and **Potter**.
  - **Training.** `song_holdout=true` removes them from whistle-pair training (1,714 pairs left of 2,296) and from the MLEnd val score used for checkpoint selection.
  - **HumTrans overlap.** HumTrans has no titles, so overlap was checked by melody. MLEnd hums were matched to their nearest HumTrans segment with the D-012 model. The largest share for any one HumTrans song was 0.14 (Hakuna) and 0.10 (Potter), under the 0.25 threshold, so no HumTrans song was excluded (`d016/overlap_check.json`). Two training songs, Frozen (0.26) and Showman (0.27), are just over the threshold; this does not affect the test.
  - **Runs.** The combined D-015 config, 3 seeds: W&B `awvchwtn`, `69hkhs0u`, `6mbr6c2b`.
  - **Evaluation.** `eval_mlend.py --split test --song-holdout`. Held-out-song queries from test people are still ranked against all 8 songs.
- **Pairs-only seeds 1 and 2** (`na6niugm`, `9w366cik`). With D-015's seed 0 this gives 3 seeds. The config was compared on val only.
- **Outcome.** **D-012 stays the default model for search and the app.** D-015 and pairs-only are not adopted. Whistle robustness needs whistle data that covers many songs.

**Results.** Top-1, mean ± sd over 3 seeds, test people, val-selected checkpoints. Files: `docs/paper/results/contour/d016/` (`summary.json`, `summarize.py`, per-run JSON).

Held-out songs (Hakuna, Potter) vs seen songs:

| Model | Unseen: whistle → hums | Unseen: hum → hums | Unseen: whistle → whistles | Seen: whistle → hums | Seen: hum → hums |
|---|---|---|---|---|---|
| D-012 (no MLEnd in training) | **0.630±0.010** | **0.859±0.008** | **0.581±0.014** | 0.507±0.010 | 0.859±0.011 |
| D-015, trained on all 8 songs (these songs seen) | 0.722±0.018 | 0.929±0.007 | 0.715±0.019 | 0.747±0.003 | 0.942±0.005 |
| **Whistle training, 2 songs held out** | **0.456±0.045** | **0.781±0.014** | 0.548±0.023 | 0.754±0.015 | 0.950±0.002 |

Pairs-only vs combined (3 seeds each):

| | +2000 `anywhere` top-1 / top-10 | +2000 `start_multi` top-10 | HumTrans test top-1 (+7 st) | MLEnd test whistle → hums | Val selection score |
|---|---|---|---|---|---|
| D-012 | **0.866±0.007 / 0.958±0.002** | **0.974** | **0.996 (0.993)** | 0.540 | – |
| Combined (D-015) | 0.842±0.011 / 0.949±0.004 | 0.969 | 0.992 (0.988) | 0.740±0.006 | 0.955±0.001 |
| Pairs only | 0.836±0.027 / 0.952±0.007 | 0.971 | 0.992 (0.989) | 0.730±0.016 | 0.958±0.002 |

**Takeaways.**
- **The D-015 whistle gain is mostly song familiarity.** Once the test songs are left out of whistle training, whistle → hums on those songs drops to 0.456±0.045. That is below D-012 (0.630), which never saw MLEnd. Hum → hums on those songs also drops (0.781 vs 0.859). On the songs it did see, the same model scores 0.754 and 0.950. Training on MLEnd's 8 songs teaches those melodies, not whistling in general, and it slightly hurts new songs.
- **Pairs-only and combined are equivalent.** Val is 0.958 vs 0.955, MLEnd test whistles 0.730 vs 0.740, and MIR +2000 `anywhere` top-1 0.836 vs 0.842, all within seed spread. Neither removes the MIR dip against D-012 (0.866).

**Trade-off / what we gave up.**
- **Whistles stay weak.** D-012 gets about 0.54 to 0.63 whistle top-1 on MLEnd test people in this 8-way setting, well below hums (0.86). That is the honest number for a new song.
- **The evidence is narrow.** Only 2 held-out songs and 3 seeds (sd 0.045 on the key number). Another pair of songs could give a different size of drop. Its direction, below D-012, held on all 3 seeds.
- **Code kept, config off.** The whistle-training code (`mlend.py`, `WhistleAugment`, `train_contour_whistle.yaml`) stays. The default checkpoints are D-012's `contour_v2_s*/last.pt`.

**Revisit when.**
- Whistle data covering many songs (tens to hundreds) exists, ideally with melody references. Then retrain with whistle pairs and evaluate on held-out songs from the start.
- A whistle-specific approach that does not memorize melodies is tried, e.g. MIDI-only synthetic whistles across the Essen corpus. Evaluate it with this unseen-song protocol.

## D-017 · First paper song library: FMA full, vocal stem → RMVPE, 10 s chunks in pgvector, minimal search API
**Date:** 2026-09-28

**Context.** Phase 2 needs a song library that the D-012 contour model can search from real audio, under D-009 (open-licence audio only, no YouTube). The first build has to finish in one night on the single A100, stay reproducible, and give an end-to-end search: query audio in, top-k songs out.

**Options.**
- **Audio source:** `fma_large` (30 s clips) vs `fma_full` (full-length tracks, 943.6 GB zip).
- **Download:** the whole archive vs ranged HTTP reads of selected zip members.
- **Melody:** RMVPE on the mix vs htdemucs vocals → RMVPE (D-014).
- **Chunking:** whole-song embeddings vs fixed windows.
- **Store:** numpy/FAISS files vs Postgres with pgvector.

**Decision.**
- **Source: `fma_full`.** Hums can target any part of a song, not a 30 s excerpt. The zip supports HTTP ranges, so `remotezip` (`catalog/fma.py`, `scripts/fetch_fma.py`) reads only the chosen members: 3,000 tracks in about 5 minutes with 24 threads, with no full download.
- **Selection (fixed before indexing):** seed 20260928 draws 3,000 tracks from vocal-leaning top genres (Rock, Pop, Folk, Hip-Hop, International, Country, Soul-RnB, Blues) lasting 60–420 s. Result: Rock 1729, Hip-Hop 438, Folk 347, Pop 262, International 147, Country 28, Soul-RnB 26, Blues 23. Rows keep title, artist, genre and licence (`library/fma_full_3k/songs.jsonl`).
- **Melody: htdemucs vocals → RMVPE, as in D-014.** Two engineering changes that do not change the method:
  - ffmpeg decodes each mp3 straight to 44.1 kHz mono, so htdemucs gets the full band and a few mp3s libsndfile cannot read still decode. The stem goes to 16 kHz through torchaudio's anti-aliased resampler.
  - The htdemucs segments of a song run in batches of 16 (`separate_vocals_batched`). It uses the same segment length, 25% overlap and triangular weights as demucs `apply_model`. On two songs its max relative difference from `apply_model` is ≤ 2e-3 and RMVPE voicing agrees on 100% of frames. It is about 3× faster.
  - RMVPE tracks are cached as `.npy` (`library/fma_full_3k/tracks/`), so re-chunking or a new checkpoint does not repeat separation.
- **Chunks: 10 s windows with a 5 s hop** on the 20 ms contour grid. A window is kept only if at least 25% of its frames are voiced; instrumental stretches have an empty vocal stem and nothing to match. Songs with no voiced window stay in `songs` with `chunk_count = 0`, so reruns skip them, but they are not searchable.
- **Embeddings: D-012 `contour_v2_s0/last.pt`** (the default model, D-016), 256-d, L2-normalized.
- **Store: Postgres 16 + pgvector** (`sql/001_library.sql`). Table `songs` holds metadata, source tier, licence, `model_ver` and `chunk_count`. Table `chunks` holds `song_id`, `start_s`, the voiced fraction and `vector(256)`, with an HNSW cosine index. **Song score = its best chunk's cosine similarity** among the 400 nearest chunks (`ef_search` 400).
- **API** (`hum2song.server.api`, FastAPI):
  - `POST /search` takes an audio file (≤ 10 MB) and returns the top-k songs with score and best chunk start. It uses the same query path as every contour eval: RMVPE → cleaned contour → D-012 encoder.
  - Queries with under 1 s of voiced audio return no results.
  - `GET /health` reports counts and the model.
- **Sanity probe** (`scripts/probe_library.py`). For songs chosen with a fixed seed, it renders a hum from each song's own extracted melody: humanized and augmented as in `song_eval`, with a random, mostly voiced 8–12 s crop, as a harmonic tone plus noise. The hum then goes through the full audio search path. It is a proxy, not a person humming, and it tests the pipeline and the chunking, not real-hum accuracy.

**Results (interim, while the build runs).**
- The build ran 3 processes sharing the GPU at about 0.6 songs/s.
- 325 songs indexed (310 searchable, 7,723 chunks). The rendered-hum probe, 200 queries (`docs/paper/results/library/probe_interim.json`): **top-1 0.705, top-5 0.815, top-10 0.87, MRR 0.763.**
- One real hum: MLEnd `0002.wav` (a test-split person humming *Harry Potter*, which is not in FMA) returns FMA songs at cosine 0.46–0.54. That only shows the real-hum path runs end to end; the correct song is not in the library.
- A first try that cropped 50–80% of the whole voiced span (queries capped at 20 s, against 10 s chunks) gave top-1 0.36 on 95 songs. The drop comes from the query/chunk length mismatch.

**Trade-off / what we gave up.**
- **Genre skew:** 58% Rock, as FMA is. It fits a paper subset but not a general catalog.
- **Vocal melody only.** Instrumental hooks (riffs, synth leads) are invisible, because chunks come from the vocal stem.
- **One chunk length.** 10 s chunks suit 8–12 s hums. Longer queries straddle chunks and score lower.
- **No real hums of indexed songs yet**, so there is no real-hum accuracy number for this library.

**Revisit when.**
- Real hums of FMA songs are recorded (the benchmark plan in D-009). Measure real-hum top-k on this library then.
- Queries longer than 12 s become common. Then add 20 s chunks or aggregate several query windows.
- The library grows past about 100k chunks. Then retune HNSW `m`, `ef_construction` and `ef_search` against exact search.
- The contour model changes. Then re-embed from the cached tracks; `model_ver` marks which rows to redo.
