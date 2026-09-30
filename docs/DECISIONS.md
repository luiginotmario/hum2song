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

**Results.** Files: `docs/paper/results/library/` (`probe_full.json` is final, `probe_interim.json` is from 325 songs).
- **Build:** all 3,000 tracks indexed; no decode failures. **2,783 are searchable (70,844 chunks)**; 217 (7.2%) have no voiced 10 s window. It took 80 minutes (02:30–03:51 UTC) with 3 processes sharing the GPU, about 0.6 songs/s. Phase 2 GPU time through the probe was about 1.75 h.
- **Rendered-hum probe on the full library, 500 queries** (`probe_full.json`): **top-1 0.582, top-5 0.716, top-10 0.764, MRR 0.643.**
- **Interim on 325 songs (310 searchable), 200 queries:** top-1 0.705, top-5 0.815, top-10 0.87, MRR 0.763. Going from 310 to 2,783 songs costs 0.12 top-1, so distractor count matters a lot for this proxy.
- One real hum: MLEnd `0002.wav` (a test-split person humming *Harry Potter*, which is not in FMA) returns FMA songs at cosine 0.46–0.54 on 325 songs (0.54–0.58 on the full library). That only shows the real-hum path runs end to end; the correct song is not in the library.
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

## D-018 · Real hums vs real commercial recordings, using 30 s iTunes and Deezer previews
**Date:** 2026-09-28

**Context.** Every accuracy number so far pairs hums with MIDI renders, other hums or synthetic hums. D-017's library has no song anyone in a hum dataset hummed. Luigi asked for commercial and popular songs so the pipeline (htdemucs vocals → RMVPE → D-012) can be tested against real recordings. The hum sets with commercial songs are CHAD (290 groups, 5,417 hums, each with its original YouTube video and the hummed interval in it), MTG-QBH (118 sung queries, 81 songs) and MLEnd (8 movie songs, 4,804 hums, 1,797 whistles). **This goes beyond D-009**, which kept Apple and Deezer previews in Tier L until written permission. Luigi directed this experiment for his personal research. It keeps audio only transiently and stores features plus metadata only.

**Options.**
- **Full tracks from YouTube via yt-dlp.** Luigi allowed this for this project. It was tried and blocked (see Decision).
- **Official 30 s previews:** the iTunes Search API `previewUrl` and Deezer's public API `preview` field. Both need no login.
- **Wait for licensed audio** (D-009 Tier L).

**Decision.** Previews, audio kept transiently. Code: `catalog/previews.py`, `catalog/targets.py`, `catalog/real_eval.py`, `scripts/collect_previews.py`, `extract_previews.py` and `eval_previews.py`.
- **YouTube was not used for audio.** yt-dlp from the Lambda host and from the agent box both returned "Sign in to confirm you're not a bot". Passing a browser session's cookies would work around that check, so it was not done. **No YouTube audio was downloaded.** Only public oEmbed metadata was read: the video titles of the 290 CHAD originals, to learn which song each group is.
- **Targets.**
  - MLEnd: the 8 songs named in the dataset brief, e.g. "This Is Me" (Keala Settle) for Showman and "The Imperial March" for StarWars.
  - MTG-QBH: title from the queries file, artist from the canonical-version collection.
  - CHAD: artist and title parsed from the YouTube title.
  - Each target is searched on iTunes and on Deezer. A result must match the title, blended 70/30 with artist overlap, at ≥ 0.8; title-only targets need ≥ 0.9.
  - Matched: MLEnd 8/8, MTG-QBH 81/81, CHAD 264/290 on either service. 6 CHAD videos had no usable title and 20 had no confident match.
- **Distractors:** 3,000 unique songs from Deezer's chart playlists, deduplicated by artist, title and ISRC. Any song titled like a target is dropped. Apple RSS charts were fetched too, but the 3,000 cap filled from Deezer first. The pool includes a lo-fi playlist and similar, so it is not all vocal pop.
- **Transient audio.** Each preview is downloaded to a temporary file, decoded by ffmpeg to 44.1 kHz mono and deleted. Deezer URLs are re-signed just before download, because they expire within minutes. Kept per song:
  - `vocals`: the htdemucs vocal stem → RMVPE (D-014);
  - `mix`: RMVPE on the whole mix;
  - both as `(2, frames)` F0/confidence tracks in `library/previews_v1/tracks/*.npz`, plus metadata.
  - No preview audio remains on disk. Preview URLs are not stored in the results.
- **Rate limits and terms relied on.**
  - iTunes Search: about 20 calls/minute, documented by Apple as approximate; we used 1 call per 3.1 s, and batched `lookup` for chart ids.
  - Deezer: 50 calls per 5 s; we used 1 per 0.15 s.
  - Apple's affiliate/preview terms say previews are for promotion and streamed, not downloaded or cached. Deezer's terms bar storing audio.
  - We downloaded and decoded each preview and kept only derived features. **That is still a download**, so this is a research-use exception that needs checking before any publication, not a clean licence.
- **Evaluation** (exact cosine search, a song's score is its best chunk as in the API, the D-012 `contour_v2_s0` model):
  - Queries go through the API path.
  - Library settings: `targets_only`, `charts` (+ ~2,400 searchable chart previews), `charts_fma` (+ the 2,783 searchable FMA songs).
  - Target versions: iTunes only, Deezer only, or both.
  - A query whose song has no usable preview counts as a miss in `all_queries`; `target_present` excludes it.
  - The headline setting was fixed before running: `vocals_or_mix` (vocal stem, falling back to the mix if it gives no chunk), iTunes versions, `charts_fma`.

**Results.** Files: `docs/paper/results/previews/real_hum_eval.json` (every setting) and `targets.jsonl` (targets and their matches; no audio, no URLs).

Extraction (3,678 previews):
- **Downloads:** 3,628 decoded. The 50 failures are all distractors: 26 had no preview URL, 22 failed to download or decode, 2 were under 5 s.
- **Voiced chunks:**
  - 64% of previews gave a voiced chunk from the vocal stem.
  - With the mix fallback it is 85%; 548 have none, mostly instrumental chart tracks.
  - Targets: 663 of the 678 target previews are usable: 635 from the vocal stem, 28 from the mix fallback, and 15 (2%) with nothing.
  - For MLEnd, the iTunes Imperial March and the Deezer Pink Panther give no chunk at all.

Real hums vs real recordings (top-1 / top-10 over all queries; library size in songs):

| Query set | Targets only | + charts (~2.7k) | + charts + FMA (~5.4k), **headline** | Both previews, + charts + FMA |
|---|---|---|---|---|
| CHAD hums (5,417) | 0.148 / 0.261 | 0.097 / 0.165 | **0.062 / 0.112** | 0.091 / 0.161 |
| MTG-QBH sung (118) | 0.161 / 0.492 | 0.025 / 0.076 | **0.000 / 0.017** | 0.000 / 0.042 |
| MLEnd hums (4,804) | 0.371 / 0.874 (7-way) | 0.034 / 0.108 | **0.011 / 0.042** | 0.026 / 0.100 |
| MLEnd whistles (1,797, RMVPE tracker) | 0.135 / 0.875 | 0.001 / 0.001 | 0.000 / 0.001 | 0.001 / 0.002 |

- CHAD queries whose song has a usable iTunes preview (4,158): top-1 0.080, top-10 0.146 at the headline setting.
- CHAD by where the hummed fragment starts in the original video (headline setting), top-1: 0–30 s 0.045, 30–60 s 0.102, 60–90 s 0.082, 90 s+ 0.064.
- MLEnd per song, top-1 at the headline setting: Mamma Mia 0.08; all the others ≤ 0.005.
- Model-based hook-coverage estimate: the share of target songs whose median query similarity beats the 99th percentile of wrong-song similarities. CHAD 14%, MTG-QBH 13%, MLEnd 1 of 7.

**Takeaways.**
- **The headline number is low.** Real hums find the real recording's 30 s preview at top-10 11% of the time among about 5.4k songs (CHAD), versus 76% for D-017's rendered hums among 2.8k. There is real signal: CHAD targets-only top-1 is 0.15 in a ~250-way set, where chance is 0.004. But it is far from usable.
- **Preview coverage is a big part of the gap.** Adding the second 30 s preview (Deezer next to iTunes) raises CHAD top-1 from 0.062 to 0.091 (+47%) and MLEnd top-10 from 0.042 to 0.100 with the same distractors. More of each song helps directly. Neither API exposes where the preview starts, so hook coverage could only be estimated from the model, about 13–14% of songs. That estimate mixes coverage with model error.
- **Extraction is not the bottleneck for vocal songs.** 98% of target previews give usable chunks. Instrumental themes are the exception: Imperial March, Pink Panther and Hedwig's Theme are nearly unmatched. The mix fallback makes more songs searchable, but using the mix for every song is worse (MLEnd `mix`: 0.009 top-10 vs `vocals` 0.042 at `charts_fma`).
- **The distractors used matter a lot.** MTG-QBH drops from 0.49 top-10 targets-only to 0.017 with charts + FMA, and MLEnd from 0.87 to 0.04. Numbers from small closed sets (like D-016's 8-way MLEnd) do not predict library-scale retrieval.
- **Whistles fail with RMVPE**, as D-007 predicted. They need the spectral-peak tracker on the query side.
- **Real hum ↔ real recording is a domain gap D-012 was not trained for.** It trained on MIDI-rendered and synthetic contours. D-017's rendered-hum probe (0.58 top-1) overstates real performance by an order of magnitude.

**Trade-off / what we gave up.**
- **Terms:** previews were downloaded transiently against Apple's "streamed only" and Deezer's no-storage wording. This conflicts with D-009, at Luigi's direction. Only features and metadata remain. **These results should not be published before the terms are checked, or permission obtained, and the paper must disclose the source.**
- **Only 30 s per song, at an unknown offset.** Hooks outside the preview are unreachable, and coverage cannot be measured directly.
- **Version mismatch:** search sometimes returns a live or remix version (e.g. iTunes "Let It Go (Live)", Deezer "Alejandro (Dave Aude Remix)"). `targets.jsonl` records the match so this can be audited.
- **CHAD songs are identified from YouTube titles**, so 26 of 290 are missing and some matches may be covers.
- **The distractor pool is chart-heavy and includes instrumental playlists**; genres are not balanced.
- **One seed** of the D-012 model (`s0`), with no retraining.

**Revisit when.**
- Full-length recordings of the target songs are available lawfully (owned audio, licence, permission, or a YouTube route that works without session cookies). Then re-measure with the same queries to separate coverage from model error.
- A model is trained on real hum ↔ real recording pairs, e.g. CHAD hums with separated vocals, under a song-level split that keeps these test songs out.
- Query-side changes: the whistle tracker, or multi-window queries.
- Before any publication, re-check Apple's and Deezer's terms (D-009).

## D-019 · Full-length YouTube recordings: real hums vs full songs, and training on CHAD real pairs
**Date:** 2026-09-28

**Context.** D-018 found that real hums rarely find a song's 30 s preview (CHAD top-1 0.062 among about 5.4k songs). It could not tell how much of that was preview coverage and how much was model error. D-018 also proposed training on real hum ↔ real recording pairs. Both need the full recordings. **This departs from D-009**, which removed yt-dlp from the pipeline. Luigi approved it for this personal research project under strict rules:
- downloads run only on Luigi's Mac, in one folder (`~/hum2song_yt/`), with the standalone yt-dlp binary;
- no installs, no shell config changes, no cookies, no browser session;
- stop if the bot check appears;
- audio is copied off, decoded once and deleted. Only melody features and metadata are kept.

**Options.**
- **Keep previews only** (D-018). No coverage answer, and no training pairs.
- **Licensed full tracks** (D-009 Tier L). No permission yet.
- **Full recordings from YouTube, transient audio, features only.** Chosen, at Luigi's direction.

**Decision.** Full recordings from YouTube, stored as **Tier Y**: features only, never published or redistributed.
- **Download.** yt-dlp 2026.08.19 (`yt-dlp_macos`) with `--no-config --no-cache-dir`, best audio stream only.
  - CHAD and MLEnd use their original video ids; MTG-QBH uses the first search result (`ytsearch1`, artist + title).
  - No bot check appeared. Without a JS runtime (deno), yt-dlp warns, and about 3% of downloads returned HTTP 403; these were skipped, not worked around.
- **Transfer and deletion.** On the Mac: tar, split into < 95 MB parts, copied to the agent box and checked by sha256, then deleted on the Mac. Then uploaded to Lambda local disk.
- **Extraction.** `extract_previews.py` (rows with `audio_path`) decodes each file once, **deletes it**, and caches `vocals` (htdemucs → RMVPE) and `mix` (RMVPE) tracks in `library/youtube_v1/tracks/youtube_<video id>.npz`.
  - RMVPE now runs in ≤ 540 s segments. cuDNN's GRU rejected a > 10 min upload; audio up to 540 s is tracked exactly as before. A model failure is recorded per song instead of stopping the run.
- **Operational incident, disclosed.** The first extraction crashed on that long upload. The ops script then deleted 107 audio files that had not yet been extracted, and a restart lost 7 more. The 130 songs were downloaded again the same way, and the 7 were queued again; they are not in these results.
- **Coverage.**
  - CHAD: 283 of 290 groups (10 videos unavailable on YouTube; 7 lost in the incident, queued again).
  - MTG-QBH: 75 of 81 (6 searches failed).
  - MLEnd: 4 of 8. Hedwig's Theme, Pink Panther, Singin' in the Rain and Let It Go were unavailable by id and are queued as searches.
  - One MTG-QBH hit (`yesterday`) is the same video as a CHAD original; it is kept twice under separate song ids.
- **Fixed CHAD song split** (`training/splits/chad_songs.json`, seed 20260928): 116 train / 29 val / 145 test groups. Test songs are never trained on and never used to pick a checkpoint.
- **Training** (`configs/train_contour_chad.yaml`): the D-012 recipe plus CHAD pairs (`chad_repeat: 3`, fixed before training). A pair is an augmented real hum with the matching window of the full recording's vocal melody at CHAD's hummed interval. CHAD val songs are logged, and the reported checkpoint is **`last.pt`, fixed in advance**. Trained from scratch, 6,000 steps, 3 seeds.
- **Evaluation** (`eval_previews.py`): the same queries, API query path, chunking and scoring as D-018.
  - Target version: iTunes preview or the full YouTube recording.
  - New settings: `full_fma` (full-length chart songs from D-020 + FMA instead of chart previews) and `all_fma` (every distractor).
  - `--chad-split test` keeps CHAD test songs only.
  - The headline setting stays `vocals_or_mix` at `charts_fma`, so D-018 is directly comparable.
- **Library.** Full songs are indexed into pgvector (`index_tracks.py`, `source_tier` Y, `coverage` full): the vocal track, or the mix when the vocals give no chunk.

**Results.** Files are in `docs/paper/results/youtube/`: `full_vs_preview_all.json`, `full_vs_preview_chadtest_v2_s0.json`, `final/*.json` (every model on CHAD test songs, with MIR-QBSH and HumTrans regressions) and `library_manifest.jsonl` (ids and titles; no audio, no URLs).

Full recording vs 30 s preview, D-012 model (`contour_v2_s0`), `charts_fma` (about 5.4k songs). Top-1 / top-10 over all queries:

| Query set | Full song | iTunes preview | Both previews (D-018) |
|---|---|---|---|
| CHAD hums (5,417) | **0.255 / 0.399** | 0.062 / 0.112 | 0.091 / 0.161 |
| CHAD, target present only | 0.290 / 0.453 (4,772 queries) | 0.080 / 0.146 (4,158) | |
| MTG-QBH sung (118) | 0.034 / 0.153 | 0.000 / 0.017 | 0.000 / 0.042 |
| MLEnd hums (4,804; 4 of 8 songs available) | 0.030 / 0.078 | 0.011 / 0.042 | 0.026 / 0.100 |

- Full recordings, targets only (277 songs): CHAD 0.360 / 0.542, against 0.148 / 0.261 for the iTunes preview.
- CHAD by where the hummed fragment starts (full song, test songs): top-1 is 0.25 at 0–30 s, 0.40 at 30–60 s, 0.29 at 60–90 s and 0.29 after 90 s. The later sections that a preview misses are now found about as often as the opening.

Training with CHAD real pairs, CHAD **test** songs (2,482 hums, 145 songs, 140 with a usable recording). Mean of 3 seeds each, `last.pt`, top-1 / top-10:

| Setting | Old model (D-012 v2, s0–s2) | New model (CHAD pairs, s0–s2) |
|---|---|---|
| Full songs, targets only (140) | 0.468 / 0.667 | 0.394 / 0.612 |
| Full songs, `charts_fma` (~5.3k) | **0.303 / 0.470** | **0.239 / 0.402** |
| Full songs, `full_fma` (~3.2k: 283 full-length chart songs + FMA) | 0.302 / 0.466 | 0.236 / 0.399 |
| iTunes previews, `charts_fma` | 0.066 / 0.112 | 0.043 / 0.085 |
| MTG-QBH, full songs, `charts_fma` | 0.040 / 0.136 | 0.040 / 0.090 |
| MLEnd, full songs, `charts_fma` | 0.040 / 0.100 | 0.028 / 0.087 |

- Per seed, new model, `charts_fma` top-1: 0.236, 0.244, 0.236. Old model: 0.300, 0.297, 0.314.
- The val-selected `best.pt` gives the same numbers within 0.004.
- Regressions:
  - MIR-QBSH (+2,000 distractors) top-1: 0.872 old (s0) vs 0.824 / 0.818 / 0.826 new.
  - HumTrans test top-1 (+7 shift): 0.995 old vs 0.991 to 0.995 new.
- CHAD val top-1 during training rose from 0.01 to about 0.32. That is in a 145-song train+val pool, where the old model scores higher on comparable test pools.
- GPU time: about 3.4 h (extraction about 0.6 h, 3 trainings about 1.9 h, evaluations about 0.9 h).

**Takeaways.**
- **Preview coverage was the main gap in D-018.** The same model and queries with the full recording raise CHAD top-1 about 4× (0.062 → 0.255) and top-10 3.6× (0.112 → 0.399) among about 5.4k songs. Hums of later sections, which a 30 s preview rarely contains, are found as often as the opening. Full-song indexing is needed for real use.
- **Model error is still most of what remains.** Even with the right recording and only ~140–280 candidates, top-1 is 0.36–0.47. Real hum ↔ real vocal melody is a harder match than D-012's MIDI and synthetic training pairs.
- **Naively adding CHAD pairs made things worse**, on every seed and every setting: −6 points top-1 on CHAD test songs, and −5 on MIR-QBSH. This is a negative result, and it is reported as run, with `last.pt` fixed in advance. Likely causes, not yet tested:
  - only 116 training songs, each repeated 3× per epoch, so the model can memorise song identity rather than learn the hum ↔ melody mapping;
  - label noise: CHAD intervals refer to the original video, and vocal-stem melody includes backing vocals and ad-libs;
  - training from scratch, instead of fine-tuning the D-012 model with a low learning rate.
- **The distractor source barely matters at this scale.** Swapping ~2.4k chart previews for 283 full-length chart songs + FMA gives almost the same CHAD numbers. A bigger full-song library is being built (D-020).

**Trade-off / what we gave up.**
- **Terms.** YouTube's terms forbid downloading. This is a documented research-use exception to D-009 at Luigi's direction: audio was transient, and only features and metadata remain. **Results must not be published before the terms are reviewed, and any paper must disclose the YouTube source.** Tier Y songs stay out of any public index.
- **Coverage gaps:** 7 CHAD, 6 MTG-QBH and 4 MLEnd songs are missing. The MLEnd row is therefore weak.
- **Version noise:** MTG-QBH recordings come from the first search result and may be remasters or live versions. The manifest records each video title.
- **One training recipe**, with repeat 3 and training from scratch. No hyperparameters were tuned against test songs.

**Revisit when.**
- A fine-tuning variant (start from D-012, low LR, lower CHAD weight) is ready. It must use the same split and the same fixed-in-advance checkpoint rule.
- The re-queued CHAD and MLEnd songs are extracted. Then re-run the full-vs-preview table.
- D-020's larger full-song library is ready. Then re-measure `full_fma`.
- Before any publication, review YouTube's terms and D-009.

## D-020 · Scaling the served library with full-length chart songs (Tier Y), and how rendered hums fare by genre and track
**Date:** 2026-09-28

**Context.** D-019 showed that full recordings, not 30 s previews, are what real hums can find. The served library (pgvector, D-017) held 3,000 FMA songs, mostly obscure and license-clean. A realistic hum-to-song service needs the songs people actually hum: chart hits. Luigi approved growing the library from YouTube under the D-019 Tier Y rules:
- Mac-only downloads in `~/hum2song_yt/` with the standalone yt-dlp;
- no cookies or installs; stop on a bot check;
- audio is transient, and only melody features and metadata are kept.

He also asked that house/instrumental songs be measured separately from vocal songs.

**Options.**
- **Chart previews** (D-018). Too short (D-019).
- **Licensed catalog** (Tier L). No permission.
- **Full-length chart songs from YouTube, features only.** Chosen.

**Decision.**
- **Queue** (`/workspace/yt/queue.tsv` on the agent box; tags in `tag_meta.json`): 13,692 tagged chart entries. After deduplication, 11,291 searches are queued, plus the first 300-song batch that ran before the queue existed. Searches use `ytsearch1` with "artist - title audio". Sources:
  - Billboard Hot 100 year-end lists 1960–2025 (6,259);
  - Deezer charts (2,783);
  - Billboard dance (650);
  - Spotify daily top 200 for 9 markets;
  - a house/electronic list (1,000).
- **Genre tags:** `pop_chart`, `chart`, `dance` and `house_electronic`. Songs whose artist/title key or video id matches any evaluation target (CHAD, MTG-QBH, MLEnd) are dropped before extraction.
- **Batches** of about 300 run through the D-019 transfer path. Extraction goes into `library/youtube_charts_v1` (role `distractor`, source `youtube_full`), then `index_tracks.py`.
  - The index uses the vocal track, or the full mix when the vocal stem gives no voiced chunk (instrumental and house songs).
  - About 3–10% of downloads fail with HTTP 403 (no JS runtime) and are skipped. No bot check so far.
- **Measurement** (`probe_library.py`, extended): the rendered-hum probe of D-017 on the served library.
  - Every probed song's indexed track is humanized and augmented, cut to 8–12 s, rendered as a harmonic tone and searched through the API path.
  - Results are broken down by genre tag and by the track the index used (vocals vs mix fallback).
  - This is a **self-retrieval probe** (the query comes from the indexed melody), so it measures the index and the model's tolerance to hum-like distortion, not real-hum accuracy (that is D-019/D-023/D-025).

**Results.** Files: `docs/paper/results/d020/`. Library at measurement time:
- 3,859 songs, of which 3,639 are searchable, with 101,549 chunks;
- 3,000 FMA, 363 `youtube_v1` evaluation targets and 496 `youtube_charts_v1` chart songs;
- a further batch of ~300 is downloading.

Top-1 / top-10:

| Probe | Songs | Result |
|---|---|---|
| All chart songs | 493 | 0.588 / 0.779 |
| · `pop_chart` | 387 | 0.568 / 0.780 |
| · `chart` (Deezer) | 51 | 0.569 / 0.725 |
| · `dance` | 55 | 0.745 / 0.818 |
| · indexed from the vocal stem | 474 | 0.595 / 0.791 |
| · indexed from the mix (no usable vocals) | 19 | 0.421 / 0.474 |
| FMA songs (500 of 3,000) in the same library | 500 | 0.546 / 0.750 |

For reference, the FMA probe on the FMA-only library (2,783 searchable, D-017) gave 0.582 / 0.764.

**Takeaways.**
- **Chart songs are as findable as FMA songs.** Growing the library by 30% with full-length hits costs FMA songs about 3 points top-1 on the probe, as expected from more distractors.
- **Songs without a usable vocal stem are the weak spot:** 0.42 top-1 against 0.60, on a small sample (19). Their mix-track melody mixes instruments.
  - Only 4% of chart songs fell back to the mix so far.
  - The dance songs downloaded so far have vocals and probe well.
  - The 1,000 house/electronic songs in the queue have not been downloaded yet, so the house/instrumental measurement is still pending. The breakdown is in place and will be re-run when they arrive.
- The real-hum effect of the larger library is measured with the D-025 first stage, not with this probe.

**Trade-off / what we gave up.**
- Tier Y data cannot be published. Only ids, titles and features are kept, and the paper must disclose the source.
- The probe is optimistic by construction (self-retrieval).

**Revisit when.** The house/electronic batches are in: re-run the probe by genre and track. Also when the library passes about 10k songs: re-measure real hums at `full_fma`.

## D-021 · Fine-tuning D-012 on CHAD real pairs (1 seed): no clear gain
**Date:** 2026-09-28

**Context.** D-019 trained from scratch with CHAD pairs and lost to D-012 by 6 points top-1, probably by memorising the 116 training songs. Fine-tuning the D-012 model instead is the minimal test of whether real hum pairs help at all (literature review, `docs/research/lit_review_real_audio.md` §9).

**Options.**
- Drop real pairs entirely.
- Fine-tune D-012 gently on CHAD pairs.
- Scale song diversity first (E2).

**Decision.** One gentle fine-tune, 1 seed.
- **Config:** `configs/train_contour_chad_ft.yaml`, with new `init_ckpt` and `early_stop_patience` options in `train.py`. All values were fixed before training:
  - start from `contour_v2_s0/last.pt`;
  - CHAD train pairs once per epoch, mixed with HumTrans's ~13k pairs;
  - LR 3e-5 (10× lower), warmup 100, at most 2,000 steps;
  - stronger query augmentation: stretch 0.5–1.9, warp 0.3, interval scale 0.25, drift 1.0 st, jitter 0.25 st, dropout 0.6, octave errors 0.15;
  - validate every 100 steps and stop after 5 validations without a better CHAD val top-1.
  - **Reported checkpoint:** `best.pt` (CHAD val only).
- **Plan change:** the plan was to run 3 seeds if seed 0 beat D-012 on CHAD val. Seed 1 was started and then stopped when Luigi paused model experiments pending the literature review, so **this is a single-seed result**.
- **Not run:** the hard-negative variant.

**Results.** Files: `docs/paper/results/youtube/d021/`. CHAD test songs, on the same library snapshot as the old model; top-1 / top-10.
- **CHAD val (29 songs):** D-012 0.394, fine-tuned best 0.408 (step 600; early stop at 1,100).
- **CHAD test:**

| Setting | D-012 s0 | Fine-tuned s0 |
|---|---|---|
| Full songs, targets only (142) | 0.465 / 0.670 | 0.462 / 0.648 |
| Full songs, `charts_fma` (5,342) | 0.303 / 0.469 | **0.313 / 0.468** |
| Full songs, `all_fma` (5,835) | 0.290 / 0.452 | 0.301 / 0.454 |
| iTunes previews, `charts_fma` | 0.069 / 0.114 | 0.070 / 0.117 |

- **Other query sets** (full songs, `charts_fma`):
  - MTG-QBH: 0.034 / 0.153 → 0.051 / 0.169.
  - MLEnd: 0.030 / 0.078 → 0.031 / 0.085.
- **Regression checks:**
  - MIR-QBSH (+2,000 distractors): 0.872 / 0.961 → 0.856 / 0.954 (−1.6 top-1, −0.7 top-10).
  - HumTrans test (+7 shift) top-1: 0.995 → 0.996.
- **GPU time:** about 0.4 h (fine-tune, partial seed 1, evaluations).

**Takeaways.**
- **Fine-tuning avoids D-019's collapse but gains little.** +1.0 top-1 on CHAD test at `charts_fma`, with top-10 flat. That is inside D-012's own seed spread (0.297–0.314), so it is **not evidence of a gain**. MIR-QBSH loses 1.6 top-1.
- **Real hum pairs from 116 songs are not the lever.** This matches the review's reading: song diversity (E2) and matching (E1) are the candidates, and E0 should decide between them.

**Trade-off / what we gave up.** A single seed, so small differences cannot be resolved. The hard-negative idea is left untested.

**Revisit when.** E0 shows that embedding error dominates on well-extracted references, or E2 provides thousands of songs of pairs. Then fine-tune on those, with the same stop rule.

## D-022 · E0: where CHAD real-hum queries fail
**Date:** 2026-09-28

**Context.** D-019 and D-021 left CHAD test top-1 at about 0.30 against full songs. The literature review (`docs/research/lit_review_real_audio.md`, plan E0) asks for an error breakdown before any further experiment: is the loss in the song-side melody extraction, in how queries are chunked and matched, or in the embedding itself?

**Options.**
- Pick the next experiment by intuition.
- Measure first, using CHAD's timestamps of the hummed section in each song.

**Decision.** Measure first, with no training. Script `training/scripts/chad_error_breakdown.py`, model D-012 `contour_v2_s0`, CHAD test songs, `charts_fma` pool (5,342 songs). 2,411 of 2,482 queries have a CHAD timestamp and are used. For each query:
- **Grid rank:** the normal API score (best 10 s library chunk, 5 s hop).
- **Oracle-window rank:** the target is represented by an embedding of exactly the hummed section (from the timestamp), all other songs unchanged. This measures the cost of chunking.
- **Pitch agreement:** key-normalised DTW error (semitones) between the hum and the song's vocal-stem melody in that section, plain and octave-folded. Plus the voiced fraction of the reference section, and the same on the unseparated mix track.

**Results.** Files: `docs/paper/results/e0/` (summary plus one line per query).
- **Overall:** grid 0.312 / 0.483 top-1 / top-10; oracle window 0.353 / 0.513. **Chunking costs about 4 points top-1.**
- **Decomposition** (share of all queries):

| Outcome | Share |
|---|---|
| Correct at top-1 | 31.2% |
| Missed, reference section < 30% voiced in the vocal stem | 19.4% |
| Missed, fixed by the oracle window | 4.1% |
| Missed, pitch error above median (1.36 st) | 25.6% |
| Missed, pitch error below median (embedding error) | 19.7% |

- **By reference voiced fraction:** < 0.3 (502 queries) top-1 0.068; 0.5–0.7 0.383; > 0.7 0.386 (oracle 0.456).
  - Of those 502, the mix track is voiced for 249, but only 85 have a mix melody close to the hum. So stem separation drop-outs explain at most about a sixth of this bucket. Most of it is a section with no usable sung melody in the recording (instrumental hooks, timestamp offsets or rap).
- **By pitch error:** MAE < 1 st top-1 0.469 (oracle 0.606); 2–3 st 0.246; > 3 st 0.236. **After octave folding**, only 159 queries remain above 2 st (was 429). Most large song-side errors are octave errors in the extracted melody, not wrong notes.
- **By hum length:** 5–8 s 0.215; 8–12 s 0.395; > 12 s 0.232. Long hums lose because a single 10 s chunk cannot hold them.

**Takeaways.**
- **Song-side extraction is the largest single failure** (about 19% unvoiced references plus a large part of the 26% high-error bucket, mostly octave errors). The embedding alone accounts for about 20%.
- **Matching is the cheapest lever.** Chunking costs 4 points even with a perfect window, and long hums are penalised. This justifies E1 (multi-window, time-consistent matching) before any retraining.
- Octave-robust comparison is worth adding to any re-ranking.

**Trade-off / what we gave up.** One model, one split. The "unvoiced reference" bucket mixes several causes that the script cannot separate (instrumental hooks, timestamp errors, rap).

**Revisit when.** A new song-side extractor or a new separation model is available; re-run the same script to see which buckets shrink.

## D-023 · E1: multi-window queries, time-consistent matching and DTW re-ranking (no retraining)
**Date:** 2026-09-28

**Context.** D-022 showed a 4-point chunking cost and a large penalty on long hums. The review's plan E1: cut the query into short windows, require their matches to be in order in the song, and re-rank the shortlist with key-invariant DTW, all with the existing D-012 model.

**Options.**
- Retrain with longer inputs.
- Change only the matching, on the existing embeddings.

**Decision.** Matching only. Script `training/scripts/eval_rerank.py`, helpers in `catalog/matching.py` and `catalog/real_pool.py`. Model D-012 `contour_v2_s0`, D-019 headline pool (full-song targets, chart previews, FMA; about 5,200–5,300 songs per query set).
- **base:** the API score (best 10 s chunk per song).
- **seq:** query cut into 5 s windows every 1 s (windows < 25% voiced dropped); each song cut the same way. Score = mean cosine similarity along the best monotone path through song windows, with local tempo 0–2 windows per step.
- **dtw:** minus the key-normalised, slope-constrained DTW error between the whole query and the song span the path picked.
- The top 50 songs by base are re-ranked by a per-query z-scored weighted sum. **Weights (grid 0 / 0.5 / 1 / 2) chosen on CHAD val songs only**, then applied unchanged to CHAD test, MTG-QBH and MLEnd. Chosen: base 1, seq 2, dtw 1.

**Results.** File: `docs/paper/results/e1/e1_v2_s0.json`. Top-1 / top-10.

| Query set | base (API) | seq only | base+seq | **base+seq+dtw** |
|---|---|---|---|---|
| CHAD val (568, selection) | 0.231 / 0.386 | 0.329 / 0.442 | 0.329 / 0.452 | 0.356 / 0.452 |
| CHAD test (2,482) | 0.303 / 0.469 | 0.443 / 0.541 | 0.449 / 0.547 | **0.465 / 0.550** |
| MTG-QBH sung (118) | 0.034 / 0.153 | 0.254 / 0.288 | 0.237 / 0.288 | **0.246 / 0.288** |
| MLEnd hum (4,804) | 0.030 / 0.078 | 0.124 / 0.138 | 0.116 / 0.137 | **0.119 / 0.137** |

- DTW alone (on the path's span) is weaker than seq but adds about 1.6 points top-1 on CHAD test in the fusion.
- Top-10 is capped by base's top-50 recall; the re-rank only reorders that shortlist.
- Compute: about 5 minutes for all four sets on one A100 host (embedding of windows is the main cost).

**Takeaways.**
- **The largest gain of the project on real audio, with no training.** CHAD test top-1 +16 points (0.303 → 0.465); MTG-QBH ×7 (0.034 → 0.246); MLEnd ×4.
- It confirms D-022: the model's embeddings were fine far more often than the single-chunk score showed. Order-consistent short windows fix long hums and the chunking cost.
- The shortlist is now the bottleneck at top-10. A better first stage (window-level ANN search) is the obvious next step.

**Trade-off / what we gave up.** About 5× more embeddings per song (1 s hop windows) and per-query DP and DTW on 50 songs. Fine offline, but the API would need a window index. One model seed only; weights picked on 29 val songs.

**Revisit when.** Moving this into the API (window index in pgvector), or when a new model is trained: re-run with the same weights and the same selection rule.

## D-024 · Octave-error correction of the song-side melody: no gain, not adopted
**Date:** 2026-09-28

**Context.** In D-022, octave folding shrank the "hum and song melody disagree by more than 2 semitones" group from 429 to 159 CHAD test hums. That made song-side octave errors look like the biggest fixable bucket. Luigi asked to fix them first and to measure on CHAD val before test.

**Options.**
- Octave-correct the song contours before chunking and windowing.
- Also correct the hum contours.
- Leave contours alone and fold octaves only in the DTW re-ranking cost.

**Decision.** Try all three, re-ranking exactly as in D-023, and pick on CHAD val.
- **Correction** (`catalog/octave.py`): a voiced frame that lies more than `limit` semitones from the running median of the voiced frames around it is moved by whole octaves towards that median. Two passes; gaps don't count towards the window. Ordinary melodic intervals are kept.
  - A first version folded every frame into ±6 semitones of the median. That destroys real leaps: CHAD val re-ranked top-1 fell from 0.356 to 0.250. It was dropped and its files kept in `results/d024/v1_fold_all` on the server.
- **Sweep on CHAD val** (568 hums): windows 1.5 / 3 / 6 s, limits 9 / 10.5 semitones, hums corrected too, and DTW with an octave-folded cost.
- **Stated rule:** a variant must beat the baseline by more than 0.5 point top-1 on val to be adopted, otherwise the simpler option wins.

**Results.** Files: `docs/paper/results/d024/`. Top-1 / top-10, D-023 re-ranking (base + seq + dtw, weights re-chosen on val for each variant).

| Variant | CHAD val |
|---|---|
| None (D-023) | **0.356** / 0.452 |
| Songs, window 1.5 s, limit 9 | 0.356 / 0.437 |
| Songs, window 3 s, limit 9 | 0.350 / 0.449 |
| Songs, window 6 s, limit 9 | 0.352 / 0.449 |
| Songs, window 3 s, limit 10.5 | 0.354 / 0.444 |
| Songs and hums, window 3 s | 0.345 / 0.444 |
| Octave-folded DTW cost only | 0.357 / 0.449 |
| Songs window 3 s + folded DTW | 0.349 / 0.449 |

- **Nothing passes the rule**, so nothing is adopted. For the record, two variants were also run on test (not used for any choice):
  - folded DTW: CHAD test 0.473 / 0.553 (D-023: 0.465 / 0.550), MTG-QBH 0.237 (0.246), MLEnd 0.117 (0.119);
  - song correction, window 3 s: CHAD test 0.468 / 0.542, MTG-QBH 0.203, MLEnd 0.117.
- **How much the corrector changes:** about 2% of voiced frames in CHAD target songs (10% of songs have more than 5% of frames moved), and 0.6% in CHAD hums.
- **MIR-QBSH:** unaffected by construction. Its references are MIDI, and the model is unchanged.

**Takeaways.**
- **The octave bucket was not really fixable this way.** The model was trained with octave-error augmentation (D-011/D-012), so short octave jumps already cost it little.
- The larger octave disagreements E0 measured are long stretches (longer than half a window) where the reference sits an octave away, e.g. a harmony or a different singer. A local corrector cannot tell those from real register changes.
- Correcting the hums hurts, so the hum side is not where the octave errors are.
- The bucket's size in D-022 came from the DTW measure, not from what limits retrieval.

**Trade-off / what we gave up.** Salience-based input (review E3, octave-folded salience) remains the principled route for reference-side octave and harmony errors. It needs retraining.

**Revisit when.** E3 (salience input) is run, or a melody extractor with explicit octave tracking is available.

## D-025 · Window-level first stage: 5 s window votes over the whole library, plus a pgvector window index
**Date:** 2026-09-28

**Context.** D-023's re-ranking only reorders the top 50 songs of the API's 10 s chunk score, so a target outside that shortlist cannot be recovered. On CHAD test only 58% of targets are in the base top 50, and on MTG-QBH only 29%. Luigi asked for a window-level first stage with 5 s windows indexed in pgvector.

**Options.**
- A deeper chunk shortlist.
- A first stage built from the same 5 s windows the re-ranker uses.

**Decision.** Window first stage (`scripts/eval_rerank.py --first-stage`, `catalog/rerank.py`), D-012 model, D-019 headline pool. No retraining.
- **Window vote:** each query window (5 s, 1 s hop) finds its best window in every song. A song's vote is the mean of those best similarities over the query windows. Order-free, and exact (GPU, all ~475k windows) in the offline evaluation.
- **Candidates:** the top K songs by base, by window vote, or their union, with K ∈ {50, 100, 200}.
- **Re-rank:** a z-scored weighted sum of base, window vote, seq and dtw (D-023). Weights come from {0, 0.5, 1, 2}, with base in {0, 1}.
- The candidate source, K and the weights are **all chosen on CHAD val only**. Chosen: union of the top 200 by base and by window vote; weights base 1, window 1, seq 0.5, dtw 1.
- **Serving:** `sql/002_windows.sql` (window table with an HNSW cosine index), `scripts/index_windows.py`, and `db.window_votes`. Each query window retrieves its 400 nearest windows; a song missing from a window's list gets that list's weakest similarity. `scripts/eval_window_index.py` compares these approximate votes with exact votes over the same stored windows.

**Results.** Files: `docs/paper/results/d025/`. Top-1 / top-10 (MRR), same pool and queries as D-023:

| Query set | Base (API) | D-023 re-rank | Window vote only | **First stage + re-rank** |
|---|---|---|---|---|
| CHAD val (568, selection) | 0.231 / 0.386 | 0.356 / 0.452 | 0.363 / 0.488 | 0.414 / 0.523 |
| CHAD test (2,482) | 0.303 / 0.469 | 0.465 / 0.550 | 0.448 / 0.559 | **0.512 / 0.616** (0.549) |
| MTG-QBH sung (118) | 0.034 / 0.153 | 0.246 / 0.288 | 0.602 / 0.720 | **0.619 / 0.712** (0.649) |
| MLEnd hum (4,804) | 0.030 / 0.078 | 0.119 / 0.137 | 0.183 / 0.228 | **0.193 / 0.232** (0.208) |

- **Candidate recall** (target among the candidates), base@50 → union@200:
  - CHAD test 0.579 → 0.752;
  - MTG-QBH 0.288 → 0.847;
  - MLEnd 0.143 → 0.319.
- **pgvector window index** (served library: 3,639 searchable songs, 494,528 windows; built in about 20 min). Target recall@50 / @200 on evenly spaced samples:

| Query set | Chunk HNSW (API today) | Window HNSW votes | Exact window votes |
|---|---|---|---|
| CHAD val (546) | 0.500 / 0.628 | 0.562 / 0.641 | 0.568 / 0.645 |
| CHAD test (600) | 0.587 / 0.688 | 0.650 / 0.730 | 0.658 / 0.733 |
| MTG-QBH (110) | 0.300 / 0.427 | 0.845 / 0.891 | 0.818 / 0.882 |

  - Approximate and exact votes find the target equally often (within 1 point, or better on MTG-QBH). Their top-50 lists overlap only 55–72%, because songs deep in the list get imputed scores.
  - About 0.15 s per query for all three searches together (not a tuned latency measurement).
- **MIR-QBSH:** unaffected by construction. The model is unchanged, and the MIR-QBSH protocol (contour vs MIDI, D-005) does not use the library search.

**Takeaways.**
- **The biggest single gain on real audio so far, still without training.** Against the API's score:
  - CHAD test top-1 0.303 → 0.512 and top-10 0.469 → 0.616;
  - MTG-QBH top-1 0.034 → 0.619 (18×);
  - MLEnd 0.030 → 0.193.
  - Against D-023: +4.7 top-1 and +6.6 top-10 on CHAD test.
- **Long queries were the problem, and window votes fix it.** On MTG-QBH (27 s sung queries), the window vote alone reaches 0.602 top-1, while any single-embedding query stays below 0.05.
- **It can be served.** The HNSW window index keeps the exact vote's recall, and needs about 5× the chunk index's rows.
- MLEnd stays low: 4 of its 8 songs are in the library, and its hums are short and hard (D-015/D-016).

**Trade-off / what we gave up.**
- About 5× more vectors (494k vs 102k) and one HNSW query per query window.
- The re-rank adds per-query DP and DTW on up to 400 songs, which is fine offline. The API still needs this pipeline wired in.
- One model seed only; weights were picked on 29 val songs.

**Revisit when.**
- Wiring the pipeline into `server/` (window votes → re-rank).
- After E2a or any new model: re-run with the same selection rule.
- When the library passes ~10k songs: check the recall and latency of the window HNSW index.

## D-027 · The window pipeline in the live search API
**Date:** 2026-09-28

**Context.** D-025's first stage and re-ranking raised real-hum accuracy sharply, but only in the offline evaluation. `POST /search` still ranked songs by their best 10 s chunk (D-017).

**Options.**
- Keep chunks in the API and use windows offline only.
- Serve the D-025 pipeline with exactly the offline settings.

**Decision.** Serve it (`catalog/window_search.py`, `catalog/search.py`, `server/api.py`).
- **First stage:** the top 200 songs from the chunk HNSW index (1,000 nearest chunks), plus the top 200 by window votes from the window HNSW index (400 nearest windows per 5 s query window).
- **Features:** for each candidate, computed from its stored vectors: base, window vote, seq and dtw, the same functions as `eval_rerank.py`.
- **Re-rank:** the D-025 weights (base 1, window 1, seq 0.5, dtw 1), chosen on CHAD val only.
- **Storage:** `sql/003_contours.sql` stores each song's melody contour (float16) for the DTW step. `index_windows.py` now fills both windows and contours, and the batch job runs it after `index_tracks.py`.
- **Memory:** candidate vectors are cached in the API process after first use (`SongCache`).
- **Queries:** the whole-query embedding keeps the first 20 s as before. Query windows cover up to 60 s.
- **API:** `mode=windows` (default) or `mode=chunks`. `windows` falls back to chunks when the library has no window index. `/health` reports whether the window index exists. The response adds `mode`; each result's `score` is the fused score, and `best_start_s` is where the matched path starts in the song.
- **Tests:** `tests/test_window_search.py` covers the grid placement, the ranking of a time-consistent song, songs without windows, the cache, contour storage, the fallback and mode validation.

**Results.** File: `docs/paper/results/d027/live_v2_s0.json`. `scripts/eval_live_search.py` sends real hums (cached contours, so no RMVPE time) through the same functions `/search` calls, against the served library: 3,639 searchable songs, including all CHAD, MTG-QBH and MLEnd targets. Samples are evenly spaced. Top-1 / top-10, mean search time:

| Query set | Chunks (D-017 API) | Windows (D-027) |
|---|---|---|
| CHAD test (300) | 0.300 / 0.447, 0.02 s | **0.517 / 0.657**, 0.34 s |
| CHAD val (300) | 0.147 / 0.240, 0.02 s | **0.260 / 0.337**, 0.45 s |
| MTG-QBH sung (110) | 0.036 / 0.127, 0.02 s | **0.618 / 0.755**, 0.56 s |

- The live path reproduces the offline D-025 gain (CHAD test 0.512 / 0.616 and MTG-QBH 0.619 / 0.712 offline, on a different pool).
- CHAD val is lower here than offline because the served library contains every CHAD song (train and test too) as competitors. Offline, the pool held only the val songs' own targets.
- Search time grows from about 0.02 s to 0.3–0.6 s per query on the A100 host, before RMVPE. Most of that is the HNSW window queries and the first database load of candidate vectors.

**Takeaways.**
- **The API now serves the best pipeline we have**, with the same numbers as the offline evaluation.
- Latency is acceptable for a hum search (RMVPE on the audio costs more), but it is 15–25× the chunk search.

**Trade-off / what we gave up.**
- Memory in the API process grows with the songs that have been candidates: up to ~0.5 GB of window vectors at today's library size.
- A second index and a contour table must be kept in sync by the batch job.

**Revisit when.**
- The library passes ~10k songs: measure HNSW recall, latency and cache memory, and consider batching the window queries into one SQL call.
- A new model is adopted: re-index chunks, windows and contours with it.

## D-026 · E2a: fine-tuning D-012 on self-supervised real-song windows with the CLEWS loss (1 seed): small gain, within seed noise
**Date:** 2026-09-28

**Context.** D-019 and D-021 found that real hum pairs from 116 CHAD songs do not help, and the review (E2) named song diversity on the reference side as the lever. E2a is the license-clean version: self-supervised pairs from real song melody tracks we already have, with CLEWS-style weak labels (Serrà et al., ICML 2025) so that no single reference window is assumed to be "the" match.

**Options.**
- **E2a** (no new data).
- **E2b** (CHAD cover songs; needs new downloads and Luigi's OK, not started).

**Decision.** E2a, 1 seed, recipe fixed before training (`configs/train_contour_e2a.yaml`, `contour/song_pairs.py`, `losses.clews_loss`).
- **Songs:** 1,806 songs with usable melody tracks.
  - 473 YouTube chart songs, which are distractors and never targets. 473 of the 1,806 have a mix track.
  - The **even-numbered half of FMA** (1,333 after filters).
  - Any song whose normalized title matches an evaluation target is dropped.
  - The odd FMA half stays untouched, and a "clean" pool without the trained FMA half is reported next to the headline pool.
- **Pairs:** the query is a 3–12 s window of the song's vocal-stem melody, or with probability 0.5 the same window of the full-mix melody when the song has one (a second extraction route). It is humanized and augmented with the D-021 query augmentation. The references are 4 vocal-stem windows starting within ±3 s of the query, each slightly longer or shorter and slightly stretched.
- **Loss:**
  - InfoNCE on HumTrans pairs, exactly as in D-012, plus the CLEWS loss on the song-window batch (weight 1).
  - CLEWS terms: the positive is the best of a song's 4 reference windows, and each negative is another song's closest window (R_min). Loss = mean positive d² + log(ε + mean exp(−γ d²)), with γ = 5 and ε = 1e-6.
  - d² is 2 − 2 cos on unit vectors, because the index uses cosine similarity; the paper uses unnormalized Euclidean distance.
- **Training:**
  - From `contour_v2_s0/last.pt`, LR 3e-5, at most 3,000 steps.
  - Validate every 100 steps on CHAD val songs and stop after 5 validations without improvement (the D-021 rule). The reported checkpoint is `best.pt`.
  - The song batch was cut from 128 to 64 songs after two out-of-memory crashes, before any completed training. A data bug on nearly silent windows was fixed at the same point.
  - Result: best at step 1,800, early stop at 2,300; about 0.4 GPU-h.
- **Evaluation:** `eval_rerank.py --first-stage` for both models. The base score (the old API), the D-023 re-rank and the D-025 pipeline are each chosen on CHAD val separately for each model and pool. MIR-QBSH is run with `eval_contour.py` (+2,000 Essen distractors).

**Results.** Files: `docs/paper/results/d026/`. CHAD val (selection): 0.394 → **0.440** in the training validator (145-song pool). Top-1 / top-10:

| Setting | D-012 s0 | E2a s0 |
|---|---|---|
| **Headline pool (`charts_fma`, ~5.3k songs)** | | |
| CHAD test, base score (old API) | 0.303 / 0.469 | 0.314 / 0.490 |
| CHAD test, D-023 re-rank | 0.465 / 0.550 | 0.471 / 0.577 |
| CHAD test, **D-025 pipeline** | **0.512** / 0.616 | 0.505 / **0.620** |
| MTG-QBH, D-025 pipeline | **0.619** / 0.712 | 0.593 / **0.720** |
| MLEnd, D-025 pipeline | 0.193 / 0.232 | **0.195 / 0.240** |
| **Clean pool (trained FMA half removed, ~3.8k songs)** | | |
| CHAD test, base score | 0.328 / 0.505 | 0.345 / 0.520 |
| CHAD test, **D-025 pipeline** | 0.521 / 0.635 | **0.531 / 0.641** |
| MTG-QBH, D-025 pipeline | 0.593 / **0.737** | **0.636** / 0.729 |
| MLEnd, D-025 pipeline | 0.200 / 0.249 | **0.207 / 0.254** |

- **MIR-QBSH +2,000:** 0.872 / 0.961 → 0.852 / 0.953 (−2.0 top-1, −0.8 top-10). This is inside the review's pre-set limit of 0.01 top-10, but top-1 drops as in D-021 (0.856).
- **HumTrans test:** 0.994 top-1, unchanged.
- **Candidate recall** (union@200) rises slightly: CHAD test 0.752 → 0.760, MTG-QBH 0.847 → 0.881.

**Takeaways.**
- **Better embeddings, but the pipeline absorbs most of the gain.** The model-level scores improve on CHAD test by +1.1 top-1 and +2.1 top-10 (base score), and +2.7 top-10 after the D-023 re-rank.
- With the D-025 pipeline on top, the combined system moves by −0.7 / +0.4 on the headline pool and +1.0 / +0.6 on the clean pool.
- A single seed cannot resolve these differences: D-012's own seed spread on CHAD test base top-1 is 0.297–0.314. **So E2a does not clearly beat D-012 on CHAD test**, and it is not adopted for serving.
- **Self-supervised song windows do not hurt real hums, unlike training on 116 songs of real pairs (D-019).** On the clean pool, where the trained songs cannot act as distractors, the gains are consistent across all three real-hum sets. But they are small at 1.8k songs.
- MIR-QBSH top-1 pays about 2 points again. Anything trained towards real recordings drifts slightly away from the MIDI-reference protocol.

**Trade-off / what we gave up.** One seed (about 0.4 GPU-h plus about 1.5 h of evaluation per seed). Only 1.8k songs, far from the review's 20k+ target: FMA's full audio is not downloaded beyond the 3k library.

**Revisit when.**
- Seeds 1–2, to see whether the clean-pool gain holds.
- The chart library grows several-fold from the download batches: re-train on it.
- E2b is done (D-028): clear gain; adopted.

## D-028 · E2b: fine-tuning D-012 on CHAD cover → original pairs with the CLEWS loss (1 seed): clear gain, adopted
**Date:** 2026-09-29

**Context.** D-026 (E2a) fine-tuned D-012 on self-supervised windows of the songs we already had; the gain was inside seed noise. The review (E2) named real version variation as the stronger lever. Luigi approved E2b: download CHAD's cover set from YouTube under the D-019 rules (Mac only, standalone yt-dlp, no cookies, polite pacing, stop on the bot check, delete after copying, disclose in the paper), build cover→original contour pairs, fine-tune from D-012 with the CLEWS loss, evaluate with the D-025 pipeline, and check MIR-QBSH.

**Options.**
- Keep D-012 (serving model after D-026).
- E2a (self-supervised song windows, D-026).
- **E2b** (CHAD cover → original pairs).

**Decision.** E2b, 1 seed, recipe fixed before training (`configs/train_contour_e2b.yaml`, `contour/cover_pairs.py`).
- **Downloads (Mac, D-019 rules).**
  - List: the 2,000 CHAD cover groups with the most aligned-fragment evidence (of 4,409 with an available original and at least one cover). Every CHAD hum group outside the train split (195 groups: test, val, unsplit) is excluded with all its videos. MTG-QBH and MLEnd titles are filtered by yt-dlp `--match-filter`. A post-filter also drops any recording whose video title names a CHAD / MTG / MLEnd evaluation song.
  - Per group: the original and its best-correlated cover (shared fragment intervals from CHAD).
  - First attempt: segment downloads (`--download-sections`, fragments ±4 s) at the lowest bitrate, 6 in parallel. YouTube throttled without a JS runtime (~9 downloads/min). Luigi approved a standalone `deno` binary inside `~/hum2song_yt` only; with it, six long-lived yt-dlp workers fetched whole files at 48 kbps (~46 downloads/min). Segments from the first attempt were placed on a silent timeline at their CHAD start times before extraction.
  - Result: 3,004 audio files → 2,567 library rows after the title filter (206 blocked, 0 known). Melody extraction succeeded for all 2,596 tracks (29 already from an earlier harvest). Audio was deleted from the Mac and from Lambda after extraction. Scripts that rebuild the list are in `docs/paper/results/d028/download/`.
- **Pairs:** 1,156 cover → original items (one cover per group; both tracks extracted). The query is a 3–12 s window of the cover's melody inside a shared fragment (vocal stem, or with probability 0.5 the full-mix track), humanized and augmented like a hum. The references are 4 windows of the original around the aligned position (±3 s, slightly longer or shorter, slightly stretched). The CLEWS loss takes the best reference window as the positive.
- **Loss / training:** same as E2a (InfoNCE on HumTrans + CLEWS weight 1; from `contour_v2_s0/last.pt`, LR 3e-5, at most 3,000 steps; validate every 100 steps on CHAD val; stop after 5 validations without improvement). Best at step 800, early stop at 1,300; about 0.2 GPU-h.
- **Evaluation:** `eval_rerank.py --first-stage` for both models on the same ~5.3k-song pool (charts + FMA + YouTube targets). Weights and first stage chosen on CHAD val separately per model. MIR-QBSH with `eval_contour.py` (+2,000 Essen distractors).

**Results.** Files: `docs/paper/results/d028/`. CHAD val (selection): 0.394 → **0.434** (training validator; pipeline val 0.414 → 0.437). Top-1 / top-10:

| Setting | D-012 s0 | E2b s0 |
|---|---|---|
| CHAD test, base score (old API) | 0.303 / 0.469 | **0.332 / 0.505** |
| CHAD test, D-023 re-rank | 0.465 / 0.550 | **0.496 / 0.579** |
| CHAD test, **D-025 pipeline** | 0.512 / 0.616 | **0.533 / 0.638** |
| MTG-QBH, D-025 pipeline | 0.619 / 0.712 | **0.695 / 0.754** |
| MLEnd, D-025 pipeline | 0.344 / 0.412 | **0.376 / 0.449** |

- **MIR-QBSH +2,000 (anywhere):** 0.872 / 0.961 → 0.867 / 0.958 (−0.5 top-1, −0.3 top-10). Inside the review's pre-set limit of 0.01 top-10.
- **HumTrans test:** 0.996 top-1, unchanged.
- **Candidate recall** (union@200) rises slightly: CHAD test 0.752 → 0.766, MTG-QBH 0.847 → 0.864.

**Takeaways.**
- **Real cover → original pairs beat self-supervision.** E2a (1.8k self-supervised songs) moved CHAD test D-025 by −0.7 / +0.4; E2b (1.2k cover pairs) moves it by **+2.1 / +2.2**, and the base score by **+2.8 / +3.6**, outside D-012's own seed spread on CHAD test base top-1 (0.297–0.314).
- Gains hold on every real-hum set. MTG-QBH jumps the most (+7.6 top-1), where the window first stage already helped most.
- MIR-QBSH barely moves (−0.3 top-10), unlike D-021 and E2a (~2 top-1 points). Cover variation is closer to what the MIDI-reference protocol needs than either real-hum pairs of 116 songs or self-supervised song windows.
- **E2b is adopted as the new serving model** (`contour_e2b_s0/best.pt`). Chunks, windows and contours must be re-indexed with it before the API switches (follow-up; not done in this decision).

**Trade-off / what we gave up.**
- YouTube downloads under D-019 (personal research, transient audio, disclosed). About 1.5 h of Mac wall-clock with deno; about 2 GPU-h of extraction; about 0.2 GPU-h of training; about 1.5 h of evaluation.
- Only 1 seed. Only the best cover per group (no second cover). About half of the 2,000-group target became usable pairs (missing videos, title filter, failed downloads).
- Deno binary on the Mac (deleted with the folder). Whole-file downloads for most of the set, because long-lived yt-dlp workers cannot cut different sections per video; segments from the first attempt were kept and stitched.

**Revisit when.**
- Seeds 1–2, or a second cover per group, to see how far the gain goes.
- Done in D-029: library and live API are on `contour_e2b_s0`.
- More CHAD covers become available without the bot check (or with a JS runtime that stays inside the D-019 rules).

## D-029 · Live library re-indexed on the adopted E2b model
**Date:** 2026-09-29

**Context.** D-028 adopted `contour_e2b_s0/best.pt` as the serving model, but the pgvector library (chunks, windows, contours) and the live API still used D-012 (`contour_v2_s0/last.pt`). Mixed encoder and index would make search wrong.

**Options.**
- Leave the index on D-012 until a later batch.
- **Re-embed everything with E2b and point the API at it.**

**Decision.** Full re-index with E2b, then restart the API.
- `--force` on `build_library.py`, `index_tracks.py` and `index_windows.py` re-embeds songs already in the database (insert still replaces one song at a time; CASCADE clears that song's chunks/windows/contours).
- Job `jobs/d029a.sh`: stop uvicorn → re-embed FMA (`fma_full_3k`) and YouTube (`youtube_v1`, `youtube_charts_v1`) with `--ckpt ckpt/contour_e2b_s0/best.pt --force` → rebuild windows and contours → start uvicorn with `H2S_CKPT` pointing at the E2b checkpoint.
- Contours themselves do not depend on the encoder; they are rewritten because CASCADE drops them with the song row.

**Results.**
- Library: **4,001 songs**, **3,781 searchable**, **106,063 chunks**, window index present; every song's `model_ver` is `contour_e2b_s0/best.pt`.
- `/health` reports `"model": "contour_e2b_s0/best.pt"`.
- Smoke search (one CHAD hum, `mode=windows`): HTTP 200; warm latency about **0.45 s** per query (cold first call about 8.5 s for model and cache load).

**Takeaways.** The live demo now serves the model that D-028 adopted. Re-index cost was about 40 minutes on the A100 (most of it window embedding).

**Trade-off / what we gave up.** Search was offline for the re-index window. A `psql` truncate at the start of the job failed (client not on PATH); `--force` still replaced every song, so the end state is clean.

**Revisit when.** A new model is adopted (E3 or later): re-run the same job with the new checkpoint.

## D-030 · E3: soft-salience input instead of hard F0 (1 seed): clear loss, keep E2b
**Date:** 2026-09-29

**Context.** The lit-review plan (E3) asked for RMVPE's 360-bin salience, cropped to ±18 st around the clip's median voiced pitch, with a voicing channel, trained like the best E2 variant (E2b). The hope was that keeping pitch alternatives would help on reference-side octave and harmony errors that hard F0 cannot recover. D-024 already noted salience as the principled route for those errors.

**Options.**
- Keep E2b hard-F0 serving model (D-028 / D-029).
- **E3 soft-salience encoder**, same cover→original CLEWS recipe as E2b.

**Decision.** E3, 1 seed, recipe fixed before training (`configs/train_contour_e3.yaml`, `input_kind: salience`).
- **Input:** for every contour (hum, MIDI, cover, original) build an 181-bin soft peak (±18 st at 20 cents/bin) centred on the clip's median voiced pitch, plus a voicing channel (182-d features). The crop gives key invariance the same way hard F0 uses median normalisation.
- **Limitation (disclosed):** under D-019 the YouTube cover audio was deleted after F0 extraction, so this run cannot feed *real* RMVPE 360-bin salience on the song side. Soft peaks are rebuilt from the cached hard F0. That tests the salience-shaped front end and the crop, but it does **not** preserve RMVPE's alternate peaks on references. HumTrans / MIDI sides use the same soft-from-F0 path for consistency.
- **Training:** E2b cover pairs (1,156) + HumTrans InfoNCE; init from `contour_e2b_s0/best.pt` with compatible tensors only (front end random, 84/85 tensors loaded); LR 3e-5, at most 3,000 steps; select on CHAD val; early-stop patience 5. Best at step 2,600 (val top-1 **0.363**); about 1 GPU-h.
- **Evaluation:** `eval_rerank.py --first-stage` for E2b and E3 on the same pool; MIR-QBSH +2,000.

**Results.** Files: `docs/paper/results/d030/`. Top-1 / top-10:

| Setting | E2b (hard F0) | E3 (soft salience) |
|---|---|---|
| CHAD val (selection) | 0.434 | 0.363 |
| CHAD test, base score | **0.332 / 0.505** | 0.268 / 0.422 |
| CHAD test, **D-025 pipeline** | **0.533 / 0.638** | 0.481 / 0.595 |
| MTG-QBH, D-025 | **0.695 / 0.754** | 0.542 / 0.653 |
| MLEnd, D-025 | **0.376 / 0.449** | 0.287 / 0.377 |
| MIR-QBSH +2,000 anywhere | **0.867 / 0.958** | 0.728 / 0.898 |

**Takeaways.**
- Soft-from-F0 salience is **worse on every set**, including a large MIR regression (−6 top-10 points, outside the 0.01 limit).
- With no real multi-hypothesis salience on the song side, E3 cannot do what the review asked. The soft peak is a blurred hard F0; the new front end has to re-learn pitch from a wider input and does not catch up in 3k steps.
- **E2b stays the serving model.** Live index from D-029 is unchanged.

**Trade-off / what we gave up.** About 1 GPU-h train + about 2 h eval. Did not re-download cover audio for real RMVPE salience (D-019 delete-after-copy; Mac not used).

**Revisit when.** Source audio for the cover set (or another large song set) is available long enough to cache real 360-bin salience on both sides, or an octave-folded salience variant (Salamon 2013) is tried with real posteriors.

## D-031 · E4: many-melody synthetic whistle contours (1 seed): modest held-out gain, hums hurt — keep E2b
**Date:** 2026-09-29

**Context.** The lit-review plan (E4) asked for whistle robustness from **many melodies**, not from MLEnd's 8 songs. D-015's whistle pairs helped test people but D-016 showed that gain was song familiarity (held-out Hakuna+Potter whistle→hums 0.456 vs D-012's 0.630). E4(a): synthetic whistle-like queries (interval compression 0.55–0.85, octave folding, gaps) from thousands of real-song contours and Essen MIDI. E4(b) PESTO fine-tune on unlabeled MLEnd whistle audio was optional and **skipped** this run. Serving stays on E2b unless E4 also wins on main hum metrics.

**Options.**
- Keep E2b serving (D-028 / D-029); leave whistles as-is.
- **E4 synthetic many-melody whistle fine-tune** from E2b, then decide: adopt, dual whistle path, or keep E2b.

**Decision.** E4(a), 1 seed, recipe fixed before training (`configs/train_contour_e4.yaml`).
- **Train data:** HumTrans InfoNCE (light whistle-like aug, p=0.2) + **WhistleSynthDataset** always applying compress / fold (p=0.5, ±6 st) / gaps on windows from **7,545 melodies** (Essen deutschl+china minus 2,000 distractors and deut2282, plus E2a song contours: youtube_charts_v1 + even FMA). **No MLEnd whistle pairs.**
- **Init:** `contour_e2b_s0/best.pt`. LR 3e-5, at most 3,000 steps, early-stop patience 5. Select on mean of HumTrans val shift+7 and MLEnd val whistle (CHAD logged only for hum regression). Best at step **100**.
- **Code:** `WhistleSynthDataset`, `octave_fold`, `whistle_synth` config; short-contour fix in `smooth_voiced` (kernel longer than the clip).
- **Evaluation:** D-016 `eval_mlend.py --split test --song-holdout`; full MLEnd test; MIR-QBSH +2,000; D-025 `eval_rerank.py --first-stage` on CHAD + MTG-QBH. Files: `docs/paper/results/d031/`.

**Results.** Whistle = `whistle_peak→hum` centroid top-1 (8-way). Hum numbers = top-1 / top-10.

| Setting | E2b | E4 |
|---|---|---|
| MLEnd test, **held-out songs** (Hakuna+Potter) whistle→hums | 0.589 | **0.622** |
| MLEnd test, all 8 songs whistle→hums | 0.631 | 0.631 |
| MLEnd test, held-out hum→hums | 0.869 | 0.865 |
| CHAD test, **D-025 pipeline** | **0.533 / 0.638** | 0.525 / 0.635 |
| MTG-QBH, D-025 | **0.695 / 0.754** | 0.653 / 0.737 |
| MIR-QBSH +2,000 anywhere | **0.867 / 0.958** | 0.861 / 0.955 |

Lit-review expected held-out / full whistle ~0.63 → **0.68–0.72** (best case). Observed full whistle **unchanged at 0.631**; held-out +3.3 pp vs E2b but still around D-012's D-016 held-out level (~0.63), not the hoped band.

**Takeaways.**
- Many-melody synthetic whistles **do not** deliver the review's expected jump. They give a small held-out gain over E2b and **no** gain on the full 8-way test.
- Hums are **hurt**, especially MTG-QBH (−4.2 / −1.7 on D-025). CHAD test and MIR dip slightly.
- **Keep E2b.** Do **not** adopt E4 for serving and do **not** keep a dual whistle encoder — the whistle upside is too small and the hum downside is clear.
- D-016's lesson stands: teaching whistling needs something beyond contour-level squeeze-and-fold on MIDI/song F0 (or a better tracker, E4b, not tried here).

**Trade-off / what we gave up.** About 0.2 GPU-h train + about 0.5 h eval. Skipped PESTO self-supervised whistle tracker (E4b). One seed only.

**Revisit when.** E4b (PESTO / spectral-peak self-sup on unlabeled whistles) is tried, or a larger set of **real** whistle→song pairs with many melodies exists, or audio-level whistle synthesis + re-tracking changes what the peak tracker sees.

## D-032 · Decision layer: show / follow-up / hum-again from melody scores (+ optional OpenRouter phrasing)
**Date:** 2026-09-29

**Context.** Lit-review E0–E4 are done; serving stays on E2b (`contour_e2b_s0/best.pt`, D-028/D-029). PLAN/SPEC call for a decision layer after ranked search: confident clear winner, one follow-up question, or ask the user to hum again. SPEC §8 planned Jev + LLM via OpenRouter, with a **local threshold fallback**. The live API only returned ranked songs (D-017/D-027). Luigi asked for this layer next, toward a usable app path—not a full UI yet.

**Options.**
- Hard-code UI rules with no server decision.
- Full Jev-on-OpenRouter before any local path (blocked: no `OPENROUTER_API_KEY` on Lambda or the box; Jev model slug still unconfirmed in SPEC).
- **Local score policy as the branch picker** (melody scores primary), OpenRouter only for phrasing when the key is present; templates otherwise. Wire `/decide` + optional `decide=1` on `/search`.

**Decision.** Local-first decision layer (`training/src/hum2song/decide/`, `configs/decide.yaml`), matching SPEC’s fallback and the “LLM never picks the song” rule.
- **Features:** top scores, gap12/gap15, softmax P(top-1)/P(top-5), entropy, voiced_s, turn.
- **Policy:** `show` if P(top-1) ≥ `t_show` and relative gap (s1−s2)/(s1−s5) ≥ `show_min_rel_gap`; `ask_followup` if the top few carry mass and the list is not flat; `ask_retry` if too quiet, empty, or entropy too high (almost flat top-10).
- **Phrasing:** templates always; `OPENROUTER_API_KEY` + `llm_model` may rewrite follow-up / tips. `jev_model` left empty until the TypeSafe slug is confirmed—Jev is not called.
- **API:** `POST /decide` (JSON search payload) and `POST /search?decide=true` (attaches `decision`). `/health` reports `decide: true` and `openrouter: false|true`.
- **Tests:** `tests/test_decide.py` (mocked scores, no network). Live smoke on Lambda with fake payloads + one MIR-QBSH wav.

**Results (smoke, local policy; OpenRouter not configured).**
| Case | Action |
|---|---|
| Clear top (9.2 vs 5.1) | `show` — “Best match: Clear Hit — X.” |
| Close top-5 (~6.2…5.7) | `ask_followup` — which of top few / lyric-year hint; options listed |
| Almost-flat top-10 | `ask_retry` — hum chorus ~10 s |
| voiced_s 0.3 | `ask_retry` — barely caught a melody |
| Real MIR-QBSH query + `decide=true` | `show` on the live E2b index (example top gap ~1.43) |

**Takeaways.**
- Branching works without OpenRouter; scores stay the source of truth.
- **`OPENROUTER_API_KEY` is missing** on Lambda and the box. LLM phrasing and any future Jev call need Luigi to provide the key via the product secret flow (not pasted in chat). Until then, templates are used.
- Thresholds in `decide.yaml` are a starting point, not calibrated show-precision (SPEC’s `calibrate_jev.py` / ECE still future work).

**Trade-off / what we gave up.**
- No multi-turn `/v1/answer` session store yet; no metadata filter re-rank; no Jev beliefs.
- Follow-up options are top song labels (plus “not sure”), not decade/lang splits from `songs.meta` (often empty in the live library).

**Revisit when.**
- `OPENROUTER_API_KEY` is installed and a cheap `LLM_MODEL` is chosen.
- Jev slug is confirmed (`typesafe/jev-router` vs `~typesafe/jev-latest`).
- Val logs exist to calibrate `t_show` / `t_few` / entropy for a target show-precision.
- Building `/v1/answer` + session filters for the follow-up loop.

## D-033 · Grow live library with more open FMA (extra vocal 3k + Electronic 1k); no Mac, no piracy
**Date:** 2026-09-29

**Context.** Live search sat at about **4,001 songs / 3,781 searchable** (D-017 `fma_full_3k` + YouTube chart/full targets), serving E2b (`contour_e2b_s0/best.pt`). Luigi chose growth from **free open catalogs only** (FMA and similar): no Mac downloads, no YouTube chart queue, no scraping illegal sources. FMA `fma_full` still has far more CC-licensed tracks than the first 3k vocal draw. D-009 also lists MTG-Jamendo as Tier P; it has no fetch/index path in-repo yet.

**Options.**
- Pull chart YouTube / Mac house batch (rejected for this round).
- Start MTG-Jamendo (needs new catalog code + large download).
- **Reuse D-017 FMA ranged zip:** another vocal-genre 3k disjoint from `fma_full_3k`, plus 1k Electronic (melody-search gap: Electronic was excluded from D-017 vocals).

**Decision.**
- Extend `select_tracks` / `fetch_fma.py` with `--genres`, `--exclude-libraries`, and a second seed.
- **`fma_full_extra_3k`:** seed **20260929**, same vocal genres and 60–420 s as D-017, exclude ids in `fma_full_3k`. Genre mix: Rock 1755, Hip-Hop 455, Folk 321, Pop 279, International 143, Soul-RnB 18, Country 16, Blues 13. All 3,000 downloaded.
- **`fma_electronic_1k`:** seed **20260929**, genre Electronic only, 60–420 s. All 1,000 downloaded.
- Build both with **E2b** (`build_library.py`), then `index_windows.py` over `FMA_LIBRARIES = (fma_full_3k, fma_full_extra_3k, fma_electronic_1k)`.
- Mac untouched; chart YouTube queue left pending.

**Results** (Lambda `/health` + DB after `BUILD_DONE` 2026-09-30 02:19 UTC ≈ 22:19 ET Sep 29).

| Scope | Songs | Searchable | 10 s chunks |
|---|---:|---:|---:|
| Before (live) | 4,001 | 3,781 | — |
| After (live) | **8,001** | **7,245** | **187,570** |
| Δ | +4,000 | +3,464 | — |
| `fma_full_3k` (unchanged) | 3,000 | 2,783 | 70,844 |
| `fma_full_extra_3k` | 3,000 | 2,775 | 70,568 |
| `fma_electronic_1k` | 1,000 | 689 | 10,939 |
| All FMA in DB | 7,000 | 6,247 | 152,351 |
| YouTube full (unchanged) | 1,001 | 998 | 35,219 |

Window index: **7,245** songs windowed (**912,917** five-second windows); this run added **3,464** newly windowed songs (**396,039** windows). Electronic searchable share is lower (~69%) than vocal (~92.5%), as expected when vocal-stem chunks are sparse.

**Takeaways.**
- Open FMA growth is the fastest legal lever with existing tooling; +3.5k searchable in one night.
- Electronic helps catalog breadth but yields fewer voiced chunks per track under the vocal-stem pipeline.
- MTG-Jamendo / JamendoMaxCaps remain documented Tier P options (D-009) but were not ingested here.

**Trade-off / what we gave up.**
- No MTG-Jamendo this round; no chart/Mac commercial growth.
- Genre skew still FMA-heavy (Rock-dominated vocals).
- No new real-hum accuracy number on the expanded distractor pool (re-run live eval when needed).

**Revisit when.**
- MTG-Jamendo (or another CC full-track set) has a fetch + build path and disk budget.
- More FMA genres or a larger Electronic draw if hummed electronic hooks matter.
- Live CHAD / MTG-QBH / MLEnd eval on the 8k library to measure distractor cost.

## D-034 · One-screen hum UI: listen button, level bars, three decide states
**Date:** 2026-09-30

**Context.** D-032 can already show a winner, ask which song, or ask for another hum, on the live E2b API. There was no page to hum into. The API listens on `127.0.0.1:8000` on the Lambda box, and soundfile will not read a browser MediaRecorder webm. The look follows the whop-llc screen (white page, 40px bold heading, one solid blue button, slate type). No Whop name, logo, or copy.

**Options.**
- A multi-page recorder, result, and player, as in the early spec sketch.
- Open the API on the public internet and call it from a hosted site.
- **One static screen, plus a tiny same-origin proxy**, leaving the running search process alone.

**Decision.** One screen in `web/` (`index.html`, `app.css`, `app.js`).
- One listen button, the same solid control as that screen (blue, 48px, 14px radius). While the mic is open, colored bars behind it follow the mic level. The heading is 40px bold at −0.022em. Follow-up choices are quiet 12px-radius rows.
- The clip is 16-bit WAV, posted to `POST /search?decide=true` (windows mode).
- The page shows one of the three actions: the winning title, the follow-up choices, or the hum-again line.
- Picking a listed song shows that title. "Not sure" goes back to humming. There is still no `/v1/answer` session (D-032), so the pick is not a second search.
- `python web/serve.py` serves the page and forwards `/search`, `/health`, and `/decide` to `H2S_API` (default `http://127.0.0.1:8000`). Same origin, so the running API does not need a restart.
- The API also allows any browser origin (CORS `*`) the next time uvicorn starts. Search results are unchanged.

**Trade-off / what we gave up.**
- No player for the matched moment.
- A follow-up pick does not re-rank the library.
- The page is not a public URL by itself. Open the local server, and use an SSH tunnel if you are not on the box. The mic needs localhost or https.

**Revisit when.**
- `/v1/answer` exists and a follow-up pick should filter the search.
- The API is on a public host with TLS.

---

## D-035 · Listen screen is a one-tap teal circle (supersedes the D-034 look)
**Date:** 2026-09-30

**Context.** D-034 put the hum page on a white field with a blue button, matching the whop-llc screen. That page was only a loose reference for craft. The gesture is one tap to listen, and the white page was the wrong product.

**Options.**
- Keep the D-034 white page and blue button.
- A frosted phone UI.
- **A solid muted teal field, white type, and one large circle.**

**Decision.** Same `web/` screen, new look. D-034’s proxy and the three decide states stay.
- Full-bleed solid teal (`#60a088`). Bold white type. No menus, no white page, no frosted panels.
- Above the circle: a small white mic and “Tap to Listen”.
- The control is a large lighter circle with a thin white ring. Tap starts listening; tap again stops and searches. While the mic is open, white level bars and a pulse sit around the circle.
- After search, that same screen shows the song title, a short pick list, or “Hum again”.
- No Shazam name, logo, or wordmark.
- `python web/serve.py` still forwards `/search` to the API on port 8000.

**Trade-off / what we gave up.** The D-034 type match. A long title or pick list shares the circle’s screen instead of a separate results page.

**Revisit when.** The screen needs a player for the matched moment, or it moves into a native app with its own chrome.
