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
