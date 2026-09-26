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
