# Decision log

A running record of the choices made while building hum2song, the alternatives considered, and the trade-offs. Every entry states what we chose, what we gave up, and when we would revisit it.

---

## D-001 · Pitch extractor for synthetic queries: CREPE over SPICE
**Date:** 2026-09-26

**Context.** Stage 2 builds synthetic hum/whistle queries by separating vocals (Demucs), extracting the melody's pitch contour, and resynthesizing it. That needs a pitch extractor.

**Options.**
- **CREPE**: supervised pitch tracker; generally more accurate on clean vocals; heavier.
- **SPICE** (Google, [blog](https://research.google/blog/spice-self-supervised-pitch-estimation/), arXiv 1910.11664): self-supervised pitch tracker, small and fast, designed to run on-device; one of the building blocks behind Google's Hum to Search.

**Decision.** CREPE as the default. Inference runs on a server, so model size and on-device latency are not constraints, and pitch accuracy directly affects the quality of the synthetic training data.

**Trade-off / what we gave up.** A fully on-device design. If the goal were to run the whole pipeline locally on an iPhone (Apple A-series / Neural Engine), SPICE would be the natural pick, following Google's approach, trading some accuracy for size and speed.

**Revisit when.** We pursue on-device inference in the Swift app, or an ablation shows SPICE-generated training data performs on par with CREPE. Plan: keep the pitch extractor swappable and run one CREPE-vs-SPICE comparison for the paper.
