# hum2song — weekend build plan (personal research project)

Last updated 2026-09-25. Every link below was checked on 2026-09-25. Numbers come only from the cited sources.
**Query types to support:** humming, whistling, singing (with lyrics). Output: the answer, a follow-up question, or "not sure, try again".

---
## 0. Architecture (as decided, with changes for the new scope)
```
iPhone: record 6-12 s → VAD/trim → Core ML encoder (fine-tuned) → 256-d L2-normed embedding
        (+ optional: on-device query-type classifier {hum, whistle, sing}; optional lyric ASR text)
Server: Postgres+pgvector (song-clip embeddings, many windows per song) → top-K clips → aggregate to top-10 songs
        → Jev (calibrated decision): SHOW_ANSWER | ASK_FOLLOWUP | NOT_SURE_HUM_AGAIN
        → optional LLM phrases the follow-up question
```

## 1. Model choice
**Start with `m-a-p/MERT-v1-95M`** (https://huggingface.co/m-a-p/MERT-v1-95M). It is 95M params, 24 kHz, 75 Hz frames, pre-trained on 5 s context, CC-BY-NC-4.0 (fine for non-commercial use), and loads with `transformers` (`trust_remote_code=True`).
Why it still makes sense in 2026:
- It remains the smallest strong open music SSL encoder. On the *tonal* probes that matter for melody it still holds up: the MuQ paper's own MARBLE table (https://arxiv.org/abs/2501.01108) has MERT at NSynth pitch 94.4 vs MuQ 92.3 / MuQ-iter 91.3, and GS key 65.6 vs 63.5 / 65.0. MuQ wins on semantic tasks (genre, tagging, singer).
- OMAR-RQ `multifeature-25hz-fsq` (https://huggingface.co/mtg-upf/omar-rq-multifeature-25hz-fsq) is the best open tonal model: pitch .940 and chord .749 vs MERT-330M .922 / .609 in https://arxiv.org/abs/2507.03482. But the released models are a 580M Conformer, CC-BY-NC-SA-4.0, and use their own `omar_rq` package. **Use it as the ablation / server-side song-tower candidate, not the phone model.**
- One encoder handles all three query types: MERT, MuQ and OMAR-RQ all take raw audio, so hum, whistle and sung input all work. None of them is melody-invariant out of the box, though. Singing with lyrics puts phonetic/timbre content into the embedding, and whistling sits 1-2 octaves above most training vocals. The invariance has to come from **contrastive fine-tuning + augmentation across query types** (section 3), not from the choice of base model.
- Skip for the melody tower: MuQ-MuLan and CLaMP 3 (semantic text-music alignment; CLaMP 3's audio branch is built on MERT features), plus Dasheng and BEATs (general audio / AudioSet, not tonal).
- **Required baseline:** a CHAD-style CQT + ResNet18 contrastive model (https://arxiv.org/abs/2312.01092). It is about 11M params, trivial for Core ML, and was trained on hummed + sung (cover) data.
- **Stretch goal:** asymmetric dual encoder. The phone runs a small student (CQT-ResNet or MERT distilled to 4-6 layers) and the server runs MERT-95M/OMAR-RQ for song clips, both projected into the same 256-d space.

**Lyrics branch (sung queries only, optional):** run Whisper (https://github.com/openai/whisper) on queries the classifier labels as `sing`, look the text up against a lyrics table, and pass the lexical match score to Jev as an extra feature. It is cheap, and it rescues off-key singers.

## 2. Data (verified links)
| Set | What | Query type | Link |
|---|---|---|---|
| MIR-QBSH (Jang) | 4431 sung/hummed queries, 48 ground-truth MIDI | hum + sing | http://mirlab.org/dataset/public/ · direct: https://music-ir.org/evaluation/MIREX/data/qbsh/MIR-QBSH-corpus.tar.gz |
| HumTrans | ~56 h humming + MIDI labels, official split | hum | https://huggingface.co/datasets/dadinghh2/HumTrans · paper https://arxiv.org/abs/2309.09623 |
| MTG-QBH | 118 a-cappella sung melodies (songs not included) | sing/hum | https://zenodo.org/records/1290712 · https://www.upf.edu/web/mtg/mtg-qbh |
| MLEnd Hums & Whistles | ~6K recordings, 8 songs, 235 people, hum + whistle | **whistle** + hum | https://mlenddatasets.github.io/hums_whistles/ · https://www.kaggle.com/datasets/jesusrequena/mlend-hums-and-whistles · `pip install mlend` |
| CHAD | aligned hum/cover fragments metadata + YouTube IDs + download script | hum + **sung covers** | https://github.com/amanteur/CHAD |
| MIR-1K | 1000 Chinese-pop singing clips, vocals/accompaniment in separate channels | **sing (lyrics)** | https://sites.google.com/site/unvoicedsoundseparation/mir-1k |
| DAMP-VPB | Smule solo singing — **no longer downloadable** (https://ccrma.stanford.edu/damp/) | — | skip |
| Song audio | downloaded personally | — | — |

**Synthetic queries** (from each song clip, via Demucs https://github.com/facebookresearch/demucs → vocals stem → CREPE https://github.com/marl/crepe f0 + confidence):
- **sung**: the raw separated vocal stem (keeps lyrics), plus pitch shift ±4 st, time-stretch 0.8-1.25, reverb, phone-mic EQ, and noise at SNR 3-30 dB. These ranges are the CHAD augmentation set.
- **hum**: resynthesize from the f0 contour with a harmonic series (rolloff ~-12 dB/oct), a formant-ish lowpass around 1 kHz, and amplitude taken from the stem RMS. Randomize voicing gaps and portamento.
- **whistle**: sine (+ small 2nd harmonic) from f0 shifted **+12 or +24 st**, clamped to about 500-4000 Hz. Add vibrato (4-7 Hz, ±20-50 cents), breath noise (band-passed white noise at -25 dB), and random octave errors.
- **"bad singer" noise**, applied to all synthetic queries: per-note pitch drift ±50 cents, global key offset, tempo warp (piecewise-linear DTW-style), and dropped notes.
- Mix ratio per batch: roughly 35% real queries, 25% sung stems, 25% synthetic hum, 15% synthetic whistle. Tune on the validation set.

## 3. Training (see `train_skeleton.py`)
- Encoder: MERT-95M → learned weighted sum of the 12 transformer-layer outputs (captured with forward hooks: MERT remote code on transformers 5.x returns `hidden_states=None`; checked 2026-09-25 with transformers 5.17) → attentive pooling → projection head (768→512→256, GELU, L2-norm).
- Loss: symmetric InfoNCE (CLIP-style) with a learnable temperature (init 0.07), queries vs song clips, in-batch negatives. **Mask same-song false negatives.** Optional auxiliary query-type head (3-way CE, weight 0.1) that feeds Jev.
- Schedule: freeze MERT for 1 epoch (train only the head), then unfreeze the top 4-6 layers at LR 1e-5 (head 1e-3), AdamW, wd 0.01, cosine schedule, bf16. Batch as large as fits (64-128 pairs; gradient accumulation doesn't add negatives, so use a memory queue / cross-batch memory if the batch is small).
- Song side: 8 s clips at 2 s hop over vocal/melodic sections. Index every window and aggregate to song level with max-sim.
- Budget: MERT-95M forward+backward on 8 s at 24 kHz should fit around 32-64 pairs on a 24 GB GPU (estimate, not measured).

## 4. pgvector schema
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE songs (
  song_id BIGSERIAL PRIMARY KEY, title TEXT NOT NULL, artist TEXT, year INT,
  lyrics TEXT,                       -- for the optional lyric branch
  meta JSONB DEFAULT '{}'::jsonb);
CREATE TABLE clips (
  clip_id BIGSERIAL PRIMARY KEY,
  song_id BIGINT REFERENCES songs(song_id) ON DELETE CASCADE,
  start_s REAL NOT NULL, dur_s REAL NOT NULL,
  model_ver TEXT NOT NULL,          -- re-embed on every model change
  emb vector(256) NOT NULL);
CREATE INDEX clips_emb_hnsw ON clips USING hnsw (emb vector_cosine_ops);
CREATE INDEX ON clips (model_ver);
CREATE TABLE queries (               -- logging for Jev calibration + eval
  query_id BIGSERIAL PRIMARY KEY, ts TIMESTAMPTZ DEFAULT now(),
  qtype TEXT CHECK (qtype IN ('hum','whistle','sing','unknown')),
  emb vector(256), top10 JSONB, decision TEXT, followup JSONB, true_song BIGINT);

-- top clips → top-10 songs (max-sim aggregation)
WITH knn AS (
  SELECT song_id, 1 - (emb <=> $1) AS sim FROM clips
  WHERE model_ver = $2 ORDER BY emb <=> $1 LIMIT 200)
SELECT s.song_id, s.title, s.artist, MAX(sim) AS score, COUNT(*) AS hits
FROM knn JOIN songs s USING (song_id)
GROUP BY s.song_id, s.title, s.artist ORDER BY score DESC LIMIT 10;
```
(`SET hnsw.ef_search = 200;` for recall. pgvector: https://github.com/pgvector/pgvector)

## 5. Jev decision step (hum / whistle / sing aware)
Input features per query: top-10 scores; gaps s1-s2 and s1-s5; softmax entropy over the top-10 at the learned temperature; hits per song; query type + classifier confidence; CREPE voiced ratio and pitch range (catches a too-short or too-noisy query); query duration; for `sing`, the lyric-match score and whether it agrees with the audio top-1.
Policy (Jev calibrated on validation logs, with **separate calibration per query type**, because whistles score lower on the same melody):
- **SHOW_ANSWER**: calibrated P(top-1 correct) ≥ τ_show (e.g. 0.8), or lyric match and audio top-1 agree.
- **ASK_FOLLOWUP**: the correct song is probably in the top-k but not separable (P(correct in top-5) ≥ τ_set and top-1 < τ_show). Pick the question that best splits the candidate set's probability mass (max expected information gain) over metadata: decade, language, male/female vocal, genre, "was it from a film/TV/game?", "do you remember any words?" (this one switches the user to singing mode), or "hum a different part (chorus?)". Then re-rank with the answer and loop at most 2 times. The optional LLM only phrases the chosen attribute question; it never picks the answer.
- **NOT_SURE_HUM_AGAIN**: low voiced ratio / too short, or P(correct in top-10) < τ_min. Suggest a specific fix: "try humming the chorus for ~10 s", "whistling was hard to catch, try humming", "sing the words if you know them".
- Concatenate multi-turn queries: average the embeddings of 2 attempts, or intersect their top-10 lists.

## 6. Evaluation
Metrics: **Top-1, Top-10, MRR** (plus Top-3/5), reported **per query type** and overall. Add Jev metrics: accuracy when an answer is shown, answer rate, follow-up rate, and ECE of the calibrated P(correct).
Splits (**split by song, never by recording**):
- **hum/sing, MIR-QBSH**: hold all 48 songs out of training. Build "Jang-Real" style references: real recordings of the 48 songs + ~2000 distractor songs (CHAD used 1886 imposters). Report the MIDI-reference protocol too if you add MIDI synthesis.
- **sing (lyrics)**: MTG-QBH 118 queries → 118 songs + distractors, plus a held-out set of separated vocal stems from songs not in training (label them "synthetic-sung").
- **whistle**: MLEnd whistles. It has only 8 songs, so hold out **all 8 songs** for test (add their audio to the distractor DB). Also report hums from the same people, for a paired hum-vs-whistle comparison.
- **hum, HumTrans**: use the official test split with MIDI-synth references. Its songs only overlap MIR-QBSH if you check; dedupe by title.
- Your own 30-50 real iPhone recordings (hum/whistle/sing of songs in your DB) as the "in-the-wild" test.

**Numbers to compare against** (Top-10 hit rate, from CHAD Table 1, https://arxiv.org/abs/2312.01092):
| System | Jang (MIDI refs, 2600-MIDI DB) | ThinkIt | Jang Real | MTG-QBH |
|---|---|---|---|---|
| CHAD metric learning (CREPE) | 0.921 | 0.966 | 0.868 | 0.883 |
| CHAD (CQT) | 0.840 | 0.786 | 0.867 | 0.747 |
| Stasiak f0-matching | 0.948 | 0.907 | – | – |
| ACRCloud (proprietary, MIREX) | 0.990 | 0.986 | – | – |
Also: note-based CNN-HMM, MRR 0.92 on MIR-QBSH (Interspeech 2017, https://www.isca-archive.org/interspeech_2017/mostafa17_interspeech.pdf). CHAD at 90K-song scale, humming queries: Top-10 0.707 (M_fused, C+H). Newer QBH papers with no free numbers yet: ByteHum (ICASSP 2024, https://doi.org/10.1109/icassp48485.2024.10448117) and a Conformer + harmonic-aware QBH paper (ISCAS 2026, https://doi.org/10.1109/iscas66217.2026.11562747). Get the PDFs before you claim SOTA.

## 7. Core ML conversion
1. Export the query tower only: `waveform (1, 24000*T)` → 256-d embedding. Fix T = 10 s, or use `ct.RangeDim` for 3-15 s.
2. `torch.jit.trace` (check that MERT's `trust_remote_code` model traces; if it breaks, use `torch.export`), then `coremltools.convert(..., convert_to="mlprogram", compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS17)`. coremltools: https://github.com/apple/coremltools
3. Optionally apply 8-bit weight palettization/quantization with `coremltools.optimize`. Estimated size: 95M params is about 190 MB fp16 or about 95 MB int8 (arithmetic, not measured).
4. Parity test: cosine(PyTorch, Core ML) > 0.999 on 100 clips. Measure latency on the device. If >300 ms or the app is too big, ship the distilled student.
5. Resample to 24 kHz on-device (AVAudioConverter) and apply the same loudness normalization as training.

## 8. Weekend schedule
- **Sat AM**: download datasets; Demucs + CREPE over the song library (run overnight on GPU if the library is large); build `pairs.jsonl` {query_path, song_id, clip_start, qtype, source}.
- **Sat PM**: CQT-ResNet baseline + zero-shot MERT (mean-pooled, no fine-tuning) retrieval numbers; pgvector up.
- **Sat night**: MERT fine-tune run 1 (frozen), then run 2 (top-6 unfrozen).
- **Sun AM**: per-query-type eval; Jev feature export + calibration; follow-up question bank.
- **Sun PM**: Core ML export + parity test; SwiftUI record → embed → POST → show / ask / retry.

## 9. Paper / arXiv
- Scope: "Hum, whistle or sing: a calibrated query-by-voice system with foundation-model encoders and interactive follow-up". Contributions: per-query-type benchmark (with a whistle eval), synthetic whistle/hum resynthesis ablation, calibrated abstain/follow-up policy (Jev) with ECE and answer-rate curves.
- arXiv: primary **cs.SD**, cross-list **eess.AS** (the Now Playing paper https://arxiv.org/abs/1711.10958 uses exactly this pairing). First-time submitters need **endorsement** (https://info.arxiv.org/help/endorsement.html). You can get it automatically with an institutional email + claimed papers; otherwise ask an established author you know in the area, using the endorsement-code link arXiv emails you. Don't mass-email.
- **ISMIR 2026 LBD is closed.** The page says it reached its 75-poster capacity; the deadline was 2026-09-25 AoE (https://ismir2026.ismir.net/call-for-late-breaking-demo). Format for next time: 2 pages + 1 page of references, non-anonymous, submitted via CMT, demo video encouraged. Target the ISMIR 2027 LBD, or post to arXiv first. Check deadlines for ICASSP/ISMIR 2027 main tracks yourself; this plan doesn't state them.
- Reproducibility: release code, the pair lists (IDs/timestamps, not audio), synthetic-query generator, and eval scripts.
