# Query-by-humming against real recordings: literature review, novelty gap and experiment plan

*Written 2026-09-28. Covers work up to September 2026. Every source below was opened (abstract, HTML or PDF) unless it is marked **[UNVERIFIED]** or **[PARTIAL]**. Numbers are copied from the source table named next to them. "Top-k" means top-k hit rate. MRR means mean reciprocal rank.*

## 0. Where we stand (from DECISIONS.md, for context)

| Setting | Our result | Source |
|---|---|---|
| MIR-QBSH hums vs MIDI, 48 songs | top-10 0.993–0.995 | D-012 |
| MIR-QBSH + 2,000 Essen MIDI distractors, `anywhere` | 0.866 / 0.958 (top-1 / top-10) | D-012 |
| CHAD test hums (2,482 queries, 145 songs) vs **full real recordings**, targets only (140) | 0.468 / 0.667 | D-019 |
| same, among ~5.3k songs (`charts_fma`) | **0.303 / 0.470** | D-019 |
| same, after training from scratch on CHAD real pairs (116 songs) | 0.239 / 0.402 (worse) | D-019 |
| MTG-QBH sung queries vs full songs, ~5.3k songs | **0.040 / 0.136** | D-019 |
| MLEnd whistles → hums, 8-way, unseen songs | top-1 0.63 (hums 0.86) | D-016 |
| Rendered hums of extracted melodies, 2.8k FMA songs | 0.582 / 0.764 | D-017 |

Pipeline: query → RMVPE F0 → median-centred semitone contour + voicing → 5.2M-parameter conv + transformer → 256-d vector, trained with InfoNCE on HumTrans hum ↔ MIDI pairs. Songs: htdemucs vocals → RMVPE → the same encoder over 10 s windows with a 5 s hop. A song's score is its best window. A fine-tune from D-012 on CHAD pairs (`configs/train_contour_chad_ft.yaml`, D-021) is already set up and is not repeated here.

---

## 1. Systems that match hums to real recordings

### 1.1 Google Hum to Search (2020)
- **Source:** C. Frank, "The Machine Learning Behind Hum to Search", Google Research blog, 12 Nov 2020. https://research.google/blog/the-machine-learning-behind-hum-to-search/ (verified). I found no peer-reviewed paper about it. Searches for a follow-up paper found none; the blog is the only technical description.
- **Method:** a neural network maps a **spectrogram** directly to a melody embedding. There is no F0 or MIDI step. Both sides are embedded (hum or song segment), and search is nearest neighbour over embeddings of "segments of studio recordings". The system was adapted from the Now Playing / Sound Search fingerprinting models (Now Playing: arXiv:1711.10958).
- **Training data** (these details are the most useful part):
  1. mostly **sung** segments paired with recordings, with random pitch and tempo augmentation. By itself this "worked well enough for people singing, but not for people humming or whistling";
  2. **synthetic hums**: SPICE pitch extracted from sung audio → a simple tone generator, later replaced by "a neural network that generates audio resembling an actual hummed or whistled tune";
  3. **pair mining**: two singers' clips of the same song aligned with preliminary models give extra positive pairs;
  4. a triplet loss, plus an extra loss that "drives model confidence towards 100%" across the batch, so that easy and too-hard negatives also contribute.
- **Numbers:** none published. The blog says only that it has "a high level of accuracy" on more than 500k songs, searched among more than 50M segment images.
- **What it means for us:** Google's main lever was **data variety on the query side** (synthetic hum and whistle resynthesis) plus **mined song↔song pairs**, not architecture. Their stated failure mode, a sung-only model that fails on hums and whistles, matches ours with whistles. Everything else in their setup relies on proprietary paired data we cannot copy.

### 1.2 SoundHound / Midomi
- **Sources:** J. Rehmeyer, "Tracking Down a Tune", MIT Technology Review, 10 Apr 2007. https://www.technologyreview.com/2007/04/10/226000/tracking-down-a-tune/ (verified). Salamon, Serrà & Gómez (2013, §1.4 below) also describe SoundHound and Tunebot as matching "queries against other queries".
- **Method:** Midomi (Melodis, later SoundHound) matched a query against **other users' sung or hummed recordings** of each song, not against the original recording. Its CEO said it is "simply easier to match a hum to another hum rather than match it to an original recording". Matching used "melody, rhythm, and words". The database had "a few tens of thousands of songs" in 2007.
- **Numbers:** none published. SoundHound's current algorithm is not documented publicly **[UNVERIFIED beyond 2007 press]**.
- **What it means for us:** the query-by-example route has a cold-start problem (a song does not exist until someone sings it). Our D-013 MLEnd protocol is exactly this setting, which is why closed-set MLEnd numbers do not predict catalogue-scale numbers (D-018).

### 1.3 ACRCloud
- **Sources:** ACRCloud blog, "What is Query by Humming?", 10 Aug 2016, https://www.acrcloud.com/blog/what-is-query-by-humming/ (verified), and its MIREX 2016 announcement https://www.acrcloud.com/acrcloud-tops-audio-fingerprinting-query-humming-mirex-2016/ (seen in search results only). The numbers come from CHAD Table 1 (below), which cites the MIREX 2016 results.
- **Method:** proprietary. The blog describes a generic three-part QBSH design (a melody database built from MIDI, singing or polyphonic audio; query transcription; indexing and matching). It is deployed in Xiaomi MIUI and Omusic.
- **Numbers (CHAD Table 1, MIREX 2016, MIDI references):** top-10 **0.990** Jang (MIR-QBSH, 48 targets + ~2,600 MIDI DB), **0.986** ThinkIt, **0.972** MIREX Subtask 2. **No real-recording number is published.**
- **What it means for us:** ACRCloud is the ceiling on the *MIDI* protocol. Our MIR-QBSH `start_multi` top-10 of 0.974 with 2,000 Essen distractors is close to it, but ACRCloud has no public real-audio result to compare with.

### 1.4 Salamon, Serrà & Gómez (2013): audio-to-audio QbH with melody extraction
- **Source:** "Tonal representations for music retrieval: from version identification to query-by-humming", *IJMIR* 2:45–58. https://doi.org/10.1007/s13735-012-0026-0 (PDF verified).
- **Method:** Melodia melody extraction on the polyphonic song. The melody is quantised to semitones, **folded onto a single octave** and summarised as a pitch-class histogram about every 0.5 s. Matching uses **Q_max** (recurrence-quantification local alignment) between query and song. This is also the paper that introduced MTG-QBH.
- **Data:** MTG-QBH, 118 sung queries from 17 people on laptop microphones, **11–98 s long (mean 26.8 s)**.
- **Numbers (Tables 4 and 5):**
  - canonical collection (481 songs): MRR 0.45, **top-1 40.68 %, top-10 51.69 %**; good-tuning singers 50.00 / 63.75 %;
  - full collection (2,125 songs including cover versions): MRR 0.56, top-1 50.85 %, top-10 66.10 %.
  - The same paper quotes an earlier audio-to-audio system at MRR 0.57 on 427 audio songs, against 0.91 on 2,048 symbolic songs.
- **What it means for us:**
  1. A 2013 hand-built system gets **top-10 0.52 on MTG-QBH among 481 real songs**. We get **0.136 among ~5.3k**. The pools differ by 11×, but a gap this large points to something structural. The main suspect is **query length**: MTG-QBH queries average 26.8 s, while we embed one 10 s chunk against 10 s windows. D-017 already measured a large drop from query/chunk length mismatch (0.36 vs 0.58 top-1). Version noise (first YouTube search result, D-019) is a second suspect.
  2. **Several versions per song help** (+0.11 MRR). Indexing more than one recording per song is a cheap lever.
  3. **Octave folding** on both sides makes extraction octave errors harmless.

### 1.5 CHAD (Amatov et al., ISMIR 2023; arXiv:2312.01092)
- **Source:** "A Semi-Supervised Deep Learning Approach to Dataset Collection for Query-by-Humming Task". https://arxiv.org/abs/2312.01092 (full text verified). ISMIR 2023 PDF: https://archives.ismir.net/ismir2023/paper/000077.pdf. Data: https://github.com/amanteur/CHAD.
- **Method:**
  - Spleeter vocals on the song side.
  - Input is either **CREPE activations** (the 360-bin salience, not the argmax F0), trimmed to **3 octaves around the mean pitch** and downscaled to 80 × T/4, or a CQT (12 bins/octave, 7 octaves).
  - A ResNet18 on **short analysis windows**: `M_short` uses W = 3 s with a 0.25 s hop; `M_long` uses W = 8 s with a 0.64 s hop. The output is a **sequence of 128-d fingerprints** per clip.
  - Loss: NT-Xent (τ = 0.05) over groups of time-aligned fragments, with a multi-similarity miner.
  - Augmentation: pitch shift ±4 st, stretch 0.8–1.25, SpliceOut, mixing with other batch items at 5–10 dB, noise at 3–30 dB.
  - **Retrieval:** DTW on MIDI data; **maximum Pearson correlation of fingerprint sequences on real data** ("correlation coefficient gives performance improvement on real data"). At 90k scale: FAISS ANN top-5,000, then correlation re-ranking.
- **Data:**
  - H: 5,164 crowd-sourced hums (Yandex Toloka), time-aligned to original fragments, 4–20 s.
  - C: covers mined semi-supervised from YouTube: **5,494 originals, 31,630 covers**, 259 h of cover fragments, found by iterating model → cross-correlation alignment → retrain. The Billboard top 100 per year, 1960–2020, up to 10 covers each.
- **Numbers (Table 1, top-10):**

| | Jang (MIDI, ~2,600 DB) | ThinkIt | Subtask 2 | **Jang Real** (MIR-QBSH vs YouTube recordings + 1,886 imposters) | **MTG-QBH** (118 songs + 1,886 imposters) |
|---|---|---|---|---|---|
| CHAD, CREPE | 0.921 | 0.966 | 0.959 | **0.868** | **0.883** |
| CHAD, CQT | 0.840 | 0.786 | 0.866 | 0.867 | 0.747 |
| Stasiak (MIREX) | 0.948 | 0.907 | 0.968 | "near-random" (their reimplementation) | – |
| ACRCloud | 0.990 | 0.986 | 0.972 | – | – |

  Table 2 (DB90K, >90k real songs, CQT features), top-10 / top-100:
  - 126 in-house hums: `M_fused` trained on C+H **0.707 / 0.776**; trained on C only 0.621 / 0.759. Hum data matters.
  - 2,000 DAMP-VPB sung fragments: 0.899 (C+H) and 0.904 (C only).
- **What it means for us (the most important paper):**
  1. **Real audio references work at about 0.87–0.88 top-10 with about 2k songs**, on hum and sung queries whose songs were excluded from training. Our 0.47 top-10 at 5.3k (CHAD test, a different query set) shows real room to improve.
  2. Their training has **about 5.5k distinct songs**, mostly from **covers**. Our CHAD-pair run had 116 songs and memorised them (D-019). Song count, not hum count, is what we lack.
  3. Three design choices differ from ours, and each can be tested:
     - salience input instead of hard F0;
     - short windows (3 s) turned into a *sequence* of fingerprints and matched with correlation or DTW, instead of one 10 s vector;
     - a pitch-centred crop without hard quantisation.
  4. Stasiak's F0 DTW, the MIREX winner on MIDI, was "near-random" against real recordings in their hands. Pure contour matching needs a learned component on the reference side.
  5. Caveats: CHAD's imposter sets are internal and not released, so Jang Real and MTG-QBH are **not reproducible**. The C part also contains noisy fragments (correlation 0.3–0.5 flagged as uncertain).

### 1.6 ByteHum (ByteDance, ICASSP 2024)
- **Source:** X. Du, P. Zou, M. Liu, X. Liang, M. Chu, B. Zhu, "ByteHum: Fast and Accurate Query-by-Humming in the Wild", ICASSP 2024, pp. 1111–1115. https://doi.org/10.1109/ICASSP48485.2024.10448117. **[PARTIAL]** I verified only the abstract (Semantic Scholar says the paper is closed access). No arXiv version was found.
- **Method (abstract):**
  - a CNN on raw song audio, trained weakly supervised on a **source-separated cover-song-identification dataset**;
  - **unsupervised domain adaptation** to move from covers to hums;
  - "annotate[d] original recordings for three existing QBH benchmark sets", so hum benchmarks can be run against real recordings.
- **Numbers:** **[UNVERIFIED]**. The abstract says only that it "significantly outperforms existing QBH systems in terms of speed and accuracy". Get the PDF before citing numbers.
- **What it means for us:** a second group reached the same recipe independently: CSI data + separated vocals + adaptation to hums. Unsupervised domain adaptation from sung/cover to hum is an idea we have not tried: we have many unlabeled real hums (MLEnd, HumTrans, CHAD train) and many unlabeled real song contours (FMA).

### 1.7 Zalo AI Challenge 2021 "Hum to Song" (Pham et al., arXiv:2410.20352)
- **Source:** "An approach to hummed-tune and song sequences matching". https://arxiv.org/abs/2410.20352 (verified). Related repo: https://github.com/vovanphuc/hum2song (not opened).
- **Method:** mel-spectrogram → ResNet34 / VGG with an ArcFace-style loss → FAISS.
- **Data:** Vietnamese songs. Train: 2,901 hummed tunes paired with 2,901 song sequences over 1,000 unique songs. Public test: 500 hums, 419 song sequences. Private test: 1,067 hums, 10,153 song sequences.
- **Numbers:** public-test MRR@10 **0.946** (modified ResNet34), ResNet18 0.932, VGG16 0.923, MobileNetV2 0.892.
- **What it means for us:** high numbers, but the references are short pre-cut song *sequences* (not full songs), the set is closed-domain, and song overlap between train and test is not stated. It is not comparable to full-song retrieval. It does show that **spectrogram models trained on about 3k real hum↔song pairs** work in-domain.

### 1.8 Kung, Chen & Ding, Conformer QbH with a harmonic-aware mechanism (ISCAS 2026)
- **Source:** "Improved Query by Humming Using Conformer-based Network with Harmonic-Aware Mechanism", https://doi.org/10.1109/ISCAS66217.2026.11562747 (DOI from PLAN.md). **[PARTIAL]** I verified only metadata and the reference list (exa.ai index). The references include htdemucs, ByteHum, CoverHunter, DisCover and HANet (melody extraction), which suggests a real-audio setting. **Numbers [UNVERIFIED].**

### 1.9 Older real-audio contour systems
- **Jeon, Ma & Cheng (Motorola), ISMIR 2009**, "An efficient signal-matching approach to melody indexing and search using continuous pitch contours and wavelets". https://ismir2009.ismir.net/proceedings/PS4-18.pdf (verified). The paper describes itself as searching "a database of recorded pop songs by humming or singing an arbitrary part".
  - Method: level-normalised segments over **several lengths**, wavelet coefficients in a K-D tree, no DTW.
  - Real polyphonic test: 613 recordings (partly RWC), queries hand-checked for clear dominant F0; inclusion rate **86 % at n = 5, 88 % at n = 20**. On a 2,048-song MIDI set: 84.9 % at n = 20.
  - For us: a multi-length segment index is an old idea for arbitrary-position queries, and a cheap one.
- **Jeong (SK Telecom T-Brain), `jdasam/qbh_project`**, https://github.com/jdasam/qbh_project (README verified; no paper or numbers found).
  - Method: a (pitch, voicing) contour from a vocal-melody extractor (Kum et al.) → 512-d embedding, triplet loss.
  - Two stages: (1) **self-supervised on a large music-audio set**: the anchor is the extracted song melody, positives are rule-based "humming" distortions of it; (2) supervised fine-tuning on hum↔song pairs.
  - The closest published design to ours, and it **trains stage 1 on extracted real-song melodies, not MIDI**. That is the domain our training never sees.

---

## 2. Cover-song identification (CSI): the source of scalable training signal

All numbers are on SHS100K-TEST, Da-TACOS, Covers80 or DVI (track-level MAP / MR1 or NAR), not QbH.

| Model | Key idea | Reported numbers | Source |
|---|---|---|---|
| **CQTNet** (Yu et al. 2020) | CQT CNN, classification loss | SHS100K-TEST MAP 0.655, Covers80 0.840 (as reported in the ByteCover3 table) | arXiv:1911.00334 (as cited by CHAD); numbers from arXiv:2303.11692 Table 1 |
| **ByteCover / ByteCover2** (Du et al. 2021/2022) | ResNet-IBN on CQT, classification + triplet; BC2 reduces dimension | SHS100K-TEST MAP 0.836 (BC, 2048-d), **0.864** (BC2, 1536-d); Da-TACOS 0.791 (BC2) | arXiv:2010.14022; ICASSP 2022 doi:10.1109/ICASSP43922.2022.9747630; numbers from arXiv:2303.11692 |
| **ByteCover3** (Du et al., ICASSP 2023) | **Local (chunk) features + two-stage retrieval + local alignment loss (LAL)** for **short queries** | SHS100K-TEST with 30 s queries: MAP **0.734** vs 0.430 for ByteCover2; full-length 0.824 | https://arxiv.org/abs/2303.11692 (Table 1 verified) |
| **CoverHunter** (Liu et al., ICME 2023) | Conformer + attention time pooling + **coarse-to-fine chunk alignment** training | SHS100K-TEST MAP 0.875 (256-d) / 0.858 (128-d); Da-TACOS 0.865 | https://arxiv.org/abs/2306.09025 (verified) |
| **DisCover** (Xun et al., SIGIR 2023) | Disentangles version-specific from version-invariant factors (knowledge-guided + gradient-adversarial modules); plugs into CQTNet and others | CQTNet-Dis SHS100K MAP 0.658 **[PARTIAL: number from a review summary, not the paper table]** | arXiv:2307.09775; doi:10.1145/3539618.3591664 |
| **CLEWS** (Serrà et al., ICML 2025) | Learns from **weakly-labelled segments**: reduces the segment×segment distance matrix to a track distance (best-pair-without-replacement for positives, **min for negatives**), plus a modified alignment/uniformity contrastive loss | DVI-Test NAR 2.70 / MAP 0.774; SHS-Test MAP **0.876** (SOTA track level); "breakthrough" segment-level results. Loss ablation on DVI-Valid MAP: CLEWS 0.804, SupCon 0.676, triplet 0.717, classification 0.205 | https://arxiv.org/abs/2502.16936 (Tables 2–4 verified); code and checkpoints https://github.com/sony/clews |
| **Doras & Peeters** (ISMIR 2019) | Embeddings of **dominant-melody salience** (not F0), triplet loss | small set MAP 0.782 (unseen works) | https://archives.ismir.net/ismir2019/paper/000010.pdf (Table 2 verified) |
| MIREX 2025 CSI | New in-house set, 10,000 tracks / 80 works | Task page only; no results read | https://music-ir.org/mirex/wiki/2025:Cover_Song_Identification |

**What this means for us:**
1. CSI moved from track to **segment level**, and segment-level training is where the recent gains came from: ByteCover3 30 s MAP 0.43 → 0.73, CoverHunter's chunk alignment, CLEWS. QbH is by definition a segment-level problem, yet our training uses one fixed pairing per example and inference uses a max over independent windows.
2. **CLEWS's reductions solve our label-noise problem directly.** CHAD timestamps refer to the original video, and vocal stems contain backing vocals (D-019 lists this as a likely cause). A "best pair among several candidate reference windows" positive and a "hardest window" negative let the model choose which reference window matches, instead of trusting one timestamp.
3. **DisCover's disentanglement** is an alternative to our hand-built invariance (median subtraction). Adversarially removing "performer/recording" information is a direct way to push out hum-vs-recording style.
4. CSI models see chords and timbre as well as melody. For hum queries only the melody is shared, which is why CHAD/ByteHum separate vocals first. We already do that.

---

## 3. Melody extraction for the reference side

| Method | Numbers (overall accuracy, OA) | Source |
|---|---|---|
| RMVPE (Wei et al., Interspeech 2023) | Our measurement: htdemucs → RMVPE OA 0.924 on ADC2004 vocal, 0.927 on MIR-1K (possibly contaminated) | arXiv:2306.15412; D-014 |
| Joint network with attention aggregation + self-consistency (Jing et al., Interspeech 2025) | OA **91.6 / 92.5 / 78.9** on ADC2004 / MIREX05 / MedleyDB | https://www.isca-archive.org/interspeech_2025/jing25_interspeech.html (verified via search snippet and D-014) |
| SpectMamba (semi-supervised, confidence binary regularisation, 2025) | OA 80.67 / 84.64 / 72.62 on the same sets | https://arxiv.org/abs/2505.08681 |
| HANet, harmonic attention (ICASSP 2025) | **[UNVERIFIED numbers]** | doi:10.1109/ICASSP49660.2025.10889955 |
| Kum et al., teacher-student vocal melody extraction (used by Jeong's QbH) | not checked | see qbh_project README |

**What this means for us:** D-014 already showed that extraction costs only 0.03–0.06 top-10 on MIR-1K, and htdemucs → RMVPE is roughly at published SOTA on vocal material. **A better extractor is not the lever.** The lever is making the encoder robust to the *kind* of reference errors real songs produce (backing vocals, harmonies, ad-libs, octave jumps, instrumental hooks). There are two ways:
- keep the extractor's **salience/posteriorgram** instead of argmax F0, as CHAD (CREPE activations) and Doras & Peeters (dominant-melody salience) do;
- train on extracted-reference contours rather than MIDI (Jeong stage 1).

---

## 4. Self-supervised music encoders

| Model | Facts relevant here | Source |
|---|---|---|
| **MERT** (Li et al., ICLR 2024) | Masked-LM music SSL (RVQ-VAE + CQT teachers), 95M/330M | https://arxiv.org/abs/2306.00107 |
| **MuQ** (Zhu et al. 2025) | SSL with Mel-RVQ tokenizer, Conformer; beats MERT on MARBLE | https://arxiv.org/abs/2501.01108 |
| **CLaMP 3** (Wu et al. 2025) | Contrastive alignment of sheet music, MIDI, audio and multilingual text; its audio branch is a 12-layer transformer on **MERT-v1-95M features** | https://arxiv.org/abs/2502.10362; https://github.com/sanderwood/clamp3 |
| MuQ/MERT as fingerprint backbones (Singh et al., SoundPatrol 2025) | Fine-tuned MuQ gives 88.2 % track top-1 under distortions vs NAFP 63.5 %. For segment localisation they fit t_ref ≈ a·t_q + b with **Huber regression** over matched segments | https://arxiv.org/abs/2511.05399 (verified) |
| Query-by-vocal-imitation (Greif et al. 2024) | AudioSet-pretrained CNN dual encoder fine-tuned with adapted NT-Xent sets SOTA on VimSketch / VocalImitationSet | https://arxiv.org/abs/2408.11638 |

**What this means for us:**
- Our own D-011 result stands: MERT fine-tuned on about 1,000 HumTrans melodies reached only top-10 0.583 on MIR-48, against 0.984 for the contour encoder. The fingerprinting evidence is about **recognising the same recording** under distortion, not melody identity across performers.
- No paper I found reports MERT/MuQ/CLaMP 3 for **hum → real recording** retrieval. It is an open question, but our prior result argues against making it the main path.
- A cheap, publishable **ablation**: frozen MuQ/MERT features of the vocal stem as a *reranking* feature fused with the contour score on the top-50. Low priority.
- CLaMP 3 matches whole pieces across modalities. Its segment-level melody invariance is unknown **[not evaluated anywhere I found]**.

---

## 5. Synthetic hum/whistle generation and voice conversion for augmentation

- **Google (2020):** SPICE pitch → tone synthesis → later a neural hum/whistle generator. This is the only large-scale evidence that synthetic hums help, and it has no numbers (blog above).
- **Voice-conversion tools that could produce synthetic hums or whistles from sung stems:**
  - Seed-VC, zero-shot diffusion-transformer VC with F0 conditioning and a singing checkpoint: https://arxiv.org/abs/2411.09943;
  - Vevo2, speech/singing generation with humming melody control (Amphion): https://arxiv.org/abs/2508.16332;
  - R2-SVC: https://arxiv.org/abs/2510.20677 (search snippet only).
  - None of these papers evaluates QbH.
- **Our own evidence:**
  - Essen MIDI synthetic pairs gave no gain on MIR-QBSH (D-012).
  - Squeezed-hum "whistle augmentation" gave only +0.026 whistle top-1 (D-015).
  - Both were **query-side, MIDI-referenced**. Neither tested synthetic pairs whose **reference side is a real extracted song contour**.
- **What this means for us:** audio-level resynthesis matters for spectrogram models (Google). Our model sees only F0, so audio timbre is irrelevant to it *unless* the resynthesis changes what RMVPE or the whistle tracker outputs. The useful form for us is **contour-level synthesis from real song melodies**: take an FMA song's extracted vocal contour, apply `humanize` and hum-style augmentation, and optionally render it as audio and re-track it with RMVPE so tracker artefacts are included. D-017's probe already does this for evaluation, but it was never used for **training**.

---

## 6. Chunking, alignment and subsequence matching

| Approach | Key point | Source |
|---|---|---|
| Stasiak, "Follow That Tune" (Archives of Acoustics 39(4), 2014) | DTW + adaptive **tune follower** that tracks slow pitch drift during alignment. MIR-QBSH vs 48 + 5,274 Essen: DTW top-10 69.44 % / top-1 47.60 % → tune follower 75.20 / 55.41 → adaptive, whole-query 79.15 / 66.26 % | http://ics.p.lodz.pl/~basta/pre-prints/Stasiak_AoA_2014.pdf (Tables 1 and 3 verified) |
| CHAD | 3 s windows, 0.25 s hop → fingerprint sequence; **max Pearson correlation** beat DTW on real data; ANN top-5,000 then correlation rerank | arXiv:2312.01092 |
| ByteCover3 | local features + two-stage retrieval + LAL for short queries | arXiv:2303.11692 |
| CoverHunter | coarse model aligns chunks, fine model trains on the aligned chunks | arXiv:2306.09025 |
| CLEWS | best-pair-without-replacement segment reductions | arXiv:2502.16936 |
| SoundPatrol (fingerprinting) | segment kNN + Huber-regression time-consistency check | arXiv:2511.05399 |
| Jeon et al. 2009 | multi-length level-normalised segments, K-D tree | ISMIR 2009 PS4-18 |
| Raffel & Ellis, ICASSP 2016 | attention-based embeddings prune subsequence-DTW search | https://colinraffel.com/publications/icassp2016pruning.pdf (not opened) |
| Kotsifakos et al., VLDB J. 2015 | embedding-based subsequence matching with gap/range tolerances, applied to QbH | https://crystal.uta.edu/~athitsos/publications/kotsifakos_vldbj2015.pdf (not opened) |

**What this means for us:**
- We use a single 10 s query vector vs 10 s windows with a 5 s hop, scored by the best window. That is the simplest possible matcher.
- Every strong real-audio system uses **multiple short windows plus sequence-level consistency** (CHAD correlation, ByteCover3, SoundPatrol).
- Our encoder is key-invariant by construction, so a **second-stage key-invariant subsequence DTW (with tune following) on the top-K candidates** is cheap and complements it.
- Our own D-008 DTW is already slope-constrained.

---

## 7. Whistle pitch tracking and query-by-whistling

- **Evidence that whistles need their own tracker:** whistling is a near-pure tone in about 500–5,000 Hz. Nilsson et al., CISP 2008, doi:10.1109/CISP.2008.415 (in REFERENCES_and_CAPTION.md). Our D-007/D-013 spectral-peak tracker beats RMVPE by 0.47 top-1.
- **Query-by-whistling literature:**
  - Shen & Lee (2007), whistle → MIDI with string matching, doi:10.1007/s11042-007-0128-5 **[PARTIAL]**;
  - Google says Hum to Search handles whistling, with no numbers;
  - MLEnd Hums & Whistles is the only public whistle QbH-style data (8 songs, no references).
  - **I found no paper reporting whistle → real-recording retrieval numbers.**
- **PESTO** (Riou et al., ISMIR 2023; TISMIR v2 2025), a self-supervised pitch estimator trained with a **transposition-equivariance** objective, needs no labels. https://arxiv.org/abs/2309.02265, https://arxiv.org/abs/2508.01488, https://github.com/SonyCSLParis/pesto.
- **What this means for us:** a pitch tracker trained self-supervised on unlabeled whistles (MLEnd, all people, no song labels needed) is a clean new idea for the tracker side. On the model side, D-016 showed that 8 songs of whistle pairs teach songs, not whistling. Whistle robustness needs **many melodies**, which contour-level synthetic whistles over thousands of real-song or Essen contours can provide.

---

## 8. The novelty gap

The claims below are "to the best of our search (Sep 2026)". Each should be re-checked before submission.

1. **No open, reproducible benchmark for hum → full real recording retrieval.**
   - CHAD's Jang Real and MTG-QBH use internal, unreleased imposters.
   - ByteHum's annotated originals are not verifiably public.
   - Google, SoundHound and ACRCloud publish no real-audio numbers.
   - Our fixed CHAD song split (145 test songs, full songs), with open-licence FMA distractors and per-query-type reporting (hum / sing / whistle), would be the first *documented* protocol.
   - Caveat: Tier Y audio cannot be released (D-019). What can be released is IDs, timestamps, the split and code. Releasing derived features needs the terms review.
2. **Nobody has quantified the MIDI → real transfer gap for a key-invariant contour model, or decomposed it.** Our numbers already show the drop (0.99 on MIDI, 0.58 on rendered hums, 0.30 on real hums vs real songs). A decomposition of that drop into coverage (preview vs full), window/chunking, reference extraction and embedding error is a publishable analysis on its own. D-018/D-019 already did the coverage part.
3. **Cover-derived contour pairs.** CHAD and ByteHum used covers in a CQT / CREPE-activation / raw-audio domain with CNNs. **No one has trained an explicitly key-invariant F0-contour encoder on cover↔original contour pairs, or tested whether cover pairs can replace real hum pairs in the contour domain.**
4. **Segment-level weak supervision (CLEWS-style reductions) has not been applied to QbH**, where the hummed-segment timestamps are noisy by nature.
5. **Whistle QbH against real recordings has no published numbers.** Nor does a self-supervised (PESTO-style) whistle tracker.
6. Salience vs hard-F0 input, in a controlled ablation for QbH against real audio. CHAD uses salience but never ablates it against argmax F0.

The strongest single paper claim would be: "an open real-audio QbH benchmark plus a contour-domain model that closes most of the MIDI→real gap using only cover/song-derived pairs (no extra hum recordings), with the gap decomposed". Items 1, 2 and 3 together.

---

## 9. Ranked plan: 3–5 highest-leverage experiments

Ordering is expected gain per GPU-hour, weighted by what it adds to the paper. The gains are **estimates, not measurements**; each experiment has a pre-registered stop rule. All of them use the fixed CHAD split (test songs never trained on or used for selection) and report CHAD test at `charts_fma`, plus MTG-QBH, MLEnd and MIR-QBSH +2000 regression.

### E0. Error decomposition on CHAD test (diagnostic, ~1 GPU-h, no training), **do first**
CHAD gives the hummed interval in each original, so we can build oracles:
- **(a) Oracle window:** embed exactly the annotated reference interval and rank it against the normal grid windows of all ~5.3k songs. The gap between grid and oracle is the **chunking cost**.
- **(b) Contour agreement:** slope-constrained DTW MAE (D-008) between the hum contour and the reference contour at the annotated interval. Bucket queries by MAE. If the failures sit at high MAE, the problem is the **reference contour** (backing vocals, octave errors, instrumental melody); if the failures sit at low MAE, it is the **embedding**.
- **(c)** Top-1 by query duration and voiced fraction, and by reference voiced fraction.
- **Why first:** it decides whether E1 or E2 matters more. Every other number in this plan depends on it.
- **Expected outcome:** a table that splits the 0.53 top-1 loss (targets-only 0.47 → ideal 1.0) into parts.

### E1. Multi-window queries + sequence-consistent reranking (no retraining; ~2–4 GPU-h)
- Index 4–6 s windows with a 1 s hop, in addition to the 10 s windows.
- Split each query into overlapping 4–6 s sub-windows. Retrieve the top-K songs per sub-window by ANN.
- Rerank the top-50 songs by **temporal consistency**: the sub-window hits must line up on a monotone t_ref ≈ a·t_q + b with a ∈ [0.5, 2] (Huber fit or DP over the similarity matrix). Also try CHAD's max-correlation variant.
- Fuse with a **key-invariant subsequence DTW (plus Stasiak tune-following)** on contours in the same top-50.
- **Why:** every strong real-audio system does sequence-level matching (CHAD, ByteCover3, SoundPatrol, Jeon 2009); we do none. It also fixes the query-length mismatch behind MTG-QBH (26.8 s queries vs 10 s windows) and D-017's length drop.
- **Expected:** MTG-QBH top-10 0.14 → 0.30–0.45 (largest effect, since its queries are long); CHAD test top-10 +0.04–0.10. **Low risk; not a paper headline, but needed for every later number.**

### E2. Scale song diversity on the reference side (headline experiment; ~6–10 GPU-h plus data)
This is the fix for the D-019 failure mode (116 songs memorised). There are two data sources, and they can run in parallel:
- **E2a, license-clean and needs no new approval: self-supervised real-song contour pairs (Jeong stage 1 + Google-style resynthesis).**
  - Take the ~2.8k indexed FMA songs already cached (grow toward 20k+ from `fma_full`).
  - Anchor: the htdemucs → RMVPE contour of a song window. Positive: the *same window* passed through `humanize` + hum augmentation + octave/dropout noise, optionally rendered as a tone and re-tracked by RMVPE.
  - A second positive: the same window extracted by a *different* route (RMVPE on the mix, or FCPE on the vocals), so the model learns extractor-error invariance.
  - Mix with HumTrans pairs. Start from the D-012 checkpoint.
- **E2b, highest expected gain but needs Luigi's explicit approval (new Tier Y downloads under the D-019 rules): CHAD cover part in the contour domain.**
  - 5,494 originals and 31,630 covers with public timestamps (github.com/amanteur/CHAD).
  - Build cover-vocal-contour ↔ original-vocal-contour pairs, and **drop every song whose title matches a CHAD val/test, MTG-QBH or MLEnd target** before training.
- **Loss:** CLEWS-style positives (best of several reference windows within ±3 s of the timestamp) and hardest-window negatives, so noisy timestamps and backing-vocal windows do not poison the positives.
- **Why:** CHAD's C-only model reached 0.62 top-10 at 90k songs; song count is the variable we have never moved; D-014/D-017 show extraction is not the bottleneck.
- **Expected:** CHAD test top-10 0.47 → 0.55–0.65 at ~5.3k (E2b > E2a). MIR-QBSH must not regress by more than 0.01 top-10.
- **Paper claims:** novelty items 3 and 4.

### E3. Salience input instead of hard F0 (~3–4 GPU-h; ablation that doubles as a gain)
- Feed the model RMVPE's 360-bin salience, cropped to ±18 st around the clip's median voiced pitch (CHAD used 3 octaves around the mean), on both sides.
- Keep key invariance by centring the crop. Keep the voicing channel.
- Optional variant: octave-folded salience (Salamon 2013), which removes octave errors by construction.
- Train the same way as the best E2 variant; ablate against it.
- **Why:** the reference-side errors (a backing vocal a third above, octave jumps) are irrecoverable once an argmax is taken; salience keeps the alternatives.
- **Expected:** +0.03–0.08 top-10 on CHAD test. **Uncertain:** it may hurt hums, where the argmax is reliable. The ablation settles it, and item 6 is a claim either way.

### E4. Whistles through many-melody synthetic whistle contours + a self-supervised whistle tracker (~4 GPU-h)
- **(a)** Generate whistle-like contours (interval compression 0.55–0.85, octave folding, dropouts, D-013 statistics) from **thousands** of real-song contours (E2a) and Essen melodies, not from 8 MLEnd songs.
- **(b)** Optionally fine-tune PESTO self-supervised on unlabeled MLEnd whistle audio (all people; no song labels are used, so no test leakage on songs, but keep the D-015 performer split for the evaluation).
- Evaluate only with the D-016 leave-songs-out protocol and MLEnd full songs.
- **Why:** D-016 showed that song familiarity, not whistling skill, drove the D-015 gain. Whistling skill has to come from melody diversity.
- **Expected:** unseen-song whistle → hums top-1 0.63 → 0.68–0.72 (8-way). Real-recording whistle numbers would be the first published (novelty item 5), even if low.

### Not recommended now (and why)
- **MERT/MuQ fine-tuning as the main encoder:** D-011 already failed on melody; the evidence for these models is recording identity, not melody. Keep only as a frozen reranking ablation after E1–E3.
- **A better melody extractor:** D-014 shows htdemucs → RMVPE is near published SOTA for vocals, and extraction costs only about 0.03–0.06.
- **More CHAD hum pairs from scratch:** the D-019 negative result. The already-configured D-021 fine-tune is the right minimal test of real hum pairs; run it, but do not expect it to scale song diversity.

---

## 10. Reference list (verified unless marked)

1. Frank, C. (2020). The Machine Learning Behind Hum to Search. Google Research blog. https://research.google/blog/the-machine-learning-behind-hum-to-search/
2. Agüera y Arcas, B. et al. (2017). Now Playing: Continuous low-power music recognition. arXiv:1711.10958.
3. Rehmeyer, J. (2007). Tracking Down a Tune (Midomi). MIT Technology Review. https://www.technologyreview.com/2007/04/10/226000/tracking-down-a-tune/
4. ACRCloud (2016). What is Query by Humming? https://www.acrcloud.com/blog/what-is-query-by-humming/
5. Salamon, J., Serrà, J., Gómez, E. (2013). Tonal representations for music retrieval: from version identification to query-by-humming. IJMIR 2:45–58. doi:10.1007/s13735-012-0026-0
6. Amatov, A. et al. (2023). A Semi-Supervised Deep Learning Approach to Dataset Collection for Query-by-Humming Task (CHAD). ISMIR 2023. arXiv:2312.01092
7. Du, X. et al. (2024). ByteHum: Fast and Accurate Query-by-Humming in the Wild. ICASSP 2024, 1111–1115. doi:10.1109/ICASSP48485.2024.10448117 [PARTIAL: abstract only]
8. Pham, B. L. et al. (2024). An approach to hummed-tune and song sequences matching. arXiv:2410.20352
9. Kung, Y.-H., Chen, K.-Y., Ding, J.-J. (2026). Improved Query by Humming Using Conformer-based Network with Harmonic-Aware Mechanism. ISCAS 2026. doi:10.1109/ISCAS66217.2026.11562747 [PARTIAL: metadata only]
10. Jeon, W., Ma, C., Cheng, Y. M. (2009). An efficient signal-matching approach to melody indexing and search using continuous pitch contours and wavelets. ISMIR 2009. https://ismir2009.ismir.net/proceedings/PS4-18.pdf
11. Jeong, D. qbh_project (code, no paper). https://github.com/jdasam/qbh_project
12. Stasiak, B. (2014). Follow That Tune – Adaptive Approach to DTW-based Query-by-Humming System. Archives of Acoustics 39(4). http://ics.p.lodz.pl/~basta/pre-prints/Stasiak_AoA_2014.pdf
13. Yu, Z. et al. (2019/2020). Learning a representation for cover song identification using CNN (CQTNet). arXiv:1911.00334 [numbers taken from ref. 15]
14. Du, X. et al. (2020). ByteCover: Cover song identification via multi-loss training. arXiv:2010.14022; ByteCover2, ICASSP 2022, doi:10.1109/ICASSP43922.2022.9747630
15. Du, X. et al. (2023). ByteCover3: Accurate Cover Song Identification on Short Queries. ICASSP 2023. arXiv:2303.11692
16. Liu, F. et al. (2023). CoverHunter: Cover Song Identification with Refined Attention and Alignments. ICME 2023. arXiv:2306.09025
17. Xun, J. et al. (2023). DisCover: Disentangled Music Representation Learning for Cover Song Identification. SIGIR 2023. arXiv:2307.09775 [PARTIAL: numbers from a summary]
18. Serrà, J., Araz, R. O., Bogdanov, D., Mitsufuji, Y. (2025). Supervised Contrastive Learning from Weakly-Labeled Audio Segments for Musical Version Matching (CLEWS). ICML 2025, PMLR 267. arXiv:2502.16936
19. Doras, G., Peeters, G. (2019). Cover detection using dominant melody embeddings. ISMIR 2019. https://archives.ismir.net/ismir2019/paper/000010.pdf
20. Wei, H. et al. (2023). RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music. Interspeech 2023. arXiv:2306.15412
21. Jing et al. (2025). A Joint Network for Singing Melody Extraction from Polyphonic Music with Attention Aggregation and Self-Consistency Training. Interspeech 2025. https://www.isca-archive.org/interspeech_2025/jing25_interspeech.html
22. SpectMamba: A Mamba-based Network for Semi-supervised Singing Melody Extraction Using Confidence Binary Regularization (2025). arXiv:2505.08681
23. Li, Y. et al. (2024). MERT. ICLR 2024. arXiv:2306.00107
24. Zhu, H. et al. (2025). MuQ. arXiv:2501.01108
25. Wu, S. et al. (2025). CLaMP 3. arXiv:2502.10362
26. Singh, S. et al. (2025). Robust Neural Audio Fingerprinting using Music Foundation Models. arXiv:2511.05399
27. Greif, J. et al. (2024). Improving Query-By-Vocal Imitation with Contrastive Learning and Audio Pretraining. arXiv:2408.11638
28. Seed-VC: Zero-shot Voice Conversion with Diffusion Transformers. arXiv:2411.09943
29. Vevo2: A Unified and Controllable Framework for Speech and Singing Voice Generation. arXiv:2508.16332
30. Riou, A. et al. PESTO: Pitch Estimation with Self-supervised Transposition-equivariant Objective. arXiv:2309.02265; v2 (TISMIR) arXiv:2508.01488
31. Nilsson, M. et al. (2008). Human whistle detection and frequency estimation. CISP 2008. doi:10.1109/CISP.2008.415
32. Shen, H.-C., Lee, C. (2007). Whistle for music. Multimedia Tools and Applications 35(3). doi:10.1007/s11042-007-0128-5 [PARTIAL]
33. Raffel, C., Ellis, D. P. W. (2016). Pruning subsequence search with attention-based embedding. ICASSP 2016. [not opened]
34. Kotsifakos, A. et al. (2015). Embedding-based subsequence matching with gaps-range tolerances: a query-by-humming application. VLDB Journal. [not opened]
35. Liu, S. et al. (2023). HumTrans: A Novel Open-Source Dataset for Humming Melody Transcription. arXiv:2309.09623
36. MIREX 2025 Cover Song Identification task page. https://music-ir.org/mirex/wiki/2025:Cover_Song_Identification
