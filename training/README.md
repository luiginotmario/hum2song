# Phase 1 training

Stage A fine-tunes `m-a-p/MERT-v1-95M` with symmetric InfoNCE on real hums. Stage B/C (synthetic hums, Demucs, CREPE, aligned pairs) is a stub in `hum2song.synth`. The server and the Jev/LLM layer are not in this phase.

Data root on the Lambda box: `/lambda/nfs/hum2song-data` (`H2S_DATA`). Paths inside `pairs_real.jsonl` are relative to that root.

## One-time setup

```bash
bash training/setup_gpu.sh
source .venv/bin/activate
export H2S_DATA=/lambda/nfs/hum2song-data
# optional
export WANDB_API_KEY=...
export WANDB_PROJECT=hum2song
# MLEnd only; the downloader skips it when these are unset
export KAGGLE_USERNAME=...
export KAGGLE_KEY=...
```

## Commands

Dry-run the downloads (no network writes):

```bash
python training/scripts/download_datasets.py \
  --datasets mirqbsh,humtrans,mtgqbh,mlend \
  --out "$H2S_DATA" \
  --dry-run
```

Download, render MIDI references, and write `$H2S_DATA/pairs_real.jsonl` plus `songs.jsonl`. Re-running is safe: finished files are skipped when the checksum matches.

```bash
python training/scripts/download_datasets.py \
  --datasets mirqbsh,humtrans,mtgqbh,mlend \
  --out "$H2S_DATA"
```

Tiny manifest while developing (`--max-items` still downloads full archives, then keeps a balanced subset):

```bash
python training/scripts/download_datasets.py \
  --datasets mirqbsh,humtrans \
  --out "$H2S_DATA" \
  --max-items 32
```

CPU config check, no MERT download. Exits 0 when the manifest is missing:

```bash
python training/scripts/train.py --config configs/train_tiny.yaml --dry-run
```

One optimizer step on four real clips (downloads MERT on first use):

```bash
python training/scripts/train.py \
  --config configs/train_stage_a.yaml \
  --dry-run \
  --limit 4
```

Stage A training. Checkpoints go to `$H2S_DATA/ckpt/stage_a/`. Logging goes to W&B only when `WANDB_API_KEY` is set.

```bash
python training/scripts/train.py --config configs/train_stage_a.yaml
```

Resume:

```bash
python training/scripts/train.py \
  --config configs/train_stage_a.yaml \
  --init "$H2S_DATA/ckpt/stage_a/last.pt"
```

Eval on MIR-QBSH. The JSON report includes top-1, top-10, and MRR next to the CHAD top-10 of **0.921**.

```bash
python training/scripts/eval.py \
  --config configs/eval.yaml \
  --ckpt "$H2S_DATA/ckpt/stage_a/last.pt" \
  --sets mirqbsh \
  --out "$H2S_DATA/runs/stage_a/report.json"
```

CPU tests (no GPU, no MERT weights):

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```

## Assumptions

- The shared manifest is `pairs_real.jsonl`. Each row has `query_type` (`hum`, `whistle`, or `sing`) and the same value in `qtype` (the SPEC name), plus `song_path` and `title`. `song_path` is the rendered MIDI reference when the dataset has no song audio.
- Matching is melody only. Sung queries may use the right lyrics, the wrong words, or nonsense. `hum2song.synth.augment_lyric_agnostic` is the Stage 2 stub that will keep a melody and replace the words. It is not implemented.
- MIR-QBSH wav stem (`00001`–`00048`) is the song id and matches `midiFile/`. `waveFile/year2006a` is the supplementary English-song session (`2006a-MIR補錄英文歌` in `yearDirInfo.txt`) and is labeled `sing`. Every other MIR-QBSH clip is `hum`, because the archive has no per-clip hum/sing flag.
- Songs that have a MIR-QBSH `sing` clip (stems `00001`–`00010`) are hash-split so those sung clips can train. Hum clips of the same song share that split. The other 38 MIR-QBSH songs stay in `test`. MTG-QBH and all 8 MLEnd songs stay in `test`. HumTrans keeps its official split, reconciled so one composition id cannot land in two splits. There is no title table in HumTrans, so cross-dataset title dedupe only runs when a title is present.
- Eval JSON reports top-1, top-3, top-10, and MRR for the set and again under `per_query_type` for hum, whistle, and sing (count 0 when that type is absent). `per_qtype` is the same object.
- MTG-QBH queries are `sing`. The song id is the class label (the piece), not the collection recording id. Those commercial tracks are not in the archive, so the pairs have no `song_path` and eval will say references are missing until song audio is added.
- MLEnd comes from Kaggle (`jesusrequena/mlend-hums-and-whistles`) using `KAGGLE_USERNAME` and `KAGGLE_KEY`. Missing credentials skip that dataset and leave the process successful.
- MIDI references are rendered with a harmonic series at 24 kHz. That is the closed-set MIR-QBSH reference, not Stage 2 Demucs/CREPE synthesis. CHAD's 0.921 used about 2600 MIDI distractors; the default eval is the 48-song closed set, so `comparable_to_chad` is false until `--distractors` supplies a real impostor pool.
- Stage A trains on rows with `split=train` (HumTrans). The song tower sees `song_path` when it exists, otherwise another query of the same song.
- Codec augmentation runs only when `ffmpeg` is on `PATH`. The other augmentations are in-process.
- Loudness is peak normalization to 0.95, not EBU R128.
- Checkpoints store the git SHA and the manifest sha256. W&B is off unless `WANDB_API_KEY` is set.
