# D-040 · Generate, train, and eval whistle and wrong-lyrics queries

Not trained in this checkout. There is no GPU and no `/lambda/nfs/hum2song-data` here. The CPU smoke that did run is `docs/paper/results/d040/smoke.json` (two files, one seed each). From the repo root, with the training venv from `training/setup_gpu.sh`:

```bash
export H2S_DATA=/lambda/nfs/hum2song-data
```

Song audio on that box is the FMA tracks already there (about 12k). YouTube audio was deleted. This script does not download anything.

## (a) Generate

Vocal F0 already on disk. Whistles are re-tracked with the spectral peak. Wrong-lyrics singing is re-tracked with RMVPE (`$H2S_DATA/fig_contours/rmvpe.pt`), which is the tracker real sung queries use. Each kept window is rendered once as a whistle and once as nonsense syllables. Songs whose track file is missing are skipped. Evaluation-target titles are dropped. A song id is train, val, or test by `zlib.crc32(song_id) % 10` (0 test, 1 val, otherwise train). Every manifest row has `origin=synthetic`.

```bash
python training/scripts/generate_query_synth.py \
  --config configs/train_contour_qsynth.yaml \
  --data-root "$H2S_DATA" \
  --sing-tracker rmvpe \
  --per-song 4
```

Writes `$H2S_DATA/query_synth/manifest.jsonl` plus `wav/`, `f0/`, `refs/`, and `meta/`. A sidecar already in `meta/` is reused, so a stopped run can be started again with the same command.

If `rmvpe.pt` is missing, the CPU fallback is `--sing-tracker cepstrum`. That tracker is what the smoke test uses. It is not what real sung queries use.

Optional MIDI, not part of the default launch (ids become `midi:<stem>`):

```bash
python training/scripts/generate_query_synth.py \
  --config configs/train_contour_qsynth.yaml \
  --data-root "$H2S_DATA" \
  --sing-tracker rmvpe \
  --per-song 4 \
  --midi-dir raw/essen_midi/deutschl
```

`--workers` stays at 1. Raising it forks one process per worker and each worker loads RMVPE on first use.

Upper bound, not a count: about 12k FMA songs × 4 windows × 2 styles is on the order of 96k clips, and only songs with at least 3 s of voiced contour are kept.

## (b) Train

Fine-tunes `ckpt/contour_e2b_s0/best.pt`. Writes `$H2S_DATA/ckpt/contour_qsynth_s0/{best,last}.pt`. `best.pt` is chosen on HumTrans **val** shift+7 top-1 only. The synthetic test split is not the select metric.

```bash
python training/scripts/train_contour.py --config configs/train_contour_qsynth.yaml
```

## (c) Eval

In-memory contour ranking. Does not read or write the live index. The JSON field `replaces_e2b` is false. Sections report top-1, top-10, and MRR.

```bash
python training/scripts/eval_query_synth.py \
  --config configs/train_contour_qsynth.yaml \
  --data-root "$H2S_DATA" \
  --ckpt "$H2S_DATA/ckpt/contour_qsynth_s0/best.pt" \
  --out "$H2S_DATA/results/d040/eval.json"
```

- `sets.synthetic_holdout` is the test split of the manifest, against up to 24 windows of 8 s from every song in that manifest. It is labeled synthetic. It is not a recording of a person.
- `sets.hum_baseline` is HumTrans test and MIR-QBSH `mir48_anywhere` (MIDI references), from `pairs_real.jsonl` and the F0 npz files. This is the small real-hum baseline.
- `sets.mlend_whistle` and `sets.mtgqbh_sing` write `status: blocked` until owned audio is on disk (below).

## Do not replace E2b

Do not run any of these. They would point search at this checkpoint or overwrite the live vectors.

```bash
# export H2S_CKPT=$H2S_DATA/ckpt/contour_qsynth_s0/best.pt
# python training/scripts/build_library.py ...
# python training/scripts/index_tracks.py ...
# python training/scripts/index_windows.py ...
```

D-038's live E2b snapshot stays the comparison if a later decision writes an adopt gate: CHAD test 0.483 / 0.657, MTG-QBH sung 0.655 / 0.791. This run has no adopt gate.

## Real whistle and sung targets

MLEnd ships the query clips, not the eight recordings. Those titles are not in the FMA tracks. The YouTube copies were deleted, and this script does not fetch them. To score whistle-against-song, place a recording you already own at:

```text
$H2S_DATA/owned/mlend_targets/<label>.wav
```

Labels: `Potter` (Hedwig's Theme), `StarWars` (Imperial March), `Panther` (Pink Panther), `Rain` (Singin' in the Rain), `Hakuna` (Hakuna Matata), `Mamma` (Mamma Mia), `Showman` (This Is Me), `Frozen` (Let It Go). `eval_mlend.py` scores whistle-against-hum and does not need these files. That is a different protocol.

MTG-QBH (Zenodo 1290712) has the 118 sung queries and does not include the collection audio. To score them, place a recording you already own at:

```text
$H2S_DATA/owned/mtgqbh/<song_id>.wav
```

`<song_id>` is the id from `pairs_real.jsonl` with characters outside `[A-Za-z0-9._-]` replaced by `_`. If the wav exists, eval runs RMVPE on that file as it is.

## CPU smoke (no data, no GPU)

Already run on 2026-10-08. The wavs are not committed. The report is `docs/paper/results/d040/smoke.json`.

```bash
python training/scripts/smoke_query_synth.py --out /tmp/qsynth-smoke
```
