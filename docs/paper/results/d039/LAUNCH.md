# D-039 · Start the hum-to-song fine-tune on the Lambda A100

Not trained in the repo checkout. Data and the GPU are on the box. From the repo root, with the training venv from `training/setup_gpu.sh`:

```bash
export H2S_DATA=/lambda/nfs/hum2song-data
export H2S_DB=postgresql://...   # live library URL (the API's H2S_DATABASE_URL)
python training/scripts/train_contour.py --config configs/train_contour_hum.yaml
```

Writes `$H2S_DATA/ckpt/contour_hum_s0/best.pt`. Same file format the search server already loads (`model`, `config`, `step`, `metrics`). No loader change. Do not point `H2S_CKPT` at it until the gate below passes.

`best.pt` is chosen on CHAD **val** top-1 only. CHAD test and MTG-QBH are not used to pick it.

## Re-index, then eval

`eval_live_search.py` embeds the query with `--ckpt` and reads song vectors from the database. A new checkpoint against the E2b index is not a comparison. Stop the API first. Chunk re-insert deletes windows (cascade), so windows come last.

```bash
CKPT=$H2S_DATA/ckpt/contour_hum_s0/best.pt

for name in fma_full_3k fma_full_extra_3k fma_electronic_1k fma_full_extra2_3k fma_electronic_extra_2k; do
  python training/scripts/build_library.py --data-root "$H2S_DATA" --name "$name" \
    --db "$H2S_DB" --ckpt "$CKPT" --force
done

python training/scripts/index_tracks.py --data-root "$H2S_DATA" \
  --names youtube_v1 youtube_charts_v1 previews_v1 \
  --db "$H2S_DB" --ckpt "$CKPT" --force

python training/scripts/index_windows.py --data-root "$H2S_DATA" \
  --db "$H2S_DB" --ckpt "$CKPT" --force

python training/scripts/eval_live_search.py --data-root "$H2S_DATA" --db "$H2S_DB" \
  --ckpt "$CKPT" --max-queries 300 \
  --out "$H2S_DATA/results/d039/live.json"
```

## Does it replace E2b?

Read `sets.chad_test.windows` and `sets.mtgqbh_sing.windows` in that JSON. The report `model` field must be `contour_hum_s0/best.pt`.

Replace E2b only if **all four** are strictly above the D-038 live-library snapshot (E2b, ~13k songs, same script, windows mode):

| Set | E2b top-1 | E2b top-10 |
|---|---:|---:|
| CHAD test (300) | 0.483 | 0.657 |
| MTG-QBH sung (110) | 0.655 | 0.791 |

If any of the four is not strictly higher, keep E2b. Re-index with `ckpt/contour_e2b_s0/best.pt` (same three commands) and start the API with `H2S_CKPT` on that file. A tie is not a win. Whistle is not part of this gate.
