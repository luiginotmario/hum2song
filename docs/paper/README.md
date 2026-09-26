# Paper material

Every measured number in `SIGNALS.md` points to a file in this folder.

| path | what |
|---|---|
| `SIGNALS.md` | signal-physics definitions and measured numbers (§S1–§S8) |
| `REFERENCES_and_CAPTION.md` | contour figure: methods, numbers (old vs new), draft caption, references |
| `WHISTLE_F0_VALIDATION.md` | whistle F0 validation on all MLEnd whistles and the downstream retrieval check |
| `scripts/contour_pipeline.py` | shared contour cleaning, spectral-peak whistle track, length-normalized slope-constrained DTW (D-008) |
| `scripts/contour_figure.py` | contour figure (`extract`, `plot`; MLEnd or `--custom-dir` recordings) |
| `scripts/validate_whistle_f0.py` | whistle F0 validation (`extract`, `analyze`, `downstream`, `synth`, `humpeak`) |
| `scripts/rmvpe_rvc.py` | RMVPE inference from RVC, CUDA-graph import stubbed out (needs the `rmvpe.pt` checkpoint) |
| `figures/` | `fig_contours_potter.{png,pdf}`, `fig_whistle_f0_validation.{png,pdf}` |
| `results/` | JSON/CSV outputs of the scripts |
| `results/superseded/` | first-version outputs (RMVPE-fix whistles, DTW with silent fallback), kept for the old-vs-new comparison |

Reproduce (CPU only; MLEnd Hums and Whistles in `$R`, run inside `scripts/`):
```
python contour_figure.py extract --mlend-root $R --song Potter --out f0cache_potter --model rmvpe.pt --procs 3
# same for StarWars and Hakuna (null songs)
python validate_whistle_f0.py extract --mlend-root $R --out wcache --model rmvpe.pt --procs 8
python validate_whistle_f0.py analyze --mlend-root $R --cache wcache --outdir ../results --peak-thr 6
python validate_whistle_f0.py downstream --cache wcache --hum-cache f0cache_potter --outdir ../results
python validate_whistle_f0.py downstream --cache wcache --hum-cache f0cache_StarWars --song StarWars \
    --null-songs Potter,Hakuna --peak-rules prom:50,tonal_db:6 --outdir ../results
python validate_whistle_f0.py downstream --cache wcache --hum-cache f0cache_Hakuna --song Hakuna \
    --null-songs Potter,StarWars --peak-rules prom:50,tonal_db:6 --outdir ../results
python contour_figure.py plot --cache f0cache_potter --whistle-cache wcache --whistle-f0 peak \
    --outdir ../figures --tag potter --null-songs StarWars,Hakuna --null-cache-pattern "f0cache_{song}" --interpreter 213
python contour_figure.py plot --cache f0cache_potter --whistle-cache wcache --whistle-f0 rmvpe_fix \
    --outdir ../results --tag potter_rmvpefix --null-songs StarWars,Hakuna --null-cache-pattern "f0cache_{song}" --interpreter 213
```
`plot` writes the figure and its `_stats.json` / `_pairs.json` into `--outdir`; in this folder the figures sit in `figures/` and the JSON in `results/`.
