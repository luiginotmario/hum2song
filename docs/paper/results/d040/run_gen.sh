#!/bin/bash
cd /lambda/nfs/hum2song-data/hum2song
export H2S_DATA=/lambda/nfs/hum2song-data PYTHONPATH=/lambda/nfs/hum2song-data/hum2song/training/src OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
echo '--- restart with --workers 8 (single worker was ~42 songs/min); sidecars reused' >> $H2S_DATA/results/d040/gen.log
exec /lambda/nfs/hum2song-data/venv/h2s/bin/python training/scripts/generate_query_synth.py --config configs/train_contour_qsynth.yaml --data-root $H2S_DATA --sing-tracker rmvpe --per-song 4 --workers 8 >> $H2S_DATA/results/d040/gen.log 2>&1 < /dev/null
