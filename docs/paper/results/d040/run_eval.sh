#!/bin/bash
# usage: run_eval.sh <ckpt> <tag>
cd /lambda/nfs/hum2song-data/hum2song
export H2S_DATA=/lambda/nfs/hum2song-data PYTHONPATH=/lambda/nfs/hum2song-data/hum2song/training/src
PY=/lambda/nfs/hum2song-data/venv/h2s/bin/python
R=$H2S_DATA/results/d040
$PY training/scripts/eval_query_synth.py --config configs/train_contour_qsynth.yaml --data-root $H2S_DATA --ckpt $1 --out $R/eval$2.json > $R/eval$2.log 2>&1
echo qsynth_exit=$? >> $R/eval$2.log
$PY training/scripts/eval_mlend.py --ckpt $1 --split test --song-holdout --out $R/eval_mlend$2.json > $R/eval_mlend$2.log 2>&1
echo mlend_exit=$? >> $R/eval_mlend$2.log
