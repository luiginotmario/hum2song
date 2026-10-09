#!/bin/bash
cd /lambda/nfs/hum2song-data/hum2song
export H2S_DATA=/lambda/nfs/hum2song-data PYTHONPATH=/lambda/nfs/hum2song-data/hum2song/training/src
exec /lambda/nfs/hum2song-data/venv/h2s/bin/python training/scripts/train_contour.py --config configs/train_contour_qsynth.yaml "$@" >> $H2S_DATA/results/d040/train.log 2>&1 < /dev/null
