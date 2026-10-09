#!/bin/bash
D=/lambda/nfs/hum2song-data; R=$D/results/d040; C=$D/ckpt/contour_qsynth_s0
cd $D/hum2song; export PYTHONPATH=$D/hum2song/training/src
$R/run_eval.sh $C/best.pt ''
$R/run_eval.sh $C/last.pt _last
for t in e2b:$D/ckpt/contour_e2b_s0/best.pt last:$C/last.pt; do
  $D/venv/h2s/bin/python $R/per_style_eval.py ${t#*:} $R/eval_style_${t%%:*}.json > $R/eval_style_${t%%:*}.log 2>&1
done
echo ALLDONE > $R/evals_done
