#!/bin/bash
# D-042 chain: train (no MLEnd data), then eval D-042 best+last / D-041 best / D-040 last / E2b on identical sets.
D=/lambda/nfs/hum2song-data; R=$D/results/d042; REPO=$D/hum2song; PY=$D/venv/h2s/bin/python
cd $REPO; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True H2S_DATA=$D PYTHONPATH=$REPO/training/src
echo "train start $(date)" >> $R/chain.log
$PY training/scripts/train_contour.py --config configs/train_contour_qsynth_d042.yaml >> $R/train.log 2>&1 < /dev/null
echo "train exit=$? $(date)" >> $R/chain.log
C=$D/ckpt/contour_qsynth_d042_s0
for t in d042_best:$C/best.pt d042_last:$C/last.pt d041_best:$D/ckpt/contour_qsynth_d041_s0/best.pt d040_last:$D/ckpt/contour_qsynth_s0/last.pt e2b:$D/ckpt/contour_e2b_s0/best.pt; do
  tag=${t%%:*}; ck=${t#*:}
  $PY training/scripts/eval_query_synth.py --config configs/train_contour_qsynth.yaml --data-root $D --ckpt $ck --out $R/eval_$tag.json > $R/eval_$tag.log 2>&1
  echo "eval $tag exit=$? $(date)" >> $R/chain.log
  $PY $D/results/d040/per_style_eval.py $ck $R/eval_style_$tag.json > $R/eval_style_$tag.log 2>&1
  echo "style $tag exit=$? $(date)" >> $R/chain.log
  $PY training/scripts/eval_mlend.py --ckpt $ck --split test --song-holdout --out $R/eval_mlend_$tag.json > $R/eval_mlend_$tag.log 2>&1
  echo "mlend $tag exit=$? $(date)" >> $R/chain.log
done
echo ALLDONE >> $R/chain.log
