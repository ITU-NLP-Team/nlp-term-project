#!/bin/bash
#SBATCH -J cf-mega
#SBATCH -A cip001
#SBATCH -p a100x4q,gpu2dq,a100q
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --time=24:00:00
#SBATCH -o logs/mega.out
# Finish everything after the pools: train-side selection, the selected pool, pointwise
# reranking of its new documents, listwise reranking, and the reports. Nodes are allocated
# exclusively, so all visible GPUs are used (one part per GPU); everything resumes from the
# stage cache.
source slurm/env.sh
N=$(nvidia-smi -L | wc -l)
g() { echo $(( $1 % N )); }
python -m chronofuse.bench select
python -m chronofuse.bench stage --split train dev --subtrack 1a 1b --stage pool_sel
R=$(python -c "import json;d=json.load(open('results/metrics/pool_selection.json'));print(sorted({v['best_rerank'].replace('+pool','') for v in d.values() if 'best_rerank' in v})[0])")
echo "reranker: $R  gpus: $N"
(export CUDA_VISIBLE_DEVICES=$(g 0)
 python -m chronofuse.bench stage --split dev --subtrack 1b --stage $R
 python -m chronofuse.bench stage --split dev --subtrack 1b --stage "$R@sel") > logs/mega_g0.out 2>&1 &
CUDA_VISIBLE_DEVICES=$(g 1) python -m chronofuse.bench stage --split train --subtrack 1a \
  --stage "$R@sel" > logs/mega_g1.out 2>&1 &
CUDA_VISIBLE_DEVICES=$(g 2) python -m chronofuse.bench stage --split dev --subtrack 1a \
  --stage "$R@sel" > logs/mega_g2.out 2>&1 &
wait
if [ "$N" -ge 3 ]; then
  CUDA_VISIBLE_DEVICES=0 python -m chronofuse.bench stage --split train --subtrack 1a --stage listwise > logs/mega_lw0.out 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python -m chronofuse.bench stage --split dev --subtrack 1a --stage listwise > logs/mega_lw1.out 2>&1 &
  CUDA_VISIBLE_DEVICES=2 python -m chronofuse.bench stage --split dev --subtrack 1b --stage listwise > logs/mega_lw2.out 2>&1 &
  wait
else
  python -m chronofuse.bench stage --split train dev --subtrack 1a --stage listwise > logs/mega_lw0.out 2>&1
  python -m chronofuse.bench stage --split dev --subtrack 1b --stage listwise > logs/mega_lw2.out 2>&1
fi
python -m chronofuse.bench report --split dev
python -m chronofuse.bench report --split train
