#!/bin/bash
#SBATCH -J cf-final
#SBATCH -A cip001
#SBATCH -p gpu2dq,a100x4q,a100q
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --time=24:00:00
#SBATCH -o logs/final.out
# Train-side selection (pool channels, reranker), the selected pool, its pointwise reranking
# (new documents only), listwise reranking, and the dev/train reports. Uses every visible GPU
# (one split per GPU) when there are at least three; otherwise runs the splits one by one.
source slurm/env.sh
python -m chronofuse.bench select
python -m chronofuse.bench stage --split train dev --subtrack 1a 1b --stage pool_sel
R=$(python -c "import json;d=json.load(open('results/metrics/pool_selection.json'));print(' '.join(sorted({v['best_rerank'].replace('+pool','') for v in d.values() if 'best_rerank' in v})))")
N=$(nvidia-smi -L | wc -l)
PARTS=("train 1a" "dev 1a" "dev 1b")
run_parts() {  # $1 = stage name
  if [ "$N" -ge 3 ]; then
    i=0
    for p in "${PARTS[@]}"; do
      set -- $p
      CUDA_VISIBLE_DEVICES=$i python -m chronofuse.bench stage --split $1 --subtrack $2 --stage "$STAGE" &
      i=$((i + 1))
    done
    wait
  else
    for p in "${PARTS[@]}"; do
      set -- $p
      python -m chronofuse.bench stage --split $1 --subtrack $2 --stage "$STAGE"
    done
  fi
}
for r in $R; do STAGE="$r@sel" run_parts; done
STAGE=listwise run_parts
python -m chronofuse.bench report --split dev
python -m chronofuse.bench report --split train
