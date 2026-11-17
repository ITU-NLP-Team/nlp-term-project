#!/bin/bash
#SBATCH -J cf-rir-embed
#SBATCH -A cip001
#SBATCH -p a100x4q,gpu2dq
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --time=24:00:00
#SBATCH -o logs/rir_embed.out
# ReasonIR-8B vectors for every passage of the 13 domains (one worker per visible GPU,
# 20k-passage shards, resumable).
source slurm/env.sh
source slurm/rir_env.sh
[ "$RIR_PY" = FAIL ] && exit 1
N=$(nvidia-smi -L | wc -l)
echo "gpus: $N"
for g in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$RIR_PY python -m chronofuse.retrieval.reasonir embed \
    --worker $g --workers $N > logs/rir_embed_g$g.out 2>&1 &
done
wait
python -m chronofuse.retrieval.reasonir status
