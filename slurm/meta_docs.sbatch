#!/bin/bash
#SBATCH -J cf-meta-docs
#SBATCH -A cip001
#SBATCH -p a100x4q,gpu2dq
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --time=24:00:00
#SBATCH -o logs/meta_docs.out
# Small-LLM metadata (period, events, entities, keywords, topic) for every document of the
# 13 domains with Qwen3-1.7B (vLLM). Nodes are exclusive: one worker per visible GPU, each
# taking every N-th 10k-document shard. Finished shards are skipped, so this resumes.
source slurm/env.sh
N=$(nvidia-smi -L | wc -l)
echo "gpus: $N"
for g in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=$g python -m chronofuse.metadata.llm_meta docs --worker $g --workers $N \
    > logs/meta_docs_g$g.out 2>&1 &
done
wait
python -m chronofuse.metadata.llm_meta status
