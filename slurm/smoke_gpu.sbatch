#!/bin/bash
#SBATCH -J cf-smoke
#SBATCH -A cip001
#SBATCH -p gpu2dq
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --time=02:00:00
#SBATCH -o logs/smoke_gpu.out
# End-to-end GPU smoke test on the smallest domain, in a separate cache.
source slurm/env.sh
export CF_CACHE=results/cache_smoke
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
python -m chronofuse.bench stage --domains iota --split dev --subtrack 1a 1b \
  --stage bm25 tprf dense dense_doc raptor raptor_t plan plan_fb llm_q2d_fb llm_multi_fb \
          pool ce_minilm ce_bge q3rr06 q3rr4 listwise
