#!/bin/bash
#SBATCH -J cf-dense
#SBATCH -A cip001
#SBATCH -p gpu2dq
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --array=0-12%4
#SBATCH --time=06:00:00
#SBATCH -o logs/dense_%a.out
# Full-corpus passage embeddings (Qwen3-Embedding-0.6B), dense retrieval, RAPTOR trees.
source slurm/env.sh
D=${DOMS[$SLURM_ARRAY_TASK_ID]}
python -m chronofuse.bench stage --domains $D --split train dev --subtrack 1a 1b \
  --stage dense dense_doc raptor raptor_t
