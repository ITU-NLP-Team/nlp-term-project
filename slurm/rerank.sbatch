#!/bin/bash
#SBATCH -J cf-rerank
#SBATCH -A cip001
#SBATCH -p gpu2dq,a100x4q
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH -o logs/rerank_%x.out
# Usage: sbatch -J <reranker> slurm/rerank.sbatch   (ce_minilm | ce_bge | q3rr06 | q3rr4)
source slurm/env.sh
python -m chronofuse.bench stage --split train dev --subtrack 1a 1b --stage $SLURM_JOB_NAME
