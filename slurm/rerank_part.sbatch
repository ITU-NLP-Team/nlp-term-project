#!/bin/bash
#SBATCH -A cip001
#SBATCH -p gpu2dq,a100x4q,a100q
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH -o logs/rerank_%x.out
# Usage: sbatch -J <reranker>_<split>_<sub> --export=ALL,RR=<reranker>,SPLIT=<split>,SUB=<sub> slurm/rerank_part.sbatch
source slurm/env.sh
python -m chronofuse.bench stage --split $SPLIT --subtrack $SUB --stage $RR
