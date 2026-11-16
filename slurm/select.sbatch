#!/bin/bash
#SBATCH -J cf-select
#SBATCH -A cip001
#SBATCH -p gpu2dq,a100x4q
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --time=02:00:00
#SBATCH -o logs/select.out
# Forward selection of the first-stage rerank pool on train (RRF over cached channels).
source slurm/env.sh
python -m chronofuse.bench select
python -m chronofuse.bench stage --split train dev --subtrack 1a 1b --stage pool
