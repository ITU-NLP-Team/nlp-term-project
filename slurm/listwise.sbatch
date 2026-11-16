#!/bin/bash
#SBATCH -J cf-listwise
#SBATCH -A cip001
#SBATCH -p gpu2dq,a100x4q
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --time=12:00:00
#SBATCH -o logs/listwise.out
# Select pool/reranker choices on train, then RankGPT-style listwise reranking (Qwen3-8B).
source slurm/env.sh
python -m chronofuse.bench select
python -m chronofuse.bench stage --split train dev --subtrack 1a 1b --stage listwise
python -m chronofuse.bench report --split dev
python -m chronofuse.bench report --split train
