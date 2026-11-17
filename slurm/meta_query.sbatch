#!/bin/bash
#SBATCH -J cf-meta-query
#SBATCH -A cip001
#SBATCH -p a100x4q,gpu2dq
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --time=12:00:00
#SBATCH -o logs/meta_query.out
# Query side: small-LLM query metadata (Qwen3-1.7B), neighbouring-event expansion with BM25
# feedback (Qwen3-8B), then the channels that need no document metadata.
source slurm/env.sh
python -m chronofuse.bench stage --split train dev --subtrack 1a 1b --stage qmeta
python -m chronofuse.bench stage --split train dev --subtrack 1a 1b --stage nbr
python -m chronofuse.bench stage --split train dev --subtrack 1a 1b --stage meta_kw nbr_bm25 nbr_hybrid
