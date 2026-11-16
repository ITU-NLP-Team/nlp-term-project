#!/bin/bash
#SBATCH -J cf-llm2
#SBATCH -A cip001
#SBATCH -p gpu2dq,a100x4q,a100q
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --time=06:00:00
#SBATCH -o logs/llm2.out
# LLM-PRF planning (Qwen3-8B with the top-5 BM25 snippets), then all LLM-derived BM25 channels.
source slurm/env.sh
python -m chronofuse.bench stage --split train dev --subtrack 1a 1b --stage plan plan_fb
python -m chronofuse.bench stage --split train dev --subtrack 1a 1b \
  --stage llm_kw llm_q2d llm_multi llm_kw_fb llm_q2d_fb llm_multi_fb
