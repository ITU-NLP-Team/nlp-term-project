#!/bin/bash
#SBATCH -J cf-llm
#SBATCH -A cip001
#SBATCH -p a100x4q,gpu2dq
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --time=12:00:00
#SBATCH -o logs/llm.out
# LLM query planning with vLLM (Qwen3-8B): plain, then (once every domain's BM25 run exists)
# with BM25 feedback; finally the LLM-derived BM25 channels.
source slurm/env.sh
python -m chronofuse.bench stage --split train dev --subtrack 1a 1b --stage plan
for i in $(seq 1 1440); do
  n=$(ls results/cache/*/train_1a/bm25.json results/cache/*/dev_1a/bm25.json 2>/dev/null | wc -l)
  [ "$n" -ge 26 ] && break; sleep 30
done
python -m chronofuse.bench stage --split train dev --subtrack 1a 1b --stage plan_fb
python -m chronofuse.bench stage --split train dev --subtrack 1a 1b \
  --stage llm_kw llm_q2d llm_multi llm_kw_fb llm_q2d_fb llm_multi_fb
