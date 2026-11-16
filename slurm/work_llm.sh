#!/bin/bash
# LLM planning (plain + BM25-feedback) and the LLM-derived BM25 channels; resumable via the cache.
source slurm/env.sh
python -m chronofuse.bench stage --split train dev --subtrack 1a 1b --stage plan plan_fb
python -m chronofuse.bench stage --split train dev --subtrack 1a 1b \
  --stage llm_kw llm_q2d llm_multi llm_kw_fb llm_q2d_fb llm_multi_fb
