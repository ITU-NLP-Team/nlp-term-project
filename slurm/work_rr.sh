#!/bin/bash
# usage: work_rr.sh <reranker> "<split>:<sub> [<split>:<sub> ...]"  — resumable via the cache.
source slurm/env.sh
for p in $2; do
  python -m chronofuse.bench stage --split ${p%%:*} --subtrack ${p##*:} --stage $1
done
