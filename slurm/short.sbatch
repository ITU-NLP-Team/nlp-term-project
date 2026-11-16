#!/bin/bash
#SBATCH -A cip001
#SBATCH -p gpu2dq,a100x4q,a100q
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -o logs/short_%x.out
# Short single-purpose GPU job (fits scheduler backfill).
# usage: sbatch -J <name> -t <time> --export=ALL,CMDS="split:sub:stage[,split:sub:stage...]" slurm/short.sbatch
source slurm/env.sh
IFS=',' read -ra parts <<< "$CMDS"
for p in "${parts[@]}"; do
  IFS=':' read -r split sub stage <<< "$p"
  python -m chronofuse.bench stage --split $split --subtrack $sub --stage "$stage"
done
