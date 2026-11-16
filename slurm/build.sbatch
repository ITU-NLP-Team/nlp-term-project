#!/bin/bash
#SBATCH -J cf-build
#SBATCH -A cip001
#SBATCH -p defq
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --array=0-12
#SBATCH --time=04:00:00
#SBATCH -o logs/build_%a.out
# BM25 passage index + metadata profiles + SQLite doc store, then the cheap stages
# (lexical / PRF / temporal / KG) on train and dev, for one domain per array task.
source slurm/env.sh
D=${DOMS[$SLURM_ARRAY_TASK_ID]}
python -m chronofuse.cli build --config configs/base.yaml --domains $D
python -m chronofuse.bench stage --domains $D --split train dev --subtrack 1a 1b \
  --stage bm25 bm25_y0 bm25_y05 rm3_a03 rm3_a05 rm3_a07 tprf timefilter kg
python -m chronofuse.bench stage --domains $D --split train dev --subtrack 1b --stage bm25_full
