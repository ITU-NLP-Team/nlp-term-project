#!/bin/bash
#SBATCH -J cf-domain
#SBATCH -A cip001
#SBATCH -p gpu2dq
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --array=0-12%4
#SBATCH --time=12:00:00
#SBATCH -o logs/domain_%a.out
# Per domain: BM25 passage index + metadata profiles + doc store; lexical / PRF /
# temporal / KG stages; full-corpus dense embeddings, dense retrieval and RAPTOR trees.
source slurm/env.sh
ORDER=(iota history hsm travel genealogy politics bitcoin economics monero cardano workplace law quant)
D=${ORDER[$SLURM_ARRAY_TASK_ID]}
set -e
python -m chronofuse.cli build --config configs/base.yaml --domains $D
python -m chronofuse.bench stage --domains $D --split train dev --subtrack 1a 1b \
  --stage bm25 bm25_y0 bm25_y05 rm3_a03 rm3_a05 rm3_a07 tprf timefilter
python -m chronofuse.bench stage --domains $D --split train dev --subtrack 1b --stage bm25_full
python -m chronofuse.bench stage --domains $D --split train dev --subtrack 1a 1b \
  --stage dense dense_doc raptor raptor_t
python -m chronofuse.bench stage --domains $D --split train dev --subtrack 1a 1b --stage kg
