#!/bin/bash
#SBATCH -J cf-meta-final
#SBATCH -A cip001
#SBATCH -p a100x4q,gpu2dq
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --time=24:00:00
#SBATCH -o logs/meta_final.out
# After meta_docs + meta_query: finish any missing document metadata, the metadata channels,
# a new train-side selection over all channels (old one kept as *_v1), the selected pool,
# rerankers on its top-100 (one GPU each), and the dev / train reports.
source slurm/env.sh
N=$(nvidia-smi -L | wc -l)
g() { echo $(( $1 % N )); }
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=$i python -m chronofuse.metadata.llm_meta docs --worker $i --workers $N \
    >> logs/meta_docs_g$i.out 2>&1 &
done
wait
python -m chronofuse.metadata.llm_meta status
ALL="--split train dev --subtrack 1a 1b"
python -m chronofuse.bench stage $ALL --stage qmeta nbr meta_kw nbr_bm25 nbr_hybrid  # resumes only
python -m chronofuse.bench stage $ALL --stage meta_field meta_tf_bm25 meta_tf_dense

# keep the previous (v1) selection, reports and selected-pool stages
M=results/metrics
for f in pool_selection bench_dev bench_train bench_dev_perquery bench_train_perquery; do
  [ -f $M/${f}_v1.json ] || cp $M/$f.json $M/${f}_v1.json
done
for f in results/cache/*/*/pool_sel.json results/cache/*/*/*@sel.json; do
  [ -f "$f" ] && [ ! -f "${f%.json}_v1.json" ] && cp "$f" "${f%.json}_v1.json"
done

python -m chronofuse.bench select
python -m chronofuse.bench stage $ALL --stage pool_sel --force
(export CUDA_VISIBLE_DEVICES=$(g 0)
 python -m chronofuse.bench stage $ALL --stage ce_bge@sel --force) > logs/meta_rr0.out 2>&1 &
(export CUDA_VISIBLE_DEVICES=$(g 1)
 python -m chronofuse.bench stage --split train --subtrack 1a 1b --stage q3rr06@sel --force) \
  > logs/meta_rr1.out 2>&1 &
(export CUDA_VISIBLE_DEVICES=$(g 2)
 python -m chronofuse.bench stage --split dev --subtrack 1a 1b --stage q3rr06@sel --force) \
  > logs/meta_rr2.out 2>&1 &
(export CUDA_VISIBLE_DEVICES=$(g 3)
 python -m chronofuse.bench stage --split train dev --subtrack 1a --stage q3rr4@sel --force) \
  > logs/meta_rr3.out 2>&1 &
wait
python -m chronofuse.bench report --split dev
python -m chronofuse.bench report --split train
