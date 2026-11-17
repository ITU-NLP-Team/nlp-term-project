#!/bin/bash
#SBATCH -J cf-final-v2
#SBATCH -A cip001
#SBATCH -p a100x4q,gpu2dq
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --time=24:00:00
#SBATCH -o logs/final_v2.out
# After meta_docs, meta_query and rir_embed: finish anything missing (all resumable), the
# LLM-reasoning and ReasonIR channels, the metadata channels, a new train-side selection over
# all channels (v1 kept as *_v1), the selected pool, rerankers on its top-100, reports.
source slurm/env.sh
N=$(nvidia-smi -L | wc -l)
g() { echo $(( $1 % N )); }
ALL="--split train dev --subtrack 1a 1b"

for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=$i python -m chronofuse.metadata.llm_meta docs --worker $i --workers $N \
    >> logs/meta_docs_g$i.out 2>&1 &
done
wait
python -m chronofuse.metadata.llm_meta status
source slurm/rir_env.sh
if [ "$RIR_PY" != FAIL ]; then
  for i in $(seq 0 $((N - 1))); do
    CUDA_VISIBLE_DEVICES=$i PYTHONPATH=$RIR_PY python -m chronofuse.retrieval.reasonir embed \
      --worker $i --workers $N >> logs/rir_embed_g$i.out 2>&1 &
  done
  wait
fi
python -m chronofuse.retrieval.reasonir status

python -m chronofuse.bench stage $ALL --stage qmeta  # small LLM (own process: GPU memory)
python -m chronofuse.bench stage $ALL --stage nbr reason
python -m chronofuse.bench stage $ALL --stage meta_kw nbr_bm25 nbr_hybrid
python -m chronofuse.bench stage $ALL --stage meta_field meta_tf_bm25 meta_tf_dense
[ "$RIR_PY" != FAIL ] && PYTHONPATH=$RIR_PY python -m chronofuse.bench stage $ALL --stage rir rir_reason

python -m chronofuse.bench select
python -m chronofuse.bench stage $ALL --stage pool_sel --force
(export CUDA_VISIBLE_DEVICES=$(g 0)
 python -m chronofuse.bench stage $ALL --stage ce_bge@sel --force) > logs/v2_rr0.out 2>&1 &
(export CUDA_VISIBLE_DEVICES=$(g 1)
 python -m chronofuse.bench stage --split train --subtrack 1a 1b --stage q3rr06@sel --force) \
  > logs/v2_rr1.out 2>&1 &
(export CUDA_VISIBLE_DEVICES=$(g 2)
 python -m chronofuse.bench stage --split dev --subtrack 1a 1b --stage q3rr06@sel --force) \
  > logs/v2_rr2.out 2>&1 &
(export CUDA_VISIBLE_DEVICES=$(g 3)
 python -m chronofuse.bench stage --split train dev --subtrack 1a --stage q3rr4@sel --force) \
  > logs/v2_rr3.out 2>&1 &
wait
python -m chronofuse.bench report --split dev
python -m chronofuse.bench report --split train
