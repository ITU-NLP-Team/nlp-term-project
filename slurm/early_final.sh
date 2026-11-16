#!/bin/bash
# Early start of the final steps inside a running allocation (resumable via the cache).
source slurm/env.sh
python -m chronofuse.bench select > logs/select_early.out 2>&1
python -m chronofuse.bench stage --split train dev --subtrack 1a 1b --stage pool_sel
CUDA_VISIBLE_DEVICES=1 python -m chronofuse.bench stage --split train --subtrack 1a --stage q3rr4@sel &
CUDA_VISIBLE_DEVICES=3 python -m chronofuse.bench stage --split dev --subtrack 1a --stage q3rr4@sel &
wait
