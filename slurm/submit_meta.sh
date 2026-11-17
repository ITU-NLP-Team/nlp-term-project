#!/bin/bash
# Submit the LLM-metadata experiments: document metadata and query side in parallel, then
# the final selection / rerank / report job once both have ended.
set -e
source slurm/env.sh
mkdir -p logs
id() { grep -oE '^[0-9]+' | head -1; }
A=$(sbatch --parsable slurm/meta_docs.sbatch | id)
B=$(sbatch --parsable slurm/meta_query.sbatch | id)
C=$(sbatch --parsable --dependency=afterany:$A:$B slurm/meta_final.sbatch | id)
echo "meta_docs=$A meta_query=$B meta_final=$C"
