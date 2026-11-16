#!/bin/bash
# Submit the post-indexing chain (afterany: a failed stage is logged and skipped, not fatal).
# usage: bash slurm/submit_rest.sh <domain_array_jobid>
cd ${CF:-$HOME/chronofuse}/proj
jid() { grep -o '[0-9]\{5,\}' | tail -1; }
DOM=$1
LLM=$(sbatch slurm/llm.sbatch 2>&1 | jid)
SEL=$(sbatch --dependency=afterany:$DOM:$LLM slurm/select.sbatch 2>&1 | jid)
RR=""
for r in ce_minilm ce_bge q3rr06 q3rr4; do
  RR="$RR:$(sbatch -J $r --dependency=afterany:$SEL slurm/rerank.sbatch 2>&1 | jid)"
done
LW=$(sbatch --dependency=afterany$RR slurm/listwise.sbatch 2>&1 | jid)
echo "llm=$LLM select=$SEL rerank=$RR listwise=$LW"
