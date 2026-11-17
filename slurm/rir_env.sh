# Pick a transformers that runs ReasonIR's remote code: the env's own, else the pinned 4.47
# copy in $CF/tf447 (ReasonIR requires transformers>=4.47 and imports private modules that
# newer releases moved). Sets RIR_PY for the ReasonIR commands only; vLLM keeps the env's.
if [ -f logs/rir_pythonpath ]; then
  RIR_PY=$(cat logs/rir_pythonpath)
elif python -m chronofuse.retrieval.reasonir smoke > logs/rir_smoke.out 2>&1; then
  RIR_PY=""
elif PYTHONPATH=$CF/tf447 python -m chronofuse.retrieval.reasonir smoke >> logs/rir_smoke.out 2>&1; then
  RIR_PY=$CF/tf447
else
  echo "ReasonIR smoke test failed with both transformers versions (logs/rir_smoke.out)"
  RIR_PY=FAIL
fi
[ "$RIR_PY" != FAIL ] && echo -n "$RIR_PY" > logs/rir_pythonpath
echo "ReasonIR PYTHONPATH: '${RIR_PY}'"
