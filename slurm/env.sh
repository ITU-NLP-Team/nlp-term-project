# Common environment for UHeM Altay jobs. Override CF to relocate.
export CF=${CF:-$HOME/chronofuse}
export HF_HOME=$CF/hf
export PATH=$CF/env/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
DOMS=(bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace)
cd $CF/proj
export VLLM_WORKER_MULTIPROC_METHOD=spawn
