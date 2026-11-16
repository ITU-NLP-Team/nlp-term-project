# ChronoFuse

**Hybrid retrieval and LLM reranking for RETECO, SemEval-2027 Task 1, Track 1 (TEMPO).**

ChronoFuse ranks evidence for time-sensitive StackExchange questions (Sub-track 1a) and for
their reasoning steps (1b). It is also a controlled benchmark: on the complete RETECO v1.1 data
(13 domains, 1.17M documents) it measures which retrieval ideas transfer to temporal
reasoning-intensive retrieval: date-based structures, a time-aware RAPTOR tree, LLM query
planning, dense retrieval, and encoder vs. decoder rerankers.

| | |
|---|---|
| Task | [RETECO](https://datascienceuibk.github.io/RETECO/), Sub-tracks **1a** (whole query) and **1b** (step-wise) |
| Data | [`DataScience-UIBK/RETECO-SemEval2027`](https://huggingface.co/datasets/DataScience-UIBK/RETECO-SemEval2027) **v1.1** (latest): 13 domains, 1.17M docs (2.41M passages), 1,730 queries, 3,976 steps |
| Metric | nDCG@10 (`pytrec_eval`), mean per domain, macro-averaged over the 13 domains |
| Result (dev) | **1a 0.371**, **1b 0.341** nDCG@10 (official BM25: 0.115 / 0.118) |
| Paper | ACL format, [`paper/main.pdf`](paper/main.pdf) |

## Results (dev, macro nDCG@10, all 13 domains)

Every choice (RM3 α, prior λ, pool channels and weights, final reranker) is selected on
**train**; dev is only evaluated. Full table with recall@100 and significance:
[`paper/tables/full.tex`](paper/tables/full.tex), raw numbers: `results/metrics/bench_dev.json`.

| System | 1a | 1b |
|---|---:|---:|
| BM25, document level (official baseline) | .115 | .118 |
| BM25, 200-word passages + MaxP | .197 | .223 |
| + temporal prior / year-index filter / temporal KG | ≤ .197 | ≤ .223 |
| LLM-PRF query2doc (Qwen3-8B plan + BM25 feedback) | .242 | .223 |
| Dense, Qwen3-Embedding-0.6B | .297 | .314 |
| Temporal RAPTOR tree | .290 | .299 |
| Train-selected hybrid pool (weighted RRF) | .332 | .325 |
| BM25 ⊕ dense top-100 → Qwen3-Reranker-4B (cross-decoder) | .352 | – |
| **ChronoFuse** (pool ⊕ reranked top-100) | **.371** | **.341** |

Main findings: retrieval granularity (passages) explains most of the gap to the official
baseline; matching dates does not help, because gold documents mention the query's years
*less* often than BM25's non-relevant top-10; LLM feedback beats RM3; decoder LLM rerankers
clearly beat encoder cross-encoders on the same candidates.

## Architecture

```
OFFLINE (per domain)                          ONLINE (per question / step)
documents.jsonl                               question ──► LLM planner (Qwen3-8B, vLLM; optional BM25 feedback)
 ├─ 200-word passages (+ contextual title)      │            intent · year window · event chain · sub-queries
 │   ├─ BM25 (bm25s, streaming builder)         │            keywords · pseudo answer (query2doc)
 │   └─ dense (Qwen3-Embedding-0.6B)            ▼
 ├─ temporal profile, entities, title         channels: BM25, RM3, year-filtered BM25, KG-PPR, dense,
 │   ├─ year index (year → docs)                       RAPTOR / temporal RAPTOR, LLM keywords / q2d / multi-query
 │   ├─ temporal KG (doc–entity–year)          ├─► pool = weighted RRF of channels (forward selection on train)
 │   └─ temporal RAPTOR tree                   └─► BM25 ⊕ dense top-100 → reranker (cross-encoder | Qwen3-Reranker)
 └─ SQLite doc store                                       final = RRF(pool, reranked top-100) → top-10
```

## Project layout

```
.
├── src/chronofuse/
│   ├── bench.py              # staged benchmark: every stage cached per domain/split/sub-track
│   ├── systems.py            # systems over cached stages, train-side selection, reports, significance
│   ├── data/                 # RETECO loaders (queries, steps, guidance, qrels, corpus)
│   ├── metadata/             # temporal extraction, chunking, profiles, year index
│   ├── indexing/             # BM25 (weighted queries, masks, streaming build), dense (FAISS)
│   ├── kg/                   # temporal knowledge graph (personalised PageRank)
│   ├── query/                # rule analyzer, LLM query planner (event chain, multi-query, q2d)
│   ├── llm/                  # vLLM (GPU), llama.cpp (CPU) and OpenAI-compatible clients, disk cache
│   ├── retrieval/            # RRF, RM3 / temporal PRF, temporal prior, temporal RAPTOR
│   ├── rerank/               # cross-encoder, Qwen3-Reranker (cross-decoder), listwise
│   ├── eval/                 # official nDCG@10 scoring, TREC I/O
│   ├── pipeline.py, cli.py   # `chronofuse build | run | evaluate` (index building, E0 runs)
├── slurm/                    # SLURM jobs used on the UHeM Altay A100 cluster
├── scripts/                  # data download, data analysis, paper tables/figures, official scorer
├── tests/                    # pytest unit tests
├── paper/                    # ACL paper (sections/, tables/ and figures/ are generated)
└── results/                  # cache/ (stages), metrics/ (JSON) — git-ignored
```

## Setup

```bash
conda env create -f environment.yml && conda activate reteco && pip install -e .
python scripts/download_data.py                      # RETECO v1.1, all 13 Track 1 domains
```

GPU stages (dense, rerankers, LLM planner) need `torch` with CUDA, `sentence-transformers`,
`transformers` and `vllm==0.8.5.post1`. On a CPU-only machine the same code falls back to
`bge-small`, MiniLM and a local Qwen2.5-1.5B GGUF via `llama-cpp-python`.

## Reproducing the benchmark

```bash
# 1) per domain: BM25 index + temporal profiles + doc store
chronofuse build --config configs/base.yaml --domains all
# 2) stages (each cached under results/cache/<domain>/<split>_<sub>/<stage>.json)
python -m chronofuse.bench stage --split train dev --subtrack 1a 1b \
  --stage bm25 bm25_full bm25_y0 bm25_y05 rm3_a03 rm3_a05 rm3_a07 tprf timefilter kg
python -m chronofuse.bench stage --split train dev --subtrack 1a 1b --stage dense dense_doc raptor raptor_t
python -m chronofuse.bench stage --split train dev --subtrack 1a 1b \
  --stage plan plan_fb llm_kw llm_q2d llm_multi llm_kw_fb llm_q2d_fb llm_multi_fb
python -m chronofuse.bench stage --split train dev --subtrack 1a 1b --stage pool ce_minilm ce_bge q3rr06 q3rr4
# 3) train-side selection, selected pool, reports, paper tables
python -m chronofuse.bench select
python -m chronofuse.bench stage --split train dev --subtrack 1a 1b --stage pool_sel
python -m chronofuse.bench report --split dev && python -m chronofuse.bench report --split train
python scripts/analyze_data.py --split train && python scripts/make_tables.py
```

On the cluster the same steps are the jobs in `slurm/` (`domain.sbatch` per domain, then
`llm2.sbatch`, `rerank.sbatch`, `mega.sbatch`).

## Tests

```bash
make test
```

## Licence and attribution

Code: MIT. RETECO text is CC BY-SA 4.0 (Stack Exchange); annotations are CC BY 4.0.
Please cite TEMPO and RECOR (see `paper/reteco_refs.bib`).
