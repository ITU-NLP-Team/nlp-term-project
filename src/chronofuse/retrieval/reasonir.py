"""ReasonIR-8B dense retrieval over every passage, with optional LLM-reasoning queries.

ReasonIR-8B (Shao et al., 2025) is a bidirectional Llama-3.1-8B encoder trained for
reasoning-intensive retrieval; on TEMPO it gains most from queries expanded with an LLM's
reasoning (27.2 -> 41.0 nDCG@10 with GPT-4o). Here the expansion comes from the planner LLM
(Qwen3-8B, stage ``reason`` in :mod:`chronofuse.bench`).

Corpus encoding is sharded (20k passages per shard, ``data/indexes/<domain>/dense_ReasonIR-8B/
part-XXXXX.{npy,json}``), resumable, and split over GPUs by shard number::

    python -m chronofuse.retrieval.reasonir smoke
    python -m chronofuse.retrieval.reasonir embed --worker 0 --workers 4
    python -m chronofuse.retrieval.reasonir status
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import orjson

from ..config import resolve_path

log = logging.getLogger("chronofuse.reasonir")

MODEL = os.getenv("CF_RIR", "reasonir/ReasonIR-8B")
NAME = MODEL.split("/")[-1]
SHARD = 20_000
DOC_MAX_LEN = 512
QUERY_MAX_LEN = 1024
QUERY_INSTRUCTION = (
    "Given a time-sensitive question from a StackExchange forum, retrieve documents that help "
    "answer the question"
)

REASON_PROMPT = """{query}

Instructions:
1. Identify the essential problem, including the time period and the events it depends on.
2. Think step by step to reason about what information relevant documents would contain \
(earlier and later events, causes, names, dates, mechanisms).
3. Draft an answer with as many thoughts as you have."""


class RIREncoder:
    def __init__(self, batch_size: int = 32):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.m = AutoModel.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        self.m = self.m.to("cuda").eval()
        self.bs = batch_size

    def encode(self, texts: list[str], instruction: str = "", max_length: int = DOC_MAX_LEN):
        """L2-normalised fp16 vectors; texts are length-sorted to limit padding."""
        out = None
        order = np.argsort([-len(t) for t in texts], kind="stable")
        step = self.bs * 16
        with self.torch.inference_mode():
            for i in range(0, len(texts), step):
                idx = order[i : i + step]
                e = self.m.encode(
                    [texts[j] for j in idx],
                    instruction=instruction,
                    batch_size=self.bs,
                    max_length=max_length,
                )
                e = np.asarray(e.float().cpu() if hasattr(e, "cpu") else e, dtype=np.float32)
                e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)
                if out is None:
                    out = np.empty((len(texts), e.shape[1]), dtype=np.float16)
                out[idx] = e.astype(np.float16)
        return out if out is not None else np.zeros((0, 1), np.float16)

    def queries(self, texts: list[str]):
        return self.encode(texts, QUERY_INSTRUCTION, QUERY_MAX_LEN)


# --------------------------------------------------------------- corpus ----
def rir_dir(domain: str) -> Path:
    return resolve_path("data/indexes") / domain / f"dense_{NAME}"


def n_passages(domain: str) -> int:
    p = resolve_path("data/indexes") / domain / "bm25" / "pids.json"
    return len(orjson.loads(p.read_bytes()))


def embed_corpus(domains: list[str], worker: int = 0, workers: int = 1) -> None:
    from ..bench import DomainContext, iter_passages

    enc = None
    base = 0
    for dom in domains:
        n_sh = -(-n_passages(dom) // SHARD)
        todo = {
            s
            for s in range(n_sh)
            if (base + s) % workers == worker and not (rir_dir(dom) / f"part-{s:05d}.npy").exists()
        }
        base += n_sh
        if not todo:
            continue
        ctx = DomainContext(dom)
        buf: list[tuple[str, str]] = []
        for i, (pid, text) in enumerate(iter_passages(ctx)):
            s = i // SHARD
            if s in todo:
                buf.append((pid, text))
            if (i + 1) % SHARD == 0 and buf:
                enc = enc or RIREncoder()
                _save(dom, s, buf, enc)
                buf = []
        if buf:
            enc = enc or RIREncoder()
            _save(dom, (i // SHARD), buf, enc)


def _save(dom: str, s: int, buf: list[tuple[str, str]], enc: RIREncoder) -> None:
    t0 = time.time()
    emb = enc.encode([t for _, t in buf])
    d = rir_dir(dom)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"part-{s:05d}.json").write_bytes(orjson.dumps([p for p, _ in buf]))
    tmp = d / f"part-{s:05d}.tmp.npy"
    np.save(tmp, emb)
    os.replace(tmp, d / f"part-{s:05d}.npy")  # the .npy marks the shard as done
    dt = time.time() - t0
    log.info("[%s] shard %d: %d passages in %.0fs (%.1f/s)", dom, s, len(buf), dt, len(buf) / dt)


def load_corpus_emb(domain: str) -> tuple[list[str], np.ndarray]:
    """All passage vectors of a domain; refuses an incomplete encoding."""
    d = rir_dir(domain)
    parts = sorted(d.glob("part-[0-9][0-9][0-9][0-9][0-9].npy"))
    n_sh = -(-n_passages(domain) // SHARD)
    if len(parts) < n_sh:
        raise RuntimeError(f"ReasonIR encoding incomplete for {domain}: {len(parts)}/{n_sh}")
    pids: list[str] = []
    embs = []
    for p in parts:
        pids += orjson.loads(p.with_suffix(".json").read_bytes())
        embs.append(np.load(p))
    return pids, np.vstack(embs)


def status(domains: list[str]) -> dict:
    out = {}
    for d in domains:
        n_sh = -(-n_passages(d) // SHARD)
        done = len(list(rir_dir(d).glob("part-[0-9][0-9][0-9][0-9][0-9].npy")))
        out[d] = f"{done}/{n_sh}"
    return out


def smoke() -> None:
    """Load the model, encode a query and three documents, print similarities."""
    enc = RIREncoder()
    q = enc.queries(["When did Bitcoin's block size debate lead to a hard fork?"])
    docs = enc.encode(
        [
            "In August 2017 Bitcoin Cash split from Bitcoin after the block size dispute.",
            "SegWit was activated on Bitcoin in August 2017.",
            "The Treaty of Westphalia was signed in 1648.",
        ]
    )
    print("dim", docs.shape[1], "sims", (docs.astype(np.float32) @ q[0].astype(np.float32)))


def main() -> None:
    from ..data.reteco import TRACK1_DOMAINS

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(prog="chronofuse.retrieval.reasonir")
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("embed")
    e.add_argument("--worker", type=int, default=0)
    e.add_argument("--workers", type=int, default=1)
    e.add_argument("--domains", nargs="+", default=["all"])
    s = sp.add_parser("status")
    s.add_argument("--domains", nargs="+", default=["all"])
    sp.add_parser("smoke")
    a = ap.parse_args()
    if a.cmd == "smoke":
        smoke()
        return
    doms = TRACK1_DOMAINS if a.domains == ["all"] else a.domains
    if a.cmd == "embed":
        embed_corpus(doms, a.worker, a.workers)
    else:
        print(json.dumps(status(doms), indent=1))


if __name__ == "__main__":
    main()
    from ..llm.vllm_client import hard_exit

    hard_exit()
